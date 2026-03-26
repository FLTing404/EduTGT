#!/usr/bin/env python3
"""
一键批跑：data_AAA / data_BBB / data_CCC。每个模块抽取 N 个互不重复随机 seed，
E1（scratch）每个 seed 跑 1 次；E2（pretrain）默认对同一 seed 跑 **多组超参**（内置网格），
CSV 中 `e2_preset` 区分，便于按验证/测试指标挑最优。

对每个 (module, seed)：
  build_student_splits.py --seed <s>
  run_train_st_pass.py --init scratch --seed <s>
  对每个 E2 预设：run_train_st_pass.py --init pretrain --seed <s> + 预设对应 CLI

汇总 CSV 写入 downstream_pass_prediction/outputs/pass_batch_summary_<timestamp>.csv

用法（在 EduTGT/EduTGT 下，与 main.py 同级）:
  python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py
  python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py --device cuda:0 --no-roc-plot
  # 只跑一组 E2（与旧版一致，超参来自 yaml / 下方可选 --e2-lr 等）:
  python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py --e2-once
  # 自定义多组（JSON 数组，每项可含 preset/name/lr/freeze_epochs/head_lr_factor/encoder_lr_scale/patience）:
  python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py --e2-presets-json path/to/presets.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNSTREAM_ROOT = REPO_ROOT / "downstream_pass_prediction"
SCRIPTS = DOWNSTREAM_ROOT / "scripts"
DEFAULT_MODULES = ("data_AAA", "data_BBB", "data_CCC")

# 默认 E2（单组，不做网格）
E2_BUILTIN_PRESETS: List[Dict[str, Any]] = [
    {
        "preset": "fixed_lr5e-5_f10_h5_enc0.6_p15",
        "lr": 5e-5,
        "freeze_epochs": 10,
        "head_lr_factor": 5.0,
        "encoder_lr_scale": 0.6,
        "patience": 15,
    }
]

CSV_FIELDS = [
    "dataset",
    "init",
    "seed",
    "e2_preset",
    "lr",
    "encoder_lr_scale",
    "patience",
    "wall_time_sec",
    "test_acc",
    "test_auroc",
    "test_auprc",
    "test_f1",
    "epochs_ran",
    "ctx_sample",
    "tmp_sample",
    "freeze_epochs",
    "head_lr_factor",
    "status",
    "error",
]


def _env() -> dict:
    e = os.environ.copy()
    e["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + e.get("PYTHONPATH", "")
    return e


def _pretrain_path(module: str) -> Path:
    return REPO_ROOT / "outputs" / "pretrain" / f"{module}.pth"


def _run(cmd: List[str], *, cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out, p.stderr or ""


def _preset_display_name(p: Dict[str, Any], index: int) -> str:
    return str(p.get("preset") or p.get("name") or f"p{index}")


def _preset_to_train_flags(p: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    if p.get("lr") is not None:
        flags.extend(["--lr", str(float(p["lr"]))])
    if p.get("freeze_epochs") is not None:
        flags.extend(["--freeze-epochs", str(int(p["freeze_epochs"]))])
    if p.get("head_lr_factor") is not None:
        flags.extend(["--head-lr-factor", str(float(p["head_lr_factor"]))])
    if p.get("encoder_lr_scale") is not None:
        flags.extend(["--encoder-lr-scale", str(float(p["encoder_lr_scale"]))])
    if p.get("patience") is not None:
        flags.extend(["--e2-patience", str(int(p["patience"]))])
    return flags


def _load_e2_presets(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.e2_once and args.e2_presets_json:
        print("错误: 不能同时指定 --e2-once 与 --e2-presets-json", file=sys.stderr)
        sys.exit(1)
    if args.e2_once:
        one: Dict[str, Any] = {"preset": "once"}
        if args.e2_lr is not None:
            one["lr"] = args.e2_lr
        if args.e2_freeze_epochs is not None:
            one["freeze_epochs"] = args.e2_freeze_epochs
        if args.e2_head_lr_factor is not None:
            one["head_lr_factor"] = args.e2_head_lr_factor
        if args.e2_encoder_lr_scale is not None:
            one["encoder_lr_scale"] = args.e2_encoder_lr_scale
        if args.e2_patience is not None:
            one["patience"] = args.e2_patience
        return [one]
    if args.e2_presets_json:
        path = Path(args.e2_presets_json)
        if not path.is_file():
            print(f"错误: --e2-presets-json 文件不存在: {path}", file=sys.stderr)
            sys.exit(1)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"错误: JSON 解析失败: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(raw, list) or not raw:
            print("错误: --e2-presets-json 须为非空 JSON 数组", file=sys.stderr)
            sys.exit(1)
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                print(f"错误: presets[{i}] 不是 JSON 对象", file=sys.stderr)
                sys.exit(1)
            out.append(dict(item))
        return out
    return [deepcopy(x) for x in E2_BUILTIN_PRESETS]


def _empty_row(
    dataset: str,
    init: str,
    seed: int,
    *,
    e2_preset: str = "",
    status: str,
    error: str,
    wall: float = 0.0,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "init": init,
        "seed": seed,
        "e2_preset": e2_preset,
        "lr": "",
        "encoder_lr_scale": "",
        "patience": "",
        "wall_time_sec": round(wall, 3) if wall else "",
        "test_acc": "",
        "test_auroc": "",
        "test_auprc": "",
        "test_f1": "",
        "epochs_ran": "",
        "ctx_sample": "",
        "tmp_sample": "",
        "freeze_epochs": "",
        "head_lr_factor": "",
        "status": status,
        "error": error[:500] if error else "",
    }


def _row_from_rec(
    rec: Dict[str, Any],
    *,
    e2_preset: str = "",
    status: str = "ok",
    error: str = "",
) -> Dict[str, Any]:
    row = {
        "dataset": rec.get("module", ""),
        "init": rec.get("init", ""),
        "seed": rec.get("seed", ""),
        "e2_preset": e2_preset or rec.get("e2_preset", ""),
        "lr": rec.get("lr", ""),
        "encoder_lr_scale": rec.get("encoder_lr_scale", ""),
        "patience": rec.get("patience", ""),
        "wall_time_sec": rec.get("wall_time_sec", ""),
        "test_acc": rec.get("test_acc", ""),
        "test_auroc": rec.get("test_auroc", ""),
        "test_auprc": rec.get("test_auprc", ""),
        "test_f1": rec.get("test_f1", ""),
        "epochs_ran": rec.get("epochs_ran", ""),
        "ctx_sample": rec.get("ctx_sample", ""),
        "tmp_sample": rec.get("tmp_sample", ""),
        "freeze_epochs": rec.get("freeze_epochs", ""),
        "head_lr_factor": rec.get("head_lr_factor", ""),
        "status": status,
        "error": error[:500] if error else "",
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AAA/BBB/CCC：每模块 N 个 seed；E1×1 + E2×多组超参（默认内置网格），汇总 CSV"
    )
    ap.add_argument(
        "--modules",
        type=str,
        default=",".join(DEFAULT_MODULES),
        help="逗号分隔 module 名，如 data_AAA,data_BBB",
    )
    ap.add_argument(
        "--seeds-per-cell",
        type=int,
        default=3,
        help="每个 dataset 抽取的随机 seed 个数；E1 与每组 E2 **共用**这些 seed",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--e2-once",
        action="store_true",
        help="E2 每个 seed 只跑 1 次（超参来自 yaml，可用下方 --e2-lr 等覆盖）",
    )
    ap.add_argument(
        "--e2-presets-json",
        type=str,
        default=None,
        help="E2 预设列表 JSON 路径（数组）；与 --e2-once 互斥",
    )
    ap.add_argument(
        "--e2-lr",
        type=float,
        default=None,
        help="仅 --e2-once：传给 run_train 的 --lr",
    )
    ap.add_argument(
        "--e2-freeze-epochs",
        type=int,
        default=None,
        help="仅 --e2-once：--freeze-epochs",
    )
    ap.add_argument(
        "--e2-head-lr-factor",
        type=float,
        default=None,
        help="仅 --e2-once：--head-lr-factor",
    )
    ap.add_argument(
        "--e2-encoder-lr-scale",
        type=float,
        default=None,
        help="仅 --e2-once：--encoder-lr-scale",
    )
    ap.add_argument(
        "--e2-patience",
        type=int,
        default=None,
        help="仅 --e2-once：--e2-patience",
    )
    ap.add_argument("--no-roc-plot", action="store_true", help="批跑时不写 ROC PNG，加快运行")
    ap.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="汇总 CSV 路径；默认 outputs/pass_batch_summary_<时间戳>.csv",
    )
    ap.add_argument(
        "--strict-pretrain",
        action="store_true",
        help="E2 所需 outputs/pretrain/<module>.pth 缺失时立即退出；默认改为跳过并记 status=no_pretrain_ckpt",
    )
    args = ap.parse_args()

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    if not modules:
        print("错误: --modules 为空", file=sys.stderr)
        sys.exit(1)

    e2_presets = _load_e2_presets(args)
    n_e2 = len(e2_presets)

    out_dir = DOWNSTREAM_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.out_csv) if args.out_csv else out_dir / f"pass_batch_summary_{tag}.csv"
    if not csv_path.is_absolute():
        csv_path = (REPO_ROOT / csv_path).resolve()

    used_seeds: set = set()

    def draw_seed() -> int:
        while True:
            s = secrets.randbelow(2**31)
            if s not in used_seeds:
                used_seeds.add(s)
                return s

    rows: List[Dict[str, Any]] = []
    seeds_log: List[Dict[str, Any]] = []
    pbar = None

    tmp_root = Path(tempfile.mkdtemp(prefix="pass_batch_"))
    try:
        run_i = 0
        print(
            f"E2 每组 seed 将跑 {n_e2} 个超参配置"
            + (" (--e2-once)" if args.e2_once else "")
            + (f"（自定义 JSON {n_e2} 条）" if args.e2_presets_json else "（内置网格）"),
            file=sys.stderr,
        )
        n_seeds = int(args.seeds_per_cell)
        total_jobs = len(modules) * n_seeds * (2 + n_e2)  # split + scratch + 每组 E2
        pbar = tqdm(total=total_jobs, desc="batch", unit="job", dynamic_ncols=True)

        for mod in modules:
            pt = _pretrain_path(mod)
            has_pt = pt.is_file()
            if not has_pt and args.strict_pretrain:
                print(f"错误: E2 需要预训练权重但不存在: {pt}", file=sys.stderr)
                sys.exit(1)

            module_seeds = [draw_seed() for _ in range(n_seeds)]

            for seed in module_seeds:
                pbar.set_description(f"{mod} seed={seed} split")
                t_split0 = time.perf_counter()
                split_cmd = [
                    sys.executable,
                    str(SCRIPTS / "build_student_splits.py"),
                    "--module",
                    mod,
                    "--seed",
                    str(seed),
                ]
                code_sp, out_sp, err_sp = _run(split_cmd, cwd=REPO_ROOT)
                wall_split = time.perf_counter() - t_split0
                pbar.update(1)

                if code_sp != 0:
                    err_txt = err_sp or out_sp
                    seeds_log.append({"dataset": mod, "init": "scratch", "seed": seed, "e2_preset": ""})
                    rows.append(
                        _empty_row(
                            mod,
                            "scratch",
                            seed,
                            e2_preset="",
                            status="fail_split",
                            error=err_txt,
                            wall=wall_split,
                        )
                    )
                    for pi, preset in enumerate(e2_presets):
                        ptag = _preset_display_name(preset, pi)
                        seeds_log.append(
                            {"dataset": mod, "init": "pretrain", "seed": seed, "e2_preset": ptag}
                        )
                        rows.append(
                            _empty_row(
                                mod,
                                "pretrain",
                                seed,
                                e2_preset=ptag,
                                status="fail_split",
                                error=err_txt,
                                wall=wall_split,
                            )
                        )
                    continue

                # E1 scratch（一次）
                pbar.set_description(f"{mod} seed={seed} E1")
                seeds_log.append({"dataset": mod, "init": "scratch", "seed": seed, "e2_preset": ""})
                t_job0 = time.perf_counter()
                dump_path = tmp_root / f"rec_{run_i}.json"
                run_i += 1
                train_cmd = [
                    sys.executable,
                    str(SCRIPTS / "run_train_st_pass.py"),
                    "--module",
                    mod,
                    "--init",
                    "scratch",
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--dump-run-json",
                    str(dump_path),
                ]
                if args.no_roc_plot:
                    train_cmd.append("--no-roc-plot")

                code, out, err = _run(train_cmd, cwd=REPO_ROOT)
                wall = time.perf_counter() - t_job0
                if code != 0:
                    rows.append(
                        _empty_row(
                            mod,
                            "scratch",
                            seed,
                            status="fail_train",
                            error=err or out,
                            wall=wall,
                        )
                    )
                elif not dump_path.is_file():
                    rows.append(
                        _empty_row(
                            mod,
                            "scratch",
                            seed,
                            status="no_dump_json",
                            error="dump-run-json 未生成",
                            wall=wall,
                        )
                    )
                else:
                    try:
                        rec = json.loads(dump_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as e:
                        rows.append(
                            _empty_row(
                                mod,
                                "scratch",
                                seed,
                                status="bad_json",
                                error=str(e),
                                wall=wall,
                            )
                        )
                    else:
                        row = _row_from_rec(rec, e2_preset="", status="ok", error="")
                        row["wall_time_sec"] = rec.get("wall_time_sec", round(wall, 3))
                        rows.append(row)
                pbar.update(1)

                # E2：split 成功则始终尝试（与 scratch 成败无关）
                for pi, preset in enumerate(e2_presets):
                    ptag = _preset_display_name(preset, pi)
                    pbar.set_description(f"{mod} seed={seed} E2={ptag}")
                    seeds_log.append(
                        {"dataset": mod, "init": "pretrain", "seed": seed, "e2_preset": ptag}
                    )

                    if not has_pt:
                        rows.append(
                            _empty_row(
                                mod,
                                "pretrain",
                                seed,
                                e2_preset=ptag,
                                status="no_pretrain_ckpt",
                                error=f"missing {pt}",
                            )
                        )
                        pbar.update(1)
                        continue

                    t_job0 = time.perf_counter()
                    dump_path = tmp_root / f"rec_{run_i}.json"
                    run_i += 1
                    train_cmd = [
                        sys.executable,
                        str(SCRIPTS / "run_train_st_pass.py"),
                        "--module",
                        mod,
                        "--init",
                        "pretrain",
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                        "--dump-run-json",
                        str(dump_path),
                    ]
                    train_cmd.extend(_preset_to_train_flags(preset))
                    if args.no_roc_plot:
                        train_cmd.append("--no-roc-plot")

                    code, out, err = _run(train_cmd, cwd=REPO_ROOT)
                    wall = time.perf_counter() - t_job0
                    if code != 0:
                        rows.append(
                            _empty_row(
                                mod,
                                "pretrain",
                                seed,
                                e2_preset=ptag,
                                status="fail_train",
                                error=err or out,
                                wall=wall,
                            )
                        )
                        pbar.update(1)
                        continue

                    if not dump_path.is_file():
                        rows.append(
                            _empty_row(
                                mod,
                                "pretrain",
                                seed,
                                e2_preset=ptag,
                                status="no_dump_json",
                                error="dump-run-json 未生成",
                                wall=wall,
                            )
                        )
                        pbar.update(1)
                        continue

                    try:
                        rec = json.loads(dump_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as e:
                        rows.append(
                            _empty_row(
                                mod,
                                "pretrain",
                                seed,
                                e2_preset=ptag,
                                status="bad_json",
                                error=str(e),
                                wall=wall,
                            )
                        )
                        pbar.update(1)
                        continue

                    row = _row_from_rec(rec, e2_preset=ptag, status="ok", error="")
                    row["wall_time_sec"] = rec.get("wall_time_sec", round(wall, 3))
                    rows.append(row)
                    pbar.update(1)

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)

        log_path = csv_path.with_name(csv_path.stem + "_seeds.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(seeds_log, f, ensure_ascii=False, indent=2)

        presets_dump = csv_path.with_name(csv_path.stem + "_e2_presets.json")
        with open(presets_dump, "w", encoding="utf-8") as f:
            json.dump(e2_presets, f, ensure_ascii=False, indent=2)

        print(f"汇总 CSV: {csv_path}")
        print(f"本次各 run 的 seed 记录: {log_path}")
        print(f"本次 E2 超参列表（与 CSV 中 e2_preset 对应）: {presets_dump}")
        print(f"共 {len(rows)} 行")
    finally:
        if pbar is not None:
            pbar.close()
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
