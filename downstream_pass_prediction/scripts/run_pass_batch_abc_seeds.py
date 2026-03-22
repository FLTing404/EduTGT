#!/usr/bin/env python3
"""
一键批跑：data_AAA / data_BBB / data_CCC。每个模块抽取 N 个互不重复随机 seed，
E1 与 E2 共用这 N 个 seed（同一学生划分、同一训练随机种子），便于配对比较。

对每个 (module, seed)：
  build_student_splits.py --seed <s>
  run_train_st_pass.py --init scratch --seed <s>
  run_train_st_pass.py --init pretrain --seed <s>   # 不重建 split；缺 pth 时记 no_pretrain_ckpt

汇总 CSV 写入 downstream_pass_prediction/outputs/pass_batch_summary_<timestamp>.csv

用法（在 EduTGT/EduTGT 下，与 main.py 同级）:
  python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py
  python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py --device cuda:0 --no-roc-plot
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNSTREAM_ROOT = REPO_ROOT / "downstream_pass_prediction"
SCRIPTS = DOWNSTREAM_ROOT / "scripts"
DEFAULT_MODULES = ("data_AAA", "data_BBB", "data_CCC")

CSV_FIELDS = [
    "dataset",
    "init",
    "seed",
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


def _empty_row(
    dataset: str,
    init: str,
    seed: int,
    *,
    status: str,
    error: str,
    wall: float = 0.0,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "init": init,
        "seed": seed,
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


def _row_from_rec(rec: Dict[str, Any], status: str = "ok", error: str = "") -> Dict[str, Any]:
    return {
        "dataset": rec.get("module", ""),
        "init": rec.get("init", ""),
        "seed": rec.get("seed", ""),
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AAA/BBB/CCC：每模块 N 个 seed，E1/E2 共用同一组 seed 批跑并汇总 CSV"
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
        help="每个 dataset 抽取的随机 seed 个数；E1/E2 **共用**这些 seed（每 seed 一对 E1+E2）",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--e2-lr", type=float, default=None, help="E2 传给 run_train 的 --lr；默认不传（用 default.yaml 中 lr）")
    ap.add_argument("--e2-freeze-epochs", type=int, default=None, help="E2 冻结编码器热身轮数；默认由 run_train 自行取 5")
    ap.add_argument("--e2-head-lr-factor", type=float, default=None, help="E2 头部 LR 倍数；默认由 run_train 自行取 10.0")
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

    tmp_root = Path(tempfile.mkdtemp(prefix="pass_batch_"))
    try:
        run_i = 0
        for mod in modules:
            pt = _pretrain_path(mod)
            has_pt = pt.is_file()
            if not has_pt and args.strict_pretrain:
                print(f"错误: E2 需要预训练权重但不存在: {pt}", file=sys.stderr)
                sys.exit(1)

            n_seeds = int(args.seeds_per_cell)
            module_seeds = [draw_seed() for _ in range(n_seeds)]

            for seed in module_seeds:
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

                if code_sp != 0:
                    err_txt = err_sp or out_sp
                    for init in ("scratch", "pretrain"):
                        seeds_log.append({"dataset": mod, "init": init, "seed": seed})
                        rows.append(
                            _empty_row(
                                mod,
                                init,
                                seed,
                                status="fail_split",
                                error=err_txt,
                                wall=wall_split,
                            )
                        )
                    continue

                for init in ("scratch", "pretrain"):
                    seeds_log.append({"dataset": mod, "init": init, "seed": seed})

                    if init == "pretrain" and not has_pt:
                        rows.append(
                            _empty_row(
                                mod,
                                init,
                                seed,
                                status="no_pretrain_ckpt",
                                error=f"missing {pt}",
                            )
                        )
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
                        init,
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                        "--dump-run-json",
                        str(dump_path),
                    ]
                    if init == "pretrain":
                        if args.e2_lr is not None:
                            train_cmd.extend(["--lr", str(args.e2_lr)])
                        if args.e2_freeze_epochs is not None:
                            train_cmd.extend(["--freeze-epochs", str(args.e2_freeze_epochs)])
                        if args.e2_head_lr_factor is not None:
                            train_cmd.extend(["--head-lr-factor", str(args.e2_head_lr_factor)])
                    if args.no_roc_plot:
                        train_cmd.append("--no-roc-plot")

                    code, out, err = _run(train_cmd, cwd=REPO_ROOT)
                    wall = time.perf_counter() - t_job0
                    if code != 0:
                        rows.append(
                            _empty_row(
                                mod,
                                init,
                                seed,
                                status="fail_train",
                                error=err or out,
                                wall=wall,
                            )
                        )
                        continue

                    if not dump_path.is_file():
                        rows.append(
                            _empty_row(
                                mod,
                                init,
                                seed,
                                status="no_dump_json",
                                error="dump-run-json 未生成",
                                wall=wall,
                            )
                        )
                        continue

                    try:
                        rec = json.loads(dump_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as e:
                        rows.append(
                            _empty_row(
                                mod,
                                init,
                                seed,
                                status="bad_json",
                                error=str(e),
                                wall=wall,
                            )
                        )
                        continue

                    row = _row_from_rec(rec, status="ok", error="")
                    row["wall_time_sec"] = rec.get("wall_time_sec", round(wall, 3))
                    rows.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)

        log_path = csv_path.with_name(csv_path.stem + "_seeds.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(seeds_log, f, ensure_ascii=False, indent=2)

        print(f"汇总 CSV: {csv_path}")
        print(f"本次各 run 的 seed 记录: {log_path}")
        print(f"共 {len(rows)} 行")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
