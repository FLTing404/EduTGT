#!/usr/bin/env python3
"""
临时批量实验辅助脚本（不改业务代码）：
  对选定数据集（AAA / BBB）依次跑 main、两个消融、三个 baseline，每种组合可指定多个 seed，
  将指标汇总到 outputs/result/ 下的 CSV。

说明（务必阅读）：
  - main.py 与两个消融使用 get_args() 的 --seed（子进程传入），与 init_seeds(args.seed) 一致；CSV 中 seed 列与随机性对应。
  - TGN / TGAT / tncn(JODIE) 的 runner 同样使用 --seed。
  - 不执行预训练；请自行保证 outputs/pretrain/<data_name>.pth 已存在。

用法（在 EduTGT/EduTGT 下）:
  python run_batch_metrics_csv.py --dataset AAA
  python run_batch_metrics_csv.py --dataset BBB --seeds 42,1,2 --device cpu

详细说明见文档: docs/run_batch_metrics_csv.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 项目根（与 main.py 同级）
ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "outputs" / "result"
JSONL_PATH = ROOT / "outputs" / "logs" / "training_runs.jsonl"

CSV_FIELDS = [
    "dataset",
    "method",
    "seed",
    "AUC",
    "ACC",
    "AP",
    "Loss",
    "epoch",
    "time_cost",
    "nn_test_auc",
    "nn_test_ap",
    "nn_test_acc",
    "status",
]


def _env() -> dict:
    e = os.environ.copy()
    e["PYTHONPATH"] = str(ROOT) + os.pathsep + e.get("PYTHONPATH", "")
    return e


def _jsonl_byte_offset() -> int:
    if not JSONL_PATH.is_file():
        return 0
    return JSONL_PATH.stat().st_size


def _parse_new_jsonl_record(start_byte: int) -> Optional[Dict[str, Any]]:
    """读取自 start_byte 起追加的内容，取最后一条完整 JSON 行。"""
    if not JSONL_PATH.is_file():
        return None
    with open(JSONL_PATH, "rb") as f:
        f.seek(start_byte)
        chunk = f.read().decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _row_from_jsonl(rec: Dict[str, Any], dataset: str, method: str, seed: int, elapsed: float) -> Dict[str, Any]:
    mt = rec.get("metrics", {})
    test = mt.get("test", {})
    nn = mt.get("nn_test", {})
    tr = rec.get("training", {})
    ep = tr.get("best_epoch")
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "AUC": test.get("auc", ""),
        "ACC": test.get("acc", ""),
        "AP": test.get("ap", ""),
        "Loss": test.get("loss", ""),
        "epoch": ep if ep is not None else "",
        "time_cost": round(elapsed, 3),
        "nn_test_auc": nn.get("auc", ""),
        "nn_test_ap": nn.get("ap", ""),
        "nn_test_acc": nn.get("acc", ""),
        "status": "ok",
    }


def _row_from_baseline_json(
    data: Dict[str, Any], dataset: str, method: str, seed: int, elapsed: float
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "AUC": data.get("test_auc", ""),
        "ACC": data.get("test_acc", ""),
        "AP": data.get("test_ap", ""),
        "Loss": "",
        "epoch": data.get("best_epoch", ""),
        "time_cost": round(elapsed, 3),
        "nn_test_auc": data.get("nn_test_auc", ""),
        "nn_test_ap": data.get("nn_test_ap", ""),
        "nn_test_acc": data.get("nn_test_acc", ""),
        "status": "ok",
    }


def _fail_row(dataset: str, method: str, seed: int, elapsed: float, msg: str) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "AUC": "",
        "ACC": "",
        "AP": "",
        "Loss": "",
        "epoch": "",
        "time_cost": round(elapsed, 3),
        "nn_test_auc": "",
        "nn_test_ap": "",
        "nn_test_acc": "",
        "status": msg[:200],
    }


def _run(
    cmd: List[str],
    *,
    dataset: str,
    method: str,
    seed: int,
    use_jsonl: bool,
    result_json: Optional[Path],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    off = _jsonl_byte_offset() if use_jsonl else 0
    try:
        subprocess.run(cmd, cwd=str(ROOT), env=_env(), check=True)
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - t0
        return _fail_row(dataset, method, seed, elapsed, f"exit_{e.returncode}")

    elapsed = time.perf_counter() - t0
    if use_jsonl:
        rec = _parse_new_jsonl_record(off)
        if not rec:
            return _fail_row(dataset, method, seed, elapsed, "no_jsonl_record")
        return _row_from_jsonl(rec, dataset, method, seed, elapsed)

    if result_json and result_json.is_file():
        with open(result_json, encoding="utf-8") as f:
            data = json.load(f)
        return _row_from_baseline_json(data, dataset, method, seed, elapsed)
    return _fail_row(dataset, method, seed, elapsed, "no_result_json")


def main() -> None:
    p = argparse.ArgumentParser(description="批量跑 main/消融/baseline 并写 CSV（不跑预训练）")
    p.add_argument("--dataset", type=str, required=True, choices=("AAA", "BBB"), help="整模块 data_AAA 或 data_BBB")
    p.add_argument("--seeds", type=str, default="42,1,2", help="逗号分隔，如 42,1,2")
    p.add_argument("--n_epoch", type=int, default=50)
    p.add_argument("--bs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ctx_sample", type=int, default=40)
    p.add_argument("--tmp_sample", type=int, default=31)
    p.add_argument("--drop_out", type=float, default=0.2)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--lambda_tc", type=float, default=0.1, help="仅 ablation_stc")
    p.add_argument("--pres_mix_uniform", type=float, default=0.35, help="仅 ablation_pres")
    p.add_argument("--baseline_epochs", type=int, default=50)
    p.add_argument("--baseline_bs", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda", help="baseline 用：cuda 或 cpu")
    p.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="输出 CSV 路径，默认 outputs/result/batch_<dataset>_<timestamp>.csv",
    )
    args = p.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        print("错误: --seeds 为空", file=sys.stderr)
        sys.exit(1)

    data_name = f"data_{args.dataset}"
    data_dir = ROOT / "data" / "processed" / data_name
    if not data_dir.is_dir():
        print(f"错误: 数据目录不存在: {data_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        from paths import pretrain_ckpt_path

        pt = pretrain_ckpt_path(data_name)
        if not pt.is_file():
            print(f"警告: 未找到预训练权重 {pt}，main/消融将报错；请先自行 pretrain。", file=sys.stderr)
    except ImportError:
        pass

    print(
        "提示: main/消融与 TGN/TGAT/JODIE 均通过子进程 --seed 控制随机性，与 CSV 中 seed 列一致。\n",
        file=sys.stderr,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    batch_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_csv) if args.out_csv else RESULT_DIR / f"batch_{args.dataset}_{batch_tag}.csv"
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()

    def _main_train_args(seed: int) -> List[str]:
        return [
            "--data_dir",
            str(data_dir),
            "--n_epoch",
            str(args.n_epoch),
            "--bs",
            str(args.bs),
            "--lr",
            str(args.lr),
            "--ctx_sample",
            str(args.ctx_sample),
            "--tmp_sample",
            str(args.tmp_sample),
            "--drop_out",
            str(args.drop_out),
            "--gpu",
            str(args.gpu),
            "--seed",
            str(seed),
        ]

    rows: List[Dict[str, Any]] = []

    jobs: List[Tuple[str, int, List[str], bool, Optional[Path]]] = []

    for seed in seeds:
        jobs.append(
            (
                "main",
                seed,
                [sys.executable, str(ROOT / "main.py"), *_main_train_args(seed)],
                True,
                None,
            )
        )
        jobs.append(
            (
                "ablation_pres",
                seed,
                [
                    sys.executable,
                    str(ROOT / "main_ablation_pres_relation.py"),
                    *_main_train_args(seed),
                    "--pres_mix_uniform",
                    str(args.pres_mix_uniform),
                ],
                True,
                None,
            )
        )
        jobs.append(
            (
                "ablation_stc",
                seed,
                [
                    sys.executable,
                    str(ROOT / "main_ablation_student_temporal.py"),
                    "--lambda_tc",
                    str(args.lambda_tc),
                    *_main_train_args(seed),
                ],
                True,
                None,
            )
        )

        run_root = RESULT_DIR / "_runs" / f"{data_name}_s{seed}"
        run_root.mkdir(parents=True, exist_ok=True)
        for tag, script in (
            ("tgn", "baseline/tgn/runner.py"),
            ("tgat", "baseline/tgat/runner.py"),
            ("jodie", "baseline/tncn/runner.py"),
        ):
            out_d = run_root / tag
            out_d.mkdir(parents=True, exist_ok=True)
            cmd_b = [
                sys.executable,
                str(ROOT / script),
                "--data_dir",
                str(data_dir),
                "--epochs",
                str(args.baseline_epochs),
                "--batch_size",
                str(args.baseline_bs),
                "--lr",
                str(args.lr),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--out_dir",
                str(out_d),
            ]
            jobs.append((tag, seed, cmd_b, False, out_d / "result.json"))

    print(f"共 {len(jobs)} 次运行，结果 -> {out_path}\n")

    for method, seed, cmd, use_jsonl, rjson in jobs:
        print(">>>", method, "seed", seed)
        print("   ", " ".join(cmd[:6]), "...")
        row = _run(cmd, dataset=args.dataset, method=method, seed=seed, use_jsonl=use_jsonl, result_json=rjson)
        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n已写入 {out_path}（{len(rows)} 行）")


if __name__ == "__main__":
    main()
