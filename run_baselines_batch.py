#!/usr/bin/env python3
"""
一键对比算法批跑脚本（TGN / TGAT / JODIE × 指定 seed → CSV）

前置条件：
  - 已安装 baseline 依赖：pip install -r baseline/requirements-baseline.txt
  - 数据目录 data/processed/data_<DATASET>/ 已存在

用法（在 EduTGT/EduTGT 下）：
  python run_baselines_batch.py --dataset AAA
  python run_baselines_batch.py --dataset CCC --seed 42
  python run_baselines_batch.py --dataset AAA --seed 2026 --device cpu --epochs 50

输出：
  outputs/result/baselines_<DATASET>_<时间戳>.csv

注意：
  - 每次只跑一个 seed（--seed），三个基线共用同一 seed。
  - 若需多 seed，可多次执行此脚本（或手动合并 CSV）。
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
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "outputs" / "result"

DATASET_MODULES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")

CSV_FIELDS = [
    "dataset",
    "method",
    "seed",
    "pres_mix_uniform",
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

BASELINE_SCRIPTS = [
    ("tgn",   "baseline/tgn/runner.py"),
    ("tgat",  "baseline/tgat/runner.py"),
    ("jodie", "baseline/tncn/runner.py"),
]


def _env() -> dict:
    e = os.environ.copy()
    e["PYTHONPATH"] = str(ROOT) + os.pathsep + e.get("PYTHONPATH", "")
    return e


def _row_from_result_json(
    data: Dict[str, Any],
    dataset: str,
    method: str,
    seed: int,
    elapsed: float,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "pres_mix_uniform": "",
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


def _fail_row(
    dataset: str, method: str, seed: int, elapsed: float, msg: str
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "pres_mix_uniform": "",
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


def _run_baseline(
    tag: str,
    script: str,
    data_dir: Path,
    seed: int,
    dataset: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> Dict[str, Any]:
    run_root = ROOT / "outputs" / "result" / "_runs" / f"{data_dir.name}_s{seed}"
    out_d = run_root / tag
    out_d.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / script),
        "--data_dir", str(data_dir),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--seed", str(seed),
        "--device", device,
        "--out_dir", str(out_d),
    ]

    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, cwd=str(ROOT), env=_env(), check=True)
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - t0
        return _fail_row(dataset, tag, seed, elapsed, f"exit_{e.returncode}")

    elapsed = time.perf_counter() - t0
    result_json = out_d / "result.json"
    if result_json.is_file():
        with open(result_json, encoding="utf-8") as f:
            data = json.load(f)
        return _row_from_result_json(data, dataset, tag, seed, elapsed)
    return _fail_row(dataset, tag, seed, elapsed, "no_result_json")


def main() -> None:
    p = argparse.ArgumentParser(
        description="一键对比算法批跑：TGN / TGAT / JODIE × 指定 seed → CSV"
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_MODULES,
        help="整模块代码，对应 data/processed/data_<MODULE>",
    )
    p.add_argument("--seed", type=int, default=42, help="随机种子（单个整数），默认 42")
    p.add_argument("--epochs", type=int, default=50, help="训练轮数，默认 50")
    p.add_argument("--batch_size", type=int, default=200, help="batch size，默认 200")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率，默认 0.001")
    p.add_argument("--device", default="cuda", help="cuda 或 cpu，默认 cuda")
    p.add_argument(
        "--out_csv",
        default=None,
        help="输出 CSV 路径，默认 outputs/result/baselines_<DATASET>_<时间戳>.csv",
    )
    args = p.parse_args()

    data_name = f"data_{args.dataset}"
    data_dir = ROOT / "data" / "processed" / data_name
    if not data_dir.is_dir():
        print(f"错误：数据目录不存在 {data_dir}", file=sys.stderr)
        sys.exit(1)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.out_csv)
        if args.out_csv
        else RESULT_DIR / f"baselines_{args.dataset}_{tag}.csv"
    )
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()

    total = len(BASELINE_SCRIPTS)
    print(
        f"\n[run_baselines_batch] 数据集={args.dataset}  seed={args.seed}  共 {total} 个基线\n"
        f"  输出 CSV → {out_path}\n",
        file=sys.stderr,
    )

    rows: List[Dict[str, Any]] = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()

        for i, (method_tag, script) in enumerate(BASELINE_SCRIPTS, 1):
            print(f"[{i}/{total}] {method_tag}  seed={args.seed} ...", file=sys.stderr)
            row = _run_baseline(
                tag=method_tag,
                script=script,
                data_dir=data_dir,
                seed=args.seed,
                dataset=args.dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )
            rows.append(row)
            writer.writerow(row)
            f.flush()
            status = row.get("status", "?")
            auc = row.get("AUC", "")
            print(f"    → status={status}  AUC={auc}", file=sys.stderr)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(
        f"\n[run_baselines_batch] 完成 {ok}/{total}  CSV → {out_path}\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
