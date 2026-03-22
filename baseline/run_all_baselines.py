#!/usr/bin/env python3
"""
一键顺序运行 TGN / TGAT / JODIE（tncn 目录）三个基线，共用同一 data_dir 与 utils.Dataset 划分。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_ROOT)


def _load_result(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip_tgn", action="store_true")
    p.add_argument("--skip_tgat", action="store_true")
    p.add_argument("--skip_tncn", action="store_true", help="跳过第三基线（目录 tncn/，实现为 JODIE 风格）")
    args = p.parse_args()

    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.normpath(os.path.join(_PROJ, data_dir))
    name = os.path.basename(os.path.normpath(data_dir))
    out_root = os.path.join(_ROOT, "results", name)
    os.makedirs(out_root, exist_ok=True)

    common = [
        "--data_dir",
        data_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]

    jobs: List[tuple[str, str, List[str]]] = []
    if not args.skip_tgn:
        jobs.append(
            (
                "tgn",
                os.path.join(out_root, "tgn"),
                [sys.executable, os.path.join(_ROOT, "tgn", "runner.py")] + common + ["--out_dir", os.path.join(out_root, "tgn")],
            )
        )
    if not args.skip_tgat:
        jobs.append(
            (
                "tgat",
                os.path.join(out_root, "tgat"),
                [sys.executable, os.path.join(_ROOT, "tgat", "runner.py")] + common + ["--out_dir", os.path.join(out_root, "tgat")],
            )
        )
    if not args.skip_tncn:
        jobs.append(
            (
                "jodie",
                os.path.join(out_root, "tncn"),
                [sys.executable, os.path.join(_ROOT, "tncn", "runner.py")] + common + ["--out_dir", os.path.join(out_root, "tncn")],
            )
        )

    rows = []
    if not jobs:
        print("未选择任何任务（全部 skip）。", file=sys.stderr)
        sys.exit(1)

    for tag, sub, cmd in jobs:
        print("=" * 60)
        print("运行:", tag, "\n命令:", " ".join(cmd))
        t0 = time.time()
        env = os.environ.copy()
        env["PYTHONPATH"] = _PROJ + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.check_call(cmd, cwd=_PROJ, env=env)
        elapsed = time.time() - t0
        res_path = os.path.join(sub, "result.json")
        rec = _load_result(res_path)
        rows.append(
            {
                "model": rec.get("model", tag),
                "val_auc": rec.get("val_auc"),
                "val_ap": rec.get("val_ap"),
                "test_auc": rec.get("test_auc"),
                "test_ap": rec.get("test_ap"),
                "test_acc": rec.get("test_acc"),
                "nn_test_auc": rec.get("nn_test_auc"),
                "nn_test_ap": rec.get("nn_test_ap"),
                "nn_test_acc": rec.get("nn_test_acc"),
                "train_time_sec": rec.get("train_time_sec", elapsed),
                "best_epoch": rec.get("best_epoch", 0),
            }
        )

    _SUMMARY_FIELDS = [
        "model",
        "val_auc",
        "val_ap",
        "test_auc",
        "test_ap",
        "test_acc",
        "nn_test_auc",
        "nn_test_ap",
        "nn_test_acc",
        "train_time_sec",
        "best_epoch",
    ]
    summary_csv = os.path.join(out_root, "summary.csv")
    summary_json = os.path.join(out_root, "summary.json")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if rows:
            w.writeheader()
            w.writerows(rows)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print("\n汇总 (summary.csv / summary.json):\n")
    if rows:
        hdr = _SUMMARY_FIELDS
        print(" | ".join(hdr))
        for r in rows:
            print(" | ".join(str(r[k]) for k in hdr))
    print(f"\n已写入: {summary_csv}\n{summary_json}")


if __name__ == "__main__":
    main()
