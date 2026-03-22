#!/usr/bin/env python3
"""按 id_student 分层划分 train/val/test，写入下游 data 目录。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except ImportError as e:
    raise SystemExit("需要 sklearn：pip install scikit-learn") from e

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG / "src") not in sys.path:
    sys.path.insert(0, str(_PKG / "src"))

from paths_downstream import module_data_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", type=str, required=True, help="如 data_AAA")
    ap.add_argument("--seed", type=int, default=60)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument(
        "--val-size-of-trainval",
        type=float,
        default=None,
        help="在 train+val 子集上第二次划分的 test_size；默认 0.15/0.85",
    )
    args = ap.parse_args()

    mod_dir = module_data_dir(args.module)
    labels_path = mod_dir / "labels_pass.csv"
    if not labels_path.is_file():
        raise SystemExit(f"请先运行 build_pass_labels.py，缺少 {labels_path}")

    labels = pd.read_csv(labels_path)
    if labels.empty:
        raise SystemExit("labels_pass.csv 为空")

    # 每个学生一个分层标签：任一条通过则视为 1（便于 stratify）
    stu_y = labels.groupby("id_student", as_index=False)["is_pass"].max()
    stu_y = stu_y.rename(columns={"is_pass": "strat_y"})

    val_ratio = args.val_size_of_trainval
    if val_ratio is None:
        val_ratio = args.test_size / (1.0 - args.test_size)

    try:
        tv, te = train_test_split(
            stu_y,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stu_y["strat_y"],
        )
        tr, va = train_test_split(
            tv,
            test_size=val_ratio,
            random_state=args.seed,
            stratify=tv["strat_y"],
        )
    except ValueError as e:
        raise SystemExit(
            f"分层划分失败（类别样本过少？）：{e}\n可改用更大数据或减少 split 比例。"
        ) from e

    splits = mod_dir / "splits"
    splits.mkdir(parents=True, exist_ok=True)

    def write_list(name: str, df: pd.DataFrame) -> None:
        p = splits / name
        with open(p, "w", encoding="utf-8") as f:
            for sid in sorted(df["id_student"].astype(int).unique()):
                f.write(f"{int(sid)}\n")

    write_list("train_students.txt", tr)
    write_list("val_students.txt", va)
    write_list("test_students.txt", te)
    print(f"train={len(tr)} val={len(va)} test={len(te)} -> {splits}")


if __name__ == "__main__":
    main()
