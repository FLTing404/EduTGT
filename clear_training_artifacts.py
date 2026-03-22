#!/usr/bin/env python3
"""
一键清空训练产出目录内的文件（不删目录本身）：
  outputs/pretrain/  outputs/checkpoints/  outputs/models/  outputs/middle/

可选：同时删除 outputs/logs/ 下除 .gitkeep 外的文件（--with-logs）。

用法（在 EduTGT/EduTGT 项目根目录，与 main.py 同级）:
  python clear_training_artifacts.py
  python clear_training_artifacts.py -y
  python clear_training_artifacts.py -y --with-logs
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import paths


def _clear_dir_contents(d: str, skip_names: frozenset | None = None) -> int:
    skip_names = skip_names or frozenset()
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
        return -1
    n = 0
    for name in os.listdir(d):
        if name in skip_names:
            continue
        path = os.path.join(d, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
            n += 1
        except OSError as e:
            print(f"错误: 无法删除 {path}: {e}", file=sys.stderr)
            sys.exit(1)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="清空 EduTGT outputs/ 下权重与 checkpoint")
    ap.add_argument("-y", "--yes", action="store_true", help="不询问，直接删除")
    ap.add_argument(
        "--with-logs",
        action="store_true",
        help="同时清空 outputs/logs/（保留 .gitkeep）",
    )
    args = ap.parse_args()

    targets = [
        str(paths.PRETRAIN_DIR),
        str(paths.CHECKPOINTS_DIR),
        str(paths.MODELS_DIR),
        str(paths.MIDDLE_DIR),
    ]
    print("将清空以下目录内的所有文件与子目录：")
    for t in targets:
        print(f"  {t}")
    if args.with_logs:
        print(f"  {paths.LOGS_DIR}（保留 .gitkeep）")
    if not args.yes:
        s = input("确认清空? [y/N] ").strip().lower()
        if s not in ("y", "yes"):
            print("已取消。")
            sys.exit(0)

    for d in targets:
        n = _clear_dir_contents(d)
        if n == -1:
            print(f"[新建空目录] {d}")
        else:
            print(f"已清空 ({n} 项): {d}")

    if args.with_logs:
        n = _clear_dir_contents(str(paths.LOGS_DIR), skip_names=frozenset({".gitkeep"}))
        print(f"已清空 logs（{n} 项，已保留 .gitkeep）: {paths.LOGS_DIR}")

    print("完成。")


if __name__ == "__main__":
    main()
