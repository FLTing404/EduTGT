#!/usr/bin/env python3
"""根据 node_map + stats 生成 pres_relation.json（站点 id_site Jaccard），供 pretrain / 消融读取。"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# data/scripts -> 项目根 EduTGT/EduTGT
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from course_pres import save_pres_relation_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="写出 pres_relation.json")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="含 node_map.csv、stats.json 的目录，如 data/processed/data_AAA",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出路径，默认 <data_dir>/pres_relation.json",
    )
    args = p.parse_args()
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.normpath(os.path.join(_ROOT, data_dir))
    out = save_pres_relation_json(data_dir, args.out)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
