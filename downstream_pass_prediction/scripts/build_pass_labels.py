#!/usr/bin/env python3
"""
从 raw OULAD studentInfo（final_result）与 processed 图生成 labels_pass.csv，
写入 downstream_pass_prediction/data/data_<MODULE>/，不修改 data/processed。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 包根：downstream_pass_prediction/
_PKG = Path(__file__).resolve().parents[1]
if str(_PKG / "src") not in sys.path:
    sys.path.insert(0, str(_PKG / "src"))

from io_data import (  # noqa: E402
    TS_PRES_BLOCK,
    find_content_csv,
    find_ml_csv,
    load_stats,
    student_to_ml_node,
)
from labels import load_labels_from_student_info  # noqa: E402
from paths_downstream import REPO_ROOT, module_data_dir, processed_dir  # noqa: E402


def rel_repo(p: Path) -> str:
    return str(p.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 is_pass 标签表与 manifest.json")
    ap.add_argument(
        "--module",
        type=str,
        required=True,
        help="与 data/processed 下目录名一致，如 data_AAA、data_BBB …",
    )
    ap.add_argument(
        "--raw-oulad",
        type=str,
        default=None,
        help="覆盖默认 data/raw/OULAD",
    )
    args = ap.parse_args()
    module = args.module.strip()
    proc = processed_dir(module)
    if not proc.is_dir():
        raise SystemExit(f"不存在 processed 目录: {proc}")

    stats = load_stats(proc)
    code_module = str(stats["code_module"])
    presentations = list(stats["code_presentations_merged"])

    nm = proc / "node_map.csv"
    if not nm.is_file():
        raise SystemExit(f"缺少 node_map.csv: {nm}")

    stu2nid = student_to_ml_node(nm)
    student_ids = set(stu2nid.keys())

    raw_root = Path(args.raw_oulad) if args.raw_oulad else REPO_ROOT / "data" / "raw" / "OULAD"
    if not raw_root.is_dir():
        raise SystemExit(f"raw OULAD 目录不存在: {raw_root}")
    student_info = raw_root / "studentInfo.csv"
    if not student_info.is_file():
        raise SystemExit(f"缺少 studentInfo.csv: {student_info}")

    labels = load_labels_from_student_info(
        student_info, code_module, presentations, student_ids, stu2nid
    )
    if labels.empty:
        raise SystemExit("无标签行（检查 module / raw 数据 / 图学生交集）")

    out_dir = module_data_dir(module)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels_pass.csv"
    labels.to_csv(labels_path, index=False)

    ml_path = find_ml_csv(proc)
    content_path = find_content_csv(proc, ml_path.name)

    manifest = {
        "module": module,
        "processed_dir": rel_repo(proc),
        "ml_csv": rel_repo(ml_path),
        "content_csv": rel_repo(content_path),
        "node_map_csv": rel_repo(nm),
        "stats_json": rel_repo(proc / "stats.json"),
        "labels_csv": rel_repo(labels_path),
        "student_info_csv": rel_repo(student_info),
        "ts_pres_block": TS_PRES_BLOCK,
        "num_label_rows": int(len(labels)),
    }
    man_path = out_dir / "manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"写入 {labels_path} ({len(labels)} 行)")
    print(f"写入 {man_path}")


if __name__ == "__main__":
    main()
