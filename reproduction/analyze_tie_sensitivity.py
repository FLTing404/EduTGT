#!/usr/bin/env python3
"""Describe NIPS minute-tie ordering sensitivity; do not pair ordinal event IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(root: Path, seed: int) -> dict:
    primary = root / "prepared/nips34"
    alternate = root / "sensitivity_prepared/nips34"
    split_checks = {}
    for name in ("splits.npz", "metadata.json"):
        if not (primary / name).exists() or not (alternate / name).exists():
            return {"status": "pending", "reason": "Both ordering preparations are required"}
    a = np.load(primary / "splits.npz")
    b = np.load(alternate / "splits.npz")
    if set(a.files) != set(b.files) or any(not np.array_equal(a[name], b[name]) for name in a.files):
        raise ValueError("Ordering sensitivity must retain identical student splits")
    m0 = json.loads((primary / "metadata.json").read_text())
    m1 = json.loads((alternate / "metadata.json").read_text())
    for name in ("num_users", "num_items", "num_concepts", "num_interactions"):
        if m0[name] != m1[name]:
            raise ValueError(f"Ordering changed dataset size: {name}")
        split_checks[name] = m0[name]
    output = {"status": "complete", "seed": seed, "student_splits_identical": True,
              "size": split_checks, "primary_statistics": m0["split_statistics"],
              "random_tie_statistics": m1["split_statistics"], "variants": {},
              "interpretation": "One seed per ordering; absolute sensitivity only, not a causal result or paired confidence interval. Prepared ordinal event IDs are not source IDs across orderings."}
    names = ("auc", "incorrect_pr_auc", "brier", "ece", "accuracy", "mcc", "incorrect_recall", "incorrect_f1", "cost_miss5_false_alarm1")
    for variant in ("akt", "fchgnn"):
        p0 = root / f"final/nips34/{variant}_seed{seed}.json"
        p1 = root / f"tie_sensitivity/final/nips34/{variant}_seed{seed}.json"
        if not p0.exists() or not p1.exists():
            output["status"] = "pending"
            continue
        r0 = json.loads(p0.read_text())["result"]
        r1 = json.loads(p1.read_text())["result"]
        if not r0.get("evaluated_test") or not r1.get("evaluated_test"):
            raise ValueError("Sensitivity requires final checkpoint test evaluations")
        rows = {}
        for policy in ("fixed_0_5", "f1", "mcc"):
            values0, values1 = r0["test_by_threshold"][policy], r1["test_by_threshold"][policy]
            rows[policy] = {name: {"answer_id": values0[name], "random_tie": values1[name],
                                   "random_minus_answer_id": values1[name] - values0[name]} for name in names}
        output["variants"][variant] = {"primary_validation_auc": r0["validation"]["auc"],
                                       "random_validation_auc": r1["validation"]["auc"],
                                       "thresholds": {"answer_id": r0["comparison_thresholds"], "random_tie": r1["comparison_thresholds"]},
                                       "test": rows}
    return output


def report(summary: dict) -> str:
    lines = ["# NIPS 同分钟排序敏感性", "", f"状态：{summary['status']}。", "",
             "NIPS2020 Task 3/4 的 DateAnswered 只有分钟精度，68.015% 的事件处于同学生同分钟的并列组。主实验按 AnswerId 排序，敏感性实验在同分钟内固定种子随机重排；两者均无法恢复真实先后顺序。",
             "本检查固定学生划分、相同训练种子和预算，AKT 与基础 FC-HGNN 各增加一次训练。模型仍分别用自己的验证集选 checkpoint 和阈值；表中是随机并列排序减 AnswerId 排序的绝对差值。单种子差值不表示统计显著性，也不能排除训练优化和片段边界变化的影响。", "",
             "| 模型 | 指标 | AnswerId | 随机并列 | 差值 |", "|---|---|---:|---:|---:|"]
    for variant, row in summary.get("variants", {}).items():
        for name, values in row["test"]["mcc"].items():
            lines.append(f"| {variant} / MCC阈值 | {name} | {values['answer_id']:.5f} | {values['random_tie']:.5f} | {values['random_minus_answer_id']:+.5f} |")
    lines.extend(["", "两种排序中学生集合和总交互规模相同；每个序列片段首条只提供历史，不计作预测目标，随机排序可能改变被排除事件及目标类别计数。按行分配的 event_idx 不能跨排序作为原始事件键，因此本检查不构造事件配对区间。",
                  "即使指标差值很小，也不能证明同分钟内使用先前答案符合真实时间因果关系。严格在线 KT 后续需要分钟块内不互相使用标签的预测协议，或具有更精确时间戳的数据。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "artifacts/new_datasets_dual")
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = analyze(root, args.seed)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "排序敏感性.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "排序敏感性.md").write_text(report(summary))
    print(json.dumps({"status": summary["status"], "variants": list(summary.get("variants", {}))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
