#!/usr/bin/env python3
"""Control for train-only item accuracy, using existing prepared difficulty.npy.

This script trains no model and changes no source/prepared files. A static item
probability is computed once. Its own F1/MCC thresholds use validation labels
only, and the graph variant comes from the pre-test frozen validation selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from analyze_new_datasets import (
    LABELS,
    PredictionCache,
    _canonical,
    _clean_json,
    _format,
    _prediction_path,
    _threshold_metrics,
    _validation_auc,
    aggregate_metrics,
    assert_aligned,
    load_prediction,
    load_records,
    paired_cluster_bootstrap,
)


ROOT = Path(__file__).resolve().parent
POLICIES = ("fixed_0_5", "f1", "mcc")
METRICS = ("auc", "incorrect_pr_auc", "brier", "mcc", "incorrect_f1", "incorrect_recall")


def static_prediction(reference: dict[str, np.ndarray], difficulty: np.ndarray) -> dict[str, np.ndarray]:
    """Reuse the exact prepared item vector, including its unseen-item 0.5 prior."""
    if "next_item" not in reference:
        raise ValueError("Saved predictions must include next_item to identify the predicted question")
    items = np.asarray(reference["next_item"])
    if not np.issubdtype(items.dtype, np.integer) or (items < 0).any() or (items >= len(difficulty)).any():
        raise ValueError("next_item must contain valid integer indices into difficulty.npy")
    # Subtract in the native dtype: this is exactly the prepared model feature's
    # complement, without re-estimating, smoothing, or accessing test labels.
    probabilities = (1.0 - difficulty)[items]
    result = dict(reference, probability=probabilities)
    PredictionCache(result, with_ranking=False)
    return result


def assert_same_target_items(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    assert_aligned(left, right)
    if "next_item" not in left or "next_item" not in right or not np.array_equal(left["next_item"], right["next_item"]):
        raise ValueError("Static-prior alignment failed for next_item; predicted question order must be identical")


def validation_thresholds(prediction: dict[str, np.ndarray]) -> dict[str, float]:
    # Only this pure threshold function is called; importing train does not run
    # fitting. Reuse the same exact candidate enumeration/tie rule as all runs.
    sys.path.insert(0, str(ROOT / "src"))
    from fchgnn_repro.train import best_threshold

    return {
        "fixed_0_5": 0.5,
        "f1": best_threshold(prediction["target"], prediction["probability"], "f1", "exact"),
        "mcc": best_threshold(prediction["target"], prediction["probability"], "mcc", "exact"),
    }


def analyze_static(root: Path, repetitions: int = 200, prepared_root: Path | None = None) -> dict:
    protocol_path = root / "protocol.json"
    if not protocol_path.exists():
        raise ValueError(f"Missing protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    snapshot = protocol.get("config_snapshot", {})
    datasets = list(protocol["datasets"])
    seeds = list(map(int, protocol.get("training_seeds", protocol.get("seeds", []))))
    if not seeds:
        raise ValueError("The protocol must identify the fixed graph training seeds")
    if prepared_root is None:
        configured = snapshot.get("prepared_root", protocol.get("prepared_root"))
        prepared_root = Path(configured) if configured else root / "prepared"
        if configured and not prepared_root.is_absolute():
            prepared_root = root / prepared_root
    frozen_path = root / "frozen_plan.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else {}
    screen, final = load_records(root, "screen"), load_records(root, "final")
    scope = "smoke" if "smoke" in root.name.lower() or snapshot.get("training", {}).get("epochs", 8) <= 1 else "controlled_experiment"
    report = {
        "status": "partial", "scope": scope, "root": str(root), "prepared_root": str(prepared_root),
        "protocol_path": str(protocol_path), "frozen_plan_path": str(frozen_path) if frozen else None,
        "graph_training_seeds": seeds, "static_training_runs": 0,
        "definition": "p_correct(item) = 1 - prepared difficulty.npy; train-unseen items keep the prepared 0.5 prior",
        "static_prediction_is_deterministic": True,
        "threshold_selection": "static prior has its own exact validation-only correct-F1/MCC thresholds; graph thresholds retained from its validation runs",
        "comparison_direction": "selected graph minus static item prior",
        "datasets": {},
    }
    completed = 0
    for dataset in datasets:
        block: dict = {"status": "pending", "static_training_runs": 0, "static_distinct_prediction_arrays_per_split": 1}
        report["datasets"][dataset] = block
        selected = frozen.get("selections", {}).get(dataset, {}).get("best_hypergraph")
        if not selected:
            block["reason"] = "Waiting for frozen validation-only hypergraph selection"
            continue
        if not frozen.get("frozen_before_test"):
            raise ValueError("Static-prior comparison requires the hypergraph selection to be frozen before test")
        variant = _canonical(selected)
        if variant not in ("fchgnn", "enhanced"):
            raise ValueError(f"Frozen selection is not an eligible hypergraph: {selected}")
        block.update(selected_hypergraph=selected, selection_uses="frozen validation AUC mean only", graph_training_seed_count=len(seeds))
        # Verify the frozen winner against saved validation metrics. Tests never
        # enter this check. Exact ties retain the runner's frozen choice.
        candidates = []
        for candidate in ("fchgnn", "enhanced"):
            rows = [screen.get((dataset, candidate, seed)) for seed in seeds]
            if all(row is not None for row in rows):
                candidates.append((candidate, float(np.mean([_validation_auc(row) for row in rows]))))
        chosen = next((value for name, value in candidates if name == variant), None)
        if chosen is None:
            block["reason"] = "Waiting for every selected graph validation record"
            continue
        if not math.isclose(chosen, max(value for _, value in candidates), rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"Frozen graph does not match validation-only selection for {dataset}")
        block["selected_hypergraph_validation_auc_mean"] = chosen
        valid_graph, test_graph, graph_records = [], [], []
        missing = []
        for seed in seeds:
            key = dataset, variant, seed
            if key not in screen or key not in final or not final[key]["result"].get("test_by_threshold"):
                missing.append(f"Selected graph final run for seed {seed}")
                continue
            valid_path = _prediction_path(root, screen[key], "valid")
            test_path = _prediction_path(root, final[key], "test")
            if not valid_path.exists() or not test_path.exists():
                missing.extend(str(path) for path in (valid_path, test_path) if not path.exists())
                continue
            valid_graph.append(load_prediction(valid_path))
            test_graph.append(load_prediction(test_path))
            graph_records.append(final[key])
        if missing:
            block.update(reason="Waiting for aligned selected graph predictions", missing=missing)
            continue
        for valid, test in zip(valid_graph, test_graph):
            assert_same_target_items(valid_graph[0], valid)
            assert_same_target_items(test_graph[0], test)
        difficulty_path = prepared_root / dataset / "difficulty.npy"
        difficulty = np.load(difficulty_path, allow_pickle=False)
        if difficulty.ndim != 1 or not np.isfinite(difficulty).all() or ((difficulty < 0) | (difficulty > 1)).any():
            raise ValueError(f"Malformed prepared item difficulty: {difficulty_path}")
        metadata_path = prepared_root / dataset / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["num_items"] != len(difficulty):
            raise ValueError(f"Prepared metadata item count disagrees with difficulty vector: {dataset}")
        # The following two arrays are the ONLY static predictions. Repeating a
        # reference for paired seed averaging does not create new training runs.
        static_valid = static_prediction(valid_graph[0], difficulty)
        static_test = static_prediction(test_graph[0], difficulty)
        thresholds = validation_thresholds(static_valid)
        static_tables = {
            split: {policy: PredictionCache(prediction, threshold).metrics(requested=METRICS)
                    for policy, threshold in thresholds.items()}
            for split, prediction in (("validation", static_valid), ("test", static_test))
        }
        graph_tables = {policy: aggregate_metrics([
            _threshold_metrics(record["result"], "test")[policy] for record in graph_records
        ]) for policy in POLICIES}
        graph_thresholds_by_policy = {
            policy: [float(screen[(dataset, variant, seed)]["result"]["comparison_thresholds"][policy]) for seed in seeds]
            for policy in POLICIES
        }
        point_differences = {
            policy: {metric: graph_tables[policy][metric]["mean"] - static_tables["test"][policy][metric] for metric in METRICS}
            for policy in POLICIES
        }
        # Check saved test records against saved prediction arrays at their own
        # validation thresholds. This detects mismatched artifact generations.
        graph_thresholds = []
        for record, prediction in zip(graph_records, test_graph):
            seed = int(record["seed"])
            threshold = float(screen[(dataset, variant, seed)]["result"]["comparison_thresholds"]["mcc"])
            graph_thresholds.append(threshold)
            computed = PredictionCache(prediction, threshold).metrics(requested=METRICS)
            recorded = _threshold_metrics(record["result"], "test")["mcc"]
            for metric in METRICS:
                if not np.isclose(computed[metric], recorded[metric], rtol=1e-7, atol=1e-8, equal_nan=True):
                    raise ValueError(f"Saved probability/metric mismatch for {dataset}/{selected}/seed {seed}/{metric}")
        comparison = paired_cluster_bootstrap(
            test_graph, [static_test] * len(seeds), graph_thresholds, [thresholds["mcc"]] * len(seeds),
            repetitions=repetitions, requested=METRICS,
        )
        comparison.update({
            "graph_training_seeds": seeds, "static_training_runs": 0,
            "static_distinct_prediction_arrays": 1,
            "right_prediction_reused_for_each_graph_seed": True,
            "interpretation": "Resample test students; average fixed graph-seed metrics minus the same deterministic static prior. Static model has no seed variance.",
        })
        block.update({
            "status": "complete", "prepared_metadata": metadata,
            "difficulty_path": str(difficulty_path), "difficulty_sha256": hashlib.sha256(difficulty_path.read_bytes()).hexdigest(),
            "static_thresholds": thresholds,
            "static_validation_by_threshold": static_tables["validation"], "static_test_by_threshold": static_tables["test"],
            "graph_test_by_threshold": graph_tables,
            "graph_minus_static_by_threshold": point_differences,
            "graph_thresholds_per_seed": graph_thresholds_by_policy,
            "graph_mcc_thresholds_per_seed": graph_thresholds,
            "paired_student_bootstrap_mcc_policy": comparison,
            "alignment": {"fields": ["target", "user_idx", "event_idx", "next_item"],
                          "valid_events": len(static_valid["target"]), "test_events": len(static_test["target"]),
                          "test_students": len(np.unique(static_test["user_idx"]))},
        })
        completed += 1
    report["completed_datasets"] = completed
    report["planned_datasets"] = len(datasets)
    report["status"] = "complete" if completed == len(datasets) else "partial"
    return report


def markdown(report: dict) -> str:
    smoke = report["scope"] == "smoke"
    lines = [
        "# 训练题目先验与超图模型对照", "",
        f"状态：{'已完成' if report['status'] == 'complete' else '阶段性结果，尚未完成'}；{report['completed_datasets']}/{report['planned_datasets']} 套数据完成对照。",
        "范围：技术烟雾检查，仅验证统计管线，不能作为正式模型结论。" if smoke else "范围：固定划分下的静态先验控制，作为已冻结模型实验的补充。",
        "静态概率精确取已有 prepared/difficulty.npy 的 1 − difficulty：训练题目经验正确率，训练未见题保留原定义的 0.5。没有重新估计、平滑或使用验证/测试标签计算概率；F1/MCC 阈值仅在验证目标上选择。先验只有一份确定预测，没有训练种子方差。", "",
        "超图候选来自测试前冻结的验证集跨种子平均 AUC 选择。以下‘超图 − 静态’是相同事件的差值；超图表为固定训练种子的均值 ± 样本标准差，先验表直接报告确定数值。AUC、错误 AP、MCC、错误 F1/Recall 越大越好，Brier 越小越好。", "",
    ]
    for dataset, block in report["datasets"].items():
        lines.extend([f"## {dataset}", ""])
        if block["status"] != "complete":
            lines.extend([f"待完成：{block.get('reason', '等待冻结计划与预测')}。", ""])
            continue
        variant = _canonical(block["selected_hypergraph"])
        lines.extend([
            f"选定超图：{LABELS.get(variant, variant)}；固定训练种子数 {block['graph_training_seed_count']}。对齐 {block['alignment']['valid_events']:,} 个验证目标、{block['alignment']['test_events']:,} 个测试目标和 {block['alignment']['test_students']:,} 名测试学生。",
            "静态先验概率与预测数组数量为 1（每个 split），未将它假称为三次独立训练。", "",
            "| 策略/方法 | 阈值 | AUC | 错误 AP | Brier | MCC | 错误 F1 | 错误 Recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for policy in POLICIES:
            static = block["static_test_by_threshold"][policy]
            graph = block["graph_test_by_threshold"][policy]
            static_numbers = " | ".join(f"{static[metric]:.5f}" for metric in METRICS)
            graph_numbers = " | ".join(_format(graph[metric], 5) for metric in METRICS)
            thresholds = np.asarray(block["graph_thresholds_per_seed"][policy])
            graph_threshold = f"{thresholds.mean():.5f} ± {(thresholds.std(ddof=1) if len(thresholds) > 1 else 0.0):.5f}"
            lines.append(f"| {policy} / 静态题目先验 | {block['static_thresholds'][policy]:.5f} | {static_numbers} |")
            lines.append(f"| {policy} / 选定超图 | {graph_threshold} | {graph_numbers} |")
            delta = block["graph_minus_static_by_threshold"][policy]
            lines.append(f"| {policy} / 超图 − 静态 | — | " + " | ".join(f"{delta[metric]:+.5f}" for metric in METRICS) + " |")
        lines.extend([
            "", "配对学生 bootstrap（两者分别使用各自验证集选择的 MCC 阈值）：", "",
            "| 指标 | 超图 − 静态平均差 | 95% 区间 | 有效重复 |", "|---|---:|---:|---:|",
        ])
        comparison = block["paired_student_bootstrap_mcc_policy"]
        for metric in METRICS:
            values = comparison["metrics"][metric]
            lo, hi = values["ci95"]
            lines.append(f"| {metric} | {values['mean_difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {values['valid_repetitions']} |")
        lines.append("")
    lines.extend([
        "## 解释", "",
        "每次等概率抽取测试学生，保留其全部作答，所有固定超图 seed 与同一份静态先验共享学生抽样，再平均各 seed 的指标差。静态先验重复作为右侧引用只是配对计算，不是增加独立运行。95% 百分位区间条件于训练模型与固定划分，不包含重新训练、划分或阈值拟合的不确定性。",
        "此对照用于判断训练题目难度信息本身能解释多少预测能力。若静态先验接近或超过超图，不能把相对无难度模型的涨点全归为知识状态追踪或超图传播；若超图更好，也只能说明该协议下整体模型超过静态题目基线，不能单凭此对照识别哪个模块起作用。超图特定贡献仍看增强与 noHG 的信息匹配消融。",
        "这是主实验方案启动后补充的明确标注对照：只用已冻结 graph，不改变模型选择、训练、超参数或原测试阈值。XES 的样本/作者片段限制、ASSIST17 动作口径、NIPS 同分钟伪时序限制仍适用。", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/new_datasets_dual")
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be positive")
    root = args.root.resolve()
    report = analyze_static(root, args.bootstrap_reps, args.prepared_root)
    output = args.output_dir.resolve() if args.output_dir else root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "题目先验对照.json").write_text(json.dumps(_clean_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "题目先验对照.md").write_text(markdown(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "scope": report["scope"], "completed_datasets": report["completed_datasets"],
                      "planned_datasets": report["planned_datasets"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
