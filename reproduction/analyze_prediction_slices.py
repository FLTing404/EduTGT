#!/usr/bin/env python3
"""Read saved final predictions and report predeclared slices without fitting.

All event metrics are first calculated per training seed, using that seed's
saved validation-selected decision threshold. Seeds are never pooled as events.
Missing/incomplete inputs remain pending. This program never reads checkpoints,
trains a model, selects a test threshold, or modifies experimental inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


METRICS = ("accuracy", "mcc", "incorrect_recall", "incorrect_f1", "incorrect_ap", "brier")
SLICE_LABELS = {
    "history_position": ("1–10", "11–30", "31+"),
    "train_item_exposure": ("0", "1–20", "21+"),
    "leaf_kc_count": ("单叶 KC", "多叶 KC"),
}
DEFAULT_ROOT = Path(__file__).resolve().parent / "artifacts/new_datasets_dual"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | None]:
    if not len(labels):
        return {name: None for name in METRICS}
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y = labels.astype(bool)
    predicted_correct = probabilities >= threshold
    tp = int(np.sum(y & predicted_correct))
    fp = int(np.sum(~y & predicted_correct))
    tn = int(np.sum(~y & ~predicted_correct))
    fn = int(np.sum(y & ~predicted_correct))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    # Undefined slices are reported as null, not silently treated as success.
    has_both_classes = bool(y.any() and (~y).any())
    return {
        "accuracy": float((tp + tn) / len(y)),
        "mcc": float((tp * tn - fp * fn) / denominator) if denominator else (0.0 if has_both_classes else None),
        "incorrect_recall": float(tn / (tn + fp)) if tn + fp else None,
        "incorrect_f1": float(2 * tn / (2 * tn + fp + fn)) if tn + fp else None,
        "incorrect_ap": float(average_precision_score(~y, 1.0 - probabilities)) if tn + fp else None,
        "brier": float(np.mean((labels - probabilities) ** 2)),
    }


def _expected_targets(frame: pd.DataFrame, users: set[int], sequence: dict) -> pd.DataFrame:
    """Reconstruct only target identities/positions from the saved input protocol."""
    maximum = int(sequence["max_length"])
    minimum = int(sequence["min_length"])
    if maximum <= 0 or minimum < 2:
        raise ValueError("Invalid prepared sequence protocol")
    part = frame[frame["user_idx"].isin(users)]
    group_keys = ["user_idx", "sequence_idx"] if "sequence_idx" in part else ["user_idx"]
    ordering = "event_order" if "event_order" in part else "timestamp"
    part = part.sort_values(group_keys + [ordering], kind="mergesort")
    events: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for _, group in part.groupby(group_keys, sort=False):
        if len(group) < minimum:
            continue
        identifiers = group["event_idx"].to_numpy(np.int64)
        enabled = group["target_valid"].to_numpy(bool) if "target_valid" in group else np.ones(len(group), dtype=bool)
        for start in range(0, len(group) - 1, maximum):
            end = min(start + maximum + 1, len(group))
            if end - start < minimum:
                continue
            valid = enabled[start + 1 : end]
            events.append(identifiers[start + 1 : end][valid])
            positions.append(np.arange(start + 1, end, dtype=np.int64)[valid])
    return pd.DataFrame({
        "event_idx": np.concatenate(events) if events else np.asarray([], dtype=np.int64),
        "expected_history_position": np.concatenate(positions) if positions else np.asarray([], dtype=np.int64),
    })


def load_prepared(root: Path, dataset: str, protocol: dict) -> tuple[pd.DataFrame, dict, dict]:
    configured = protocol.get("config_snapshot", {}).get("prepared_root")
    # Prefer artifact-local inputs so copied artifacts remain portable.
    prepared = root / "prepared" / dataset
    if not prepared.is_dir() and configured:
        prepared = Path(configured) / dataset
    metadata = read_json(prepared / "metadata.json")
    frame = pd.read_parquet(prepared / "interactions.parquet")
    required = {"event_idx", "user_idx", "item_idx", "correct"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Prepared events missing fields: {sorted(required - set(frame.columns))}")
    if frame["event_idx"].duplicated().any():
        raise ValueError("Prepared event_idx must be globally unique")
    with np.load(prepared / "splits.npz", allow_pickle=False) as splits:
        train_users = set(map(int, splits["train"]))
        test_users = set(map(int, splits["test"]))
    sequence = metadata.get("target_protocol", protocol.get("config_snapshot", {}).get("sequence", {}))
    if "max_length" not in sequence or "min_length" not in sequence:
        raise ValueError("Missing saved target max_length/min_length protocol")
    expected = _expected_targets(frame, test_users, sequence)
    if expected["event_idx"].duplicated().any():
        raise ValueError("The prepared protocol generates duplicate target event_idx values")
    stats = metadata.get("split_statistics", {}).get("test", {})
    if "num_targets" in stats and len(expected) != int(stats["num_targets"]):
        raise ValueError("Reconstructed test targets disagree with prepared metadata")
    exposures = frame[frame["user_idx"].isin(train_users)]["item_idx"].value_counts()
    source_columns = ["event_idx", "user_idx", "item_idx", "correct"]
    if "concepts_raw" in frame:
        source_columns.append("concepts_raw")
    targets = expected.merge(frame[source_columns], on="event_idx", how="left", validate="one_to_one")
    targets["train_item_exposure"] = targets["item_idx"].map(exposures).fillna(0).astype(np.int64)
    if "concepts_raw" in targets:
        if targets["concepts_raw"].isna().any():
            raise ValueError("Raw concept labels are missing for target events")
        targets["leaf_kc_count"] = targets["concepts_raw"].map(
            lambda value: len({tag.strip() for tag in str(value).split(";") if tag.strip()})
        )
        if targets["leaf_kc_count"].lt(1).any():
            raise ValueError("Prepared target events contain empty raw concept labels")
    availability = {
        "history_position": None,
        "train_item_exposure": None,
        "leaf_kc_count": None if "concepts_raw" in frame else "prepared events lack concepts_raw; first concept_idx cannot identify multi-leaf events",
    }
    return targets, metadata, availability


def load_run(root: Path, dataset: str, variant: str, seed: int, targets: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    final_path = root / "final" / dataset / f"{variant}_seed{seed}.json"
    final = read_json(final_path)
    result = final.get("result", final)
    for key, expected in (("dataset", dataset), ("variant", variant), ("seed", seed)):
        if key in final and final[key] != expected:
            raise ValueError(f"Saved final {key} differs from requested run identity")
    if "training_seed" in result and int(result["training_seed"]) != seed:
        raise ValueError("Saved final training_seed differs from requested run identity")
    if not result.get("evaluated_test", False):
        raise ValueError("Final run has not evaluated test")
    threshold = result.get("decision_threshold")
    if threshold is None or not np.isfinite(threshold) or not 0 <= float(threshold) <= 1:
        raise ValueError("Missing or invalid saved validation-selected decision threshold")
    local_prediction = root / "predictions" / dataset / f"{variant}_seed{seed}" / "test.npz"
    if not local_prediction.is_file() and result.get("predictions_dir"):
        local_prediction = Path(result["predictions_dir"]) / "test.npz"
    with np.load(local_prediction, allow_pickle=False) as saved:
        required = {"target", "probability", "user_idx", "event_idx", "next_item"}
        if not required.issubset(saved.files):
            raise ValueError(f"Prediction missing fields: {sorted(required - set(saved.files))}")
        columns = required | ({"history_position"} if "history_position" in saved.files else set())
        arrays = {key: np.asarray(saved[key]).reshape(-1) for key in columns}
    if len({len(value) for value in arrays.values()}) != 1:
        raise ValueError("Prediction fields have different lengths")
    prediction = pd.DataFrame(arrays)
    if prediction["event_idx"].duplicated().any():
        raise ValueError("Saved prediction event_idx is not unique")
    if len(prediction) != len(targets) or set(prediction["event_idx"]) != set(targets["event_idx"]):
        raise ValueError("Saved predictions do not cover the complete prepared test target event set")
    joined = prediction.merge(targets, on="event_idx", how="left", validate="one_to_one", suffixes=("", "_prepared"))
    if not np.array_equal(joined["user_idx"], joined["user_idx_prepared"]):
        raise ValueError("Prediction user_idx differs from prepared event identity")
    if not np.array_equal(joined["next_item"], joined["item_idx"]):
        raise ValueError("Prediction next_item differs from prepared event identity")
    if not np.array_equal(joined["target"], joined["correct"]):
        raise ValueError("Prediction target differs from prepared event label")
    if "history_position" in joined and not np.array_equal(joined["history_position"], joined["expected_history_position"]):
        raise ValueError("Saved history_position differs from prepared author segment positions")
    y = joined["target"].to_numpy()
    p = joined["probability"].to_numpy()
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid binary labels or probabilities")
    return joined, float(threshold)


def _mask(frame: pd.DataFrame, family: str, label: str) -> np.ndarray:
    values = frame[family].to_numpy()
    if family == "history_position":
        return (values >= 1) & (values <= 10) if label == "1–10" else ((values >= 11) & (values <= 30) if label == "11–30" else values >= 31)
    if family == "train_item_exposure":
        return values == 0 if label == "0" else ((values >= 1) & (values <= 20) if label == "1–20" else values >= 21)
    return values == 1 if label == "单叶 KC" else values > 1


def summarize(seed_rows: list[dict], expected: list[int]) -> dict:
    complete = [row for row in seed_rows if row["status"] == "complete"]
    unavailable = bool(seed_rows) and all(row["status"] == "unavailable" for row in seed_rows)
    pending = not unavailable and len(complete) != len(expected)
    values = {}
    for name in METRICS:
        numbers = [row["metrics"][name] for row in complete if row["metrics"][name] is not None]
        values[name] = {
            "mean": float(np.mean(numbers)) if numbers and not pending else None,
            "sd": float(np.std(numbers, ddof=1)) if len(numbers) >= 2 and not pending else None,
            "defined_seeds": len(numbers),
        }
    counts = [{key: row[key] for key in ("events", "students", "incorrect_events")} for row in complete]
    counts_consistent = bool(counts) and all(value == counts[0] for value in counts)
    return {
        "status": "unavailable" if unavailable else ("pending" if pending else "complete"), "metrics": values,
        "available_seeds": [row["seed"] for row in complete], "expected_seeds": expected,
        "counts_consistent_across_seeds": counts_consistent,
        "events": counts[0]["events"] if counts_consistent else None,
        "students": counts[0]["students"] if counts_consistent else None,
        "incorrect_events": counts[0]["incorrect_events"] if counts_consistent else None,
    }


def analyze(root: Path, datasets: list[str] | None = None, variants: list[str] | None = None) -> dict[str, Any]:
    try:
        protocol = read_json(root / "protocol.json")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"status": "pending", "artifact_root": str(root.resolve()), "protocol": str(root / "protocol.json"),
                "training_seeds": [], "three_seed_summary": False, "results": [],
                "pending_inputs": [{"path": str(root / "protocol.json"), "reason": str(error)}]}
    seeds = list(map(int, protocol.get("training_seeds", [])))
    if not seeds:
        return {"status": "pending", "artifact_root": str(root.resolve()), "protocol": str(root / "protocol.json"),
                "training_seeds": [], "three_seed_summary": False, "results": [],
                "pending_inputs": [{"path": str(root / "protocol.json"), "reason": "protocol.json lacks predeclared training_seeds"}]}
    datasets = datasets or protocol.get("datasets", [])
    variants = variants or protocol.get("variants", [])
    if not datasets or not variants:
        return {"status": "pending", "artifact_root": str(root.resolve()), "protocol": str(root / "protocol.json"),
                "training_seeds": seeds, "three_seed_summary": len(seeds) == 3, "results": [],
                "pending_inputs": [{"path": str(root / "protocol.json"), "reason": "protocol.json lacks dataset/variant plan"}]}
    output: dict[str, Any] = {
        "artifact_root": str(root.resolve()), "protocol": str(root / "protocol.json"),
        "training_seeds": seeds, "three_seed_summary": len(seeds) == 3,
        "method": {
            "threshold": "each seed's saved final.result.decision_threshold; no test or slice threshold selection",
            "aggregation": "event-weighted metrics within each seed, arithmetic seed mean and sample SD (ddof=1); seeds are not pooled",
            "history_position": "target position 1–10/11–30/31+ within the released author segment, not complete student tenure",
            "train_item_exposure": "training-student event occurrence count 0/1–20/21+; no test labels or test occurrences used",
            "leaf_kc_count": "unique semicolon-separated concepts_raw per prepared target event; raw labels absent from frozen artifacts are optional unavailable/source_not_retained",
            "undefined": "empty slices: all metrics null; no error labels: error Recall/F1/AP null; single label: MCC null; constant predictions with both labels: MCC 0",
            "counts": "events/students counted once per slice per seed; students can occur in multiple slices; slice dimensions overlap",
            "limits": "descriptive post-training slices, no slice-specific calibration or inferential confidence intervals; small slices require caution",
        },
        "results": [], "optional_analyses": [],
    }
    for dataset in datasets:
        prepared_error = None
        try:
            targets, metadata, availability = load_prepared(root, dataset, protocol)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, json.JSONDecodeError) as error:
            prepared_error = f"{type(error).__name__}: {error}"
            targets, metadata, availability = pd.DataFrame(), {}, {}
        if availability.get("leaf_kc_count"):
            output["optional_analyses"].append({"dataset": dataset, "family": "leaf_kc_count", "status": "unavailable",
                                                "reason_code": "source_not_retained", "reason": availability["leaf_kc_count"]})
        for variant in variants:
            runs = {}
            for seed in seeds:
                if prepared_error:
                    runs[seed] = {"error": prepared_error}
                    continue
                try:
                    frame, threshold = load_run(root, dataset, variant, seed, targets)
                    runs[seed] = {"frame": frame, "threshold": threshold}
                except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, json.JSONDecodeError) as error:
                    runs[seed] = {"error": f"{type(error).__name__}: {error}"}
            for family, labels in SLICE_LABELS.items():
                for label in labels:
                    rows = []
                    for seed in seeds:
                        run = runs[seed]
                        if not run.get("error") and availability.get(family):
                            rows.append({"seed": seed, "status": "unavailable", "reason_code": "source_not_retained",
                                         "reason": availability[family]})
                            continue
                        reason = run.get("error")
                        if not reason and family not in run["frame"]:
                            reason = f"saved prediction lacks {family}"
                        if reason:
                            rows.append({"seed": seed, "status": "pending", "reason": reason})
                            continue
                        selected = run["frame"].loc[_mask(run["frame"], family, label)]
                        rows.append({
                            "seed": seed, "status": "complete", "threshold": run["threshold"],
                            "events": int(len(selected)), "students": int(selected["user_idx"].nunique()),
                            "incorrect_events": int(selected["target"].eq(0).sum()),
                            "metrics": _metrics(selected["target"].to_numpy(), selected["probability"].to_numpy(), run["threshold"]),
                        })
                    output["results"].append({"dataset": dataset, "variant": variant, "family": family, "slice": label,
                                              "per_seed": rows, "summary": summarize(rows, seeds)})
    output["status"] = "pending" if any(row["summary"]["status"] == "pending" for row in output["results"]) else "complete"
    output["three_seed_summary"] = len(seeds) == 3 and output["status"] == "complete"
    return output


def _format(summary: dict, metric: str) -> str:
    item = summary["metrics"][metric]
    if item["mean"] is None:
        return "—"
    return f"{item['mean']:.4f} ± {item['sd']:.4f}" if item["sd"] is not None else f"{item['mean']:.4f}（SD 不可用）"


def markdown(report: dict) -> str:
    lines = ["# 预先指定的测试切片", "", f"输入：`{report['artifact_root']}`。预定训练种子：{report['training_seeds']}。", "",
             "每个种子复用其验证集选定的决策阈值，测试集和切片不重新选阈值。指标先逐种子计算，再取均值与样本 SD；单种子不显示伪造的 SD。缺失、不完整或身份不一致的输入标记 pending。", "",
             "历史位置是作者片段内目标位置，不代表完整用户资历。训练题目曝光仅统计训练学生的已保存事件；各维度分别切片，同一学生可以进入多个切片。错误 AP 以错误标签为正类、1−P(correct) 为分数。空切片和无错误标签的相关指标显示为 —。", "",
             "| 数据集 | 模型 | 维度 | 切片 | 状态 | 事件 / 学生 / 错误事件 | Accuracy | MCC | 错误 Recall | 错误 F1 | 错误 AP | Brier |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in report["results"]:
        summary = row["summary"]
        if summary["status"] == "unavailable":
            continue
        counts = " / ".join("—" if summary[key] is None else str(summary[key]) for key in ("events", "students", "incorrect_events"))
        metrics = " | ".join(_format(summary, name) for name in METRICS)
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['family']} | {row['slice']} | {summary['status']} | {counts} | {metrics} |")
    reasons = sorted({run["reason"] for row in report["results"] for run in row["per_seed"] if run["status"] == "pending"}
                     | {row["reason"] for row in report.get("pending_inputs", [])})
    if reasons:
        lines.extend(["", "pending 原因：", ""] + [f"- {reason}" for reason in reasons])
    if report.get("optional_analyses"):
        lines.extend(["", "可选 KC 切片已跳过：原始事件的完整 concepts_raw 没有保存在冻结制品中，不能从第一个 concept_idx 推断单叶/多叶。状态为 unavailable / source_not_retained，核心历史与曝光分析完成状态不受影响。", ""])
        lines.extend(f"- {row['dataset']}：{row['reason']}" for row in report["optional_analyses"])
    lines.extend(["", "这些是描述性切片结果，未进行切片专属调参、校准、假设检验或置信区间估计。种子 SD 只描述训练随机性，不代表学生采样不确定性。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--variants", nargs="+")
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    output_dir = args.output_dir or root / "analysis"
    report = analyze(root, args.datasets, args.variants)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "切片结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "切片结果.md").write_text(markdown(report), encoding="utf-8")
    complete = sum(row["summary"]["status"] == "complete" for row in report["results"])
    pending = sum(row["summary"]["status"] == "pending" for row in report["results"])
    unavailable = sum(row["summary"]["status"] == "unavailable" for row in report["results"])
    print(json.dumps({"output_dir": str(output_dir.resolve()), "complete_slices": complete,
                      "pending_slices": pending, "unavailable_optional_slices": unavailable,
                      "status": report["status"],
                      "training_seeds": report["training_seeds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
