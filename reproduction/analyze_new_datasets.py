#!/usr/bin/env python3
"""Read saved runs; compare frozen models with paired student-cluster bootstrap.

The bootstrap resamples students uniformly, retains every event from each drawn
student, and averages metric differences across the fixed training seeds. It does
not resample seeds, pool them as independent observations, or choose thresholds
on a resampled test set. Score sorting is cached once per prediction array.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATASETS = ("xes3g5m", "assist17", "nips34")
DEFAULT_VARIANTS = ("dkt", "akt", "simplekt", "fchgnn", "enhanced", "nohg", "weighted")
ALIASES = {"fchgnn_enhanced": "enhanced", "fchgnn_nohg": "nohg", "fchgnn_weighted": "weighted"}
LABELS = {
    "dkt": "DKT", "akt": "AKT", "simplekt": "simpleKT", "fchgnn": "FC-HGNN 基础",
    "enhanced": "FC-HGNN 增强", "nohg": "增强模型关闭超图", "weighted": "加权 BCE",
}
THRESHOLDS = ("fixed_0_5", "f1", "mcc")
RANK_METRICS = ("auc", "pr_auc", "incorrect_pr_auc", "brier", "ece")
REPORT_METRICS = (
    "auc", "incorrect_pr_auc", "accuracy", "balanced_accuracy", "mcc", "incorrect_precision",
    "incorrect_recall", "incorrect_f1", "brier", "ece", "intervention_rate",
    "incorrect_miss_rate", "false_alarm_rate", "cost_miss5_false_alarm1",
)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_metrics(tp: float, fp: float, tn: float, fn: float) -> dict[str, float]:
    """Confusion entries use correct response (label 1) as the positive class."""
    total = tp + fp + tn + fn
    if total <= 0:
        raise ValueError("No weighted evaluation events")
    recall_correct = _ratio(tp, tp + fn)
    recall_incorrect = _ratio(tn, tn + fp)
    f1_correct = _ratio(2 * tp, 2 * tp + fp + fn)
    f1_incorrect = _ratio(2 * tn, 2 * tn + fp + fn)
    recalls = [value for count, value in ((tp + fn, recall_correct), (tn + fp, recall_incorrect)) if count]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": float(np.mean(recalls)),
        "precision": _ratio(tp, tp + fp), "recall": recall_correct, "f1": f1_correct,
        "macro_f1": (f1_correct + f1_incorrect) / 2,
        "incorrect_precision": _ratio(tn, tn + fn), "incorrect_recall": recall_incorrect,
        "incorrect_f1": f1_incorrect, "mcc": _ratio(tp * tn - fp * fn, denominator),
        "intervention_rate": (tn + fn) / total,
        "incorrect_miss_rate": _ratio(fp, tn + fp), "false_alarm_rate": _ratio(fn, tp + fn),
        "cost_miss5_false_alarm1": (5 * fp + fn) / total,
    }


class ScoreOrder:
    """Descending score order and tie endpoints for exact weighted AUC and AP."""

    def __init__(self, labels: np.ndarray, scores: np.ndarray, user_codes: np.ndarray):
        self.order = np.argsort(-scores, kind="stable")
        self.labels = labels[self.order].astype(np.float64)
        self.user_codes = user_codes[self.order]
        ordered = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]

    def ranking(self, user_weights: np.ndarray) -> tuple[float, float]:
        weights = user_weights[self.user_codes]
        positive = np.add.reduceat(weights * self.labels, self.starts)
        negative = np.add.reduceat(weights * (1.0 - self.labels), self.starts)
        total_positive, total_negative = float(positive.sum()), float(negative.sum())
        if not total_positive or not total_negative:
            return float("nan"), float("nan")
        tp = np.cumsum(positive)
        fp = np.cumsum(negative)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        ap = float(np.dot(positive, precision) / total_positive)
        # All lower-scored negatives win; same-score positive/negative pairs tie.
        auc = float(np.dot(positive, total_negative - fp + 0.5 * negative) / (total_positive * total_negative))
        return auc, ap


class PredictionCache:
    """Reusable grouped confusion/calibration sums and score orders."""

    def __init__(self, prediction: dict[str, np.ndarray], threshold: float = 0.5, with_ranking: bool = True):
        required = ("target", "probability", "user_idx", "event_idx")
        missing = [key for key in required if key not in prediction]
        if missing:
            raise ValueError(f"Prediction is missing fields: {missing}")
        self.prediction = {key: np.asarray(value).reshape(-1) for key, value in prediction.items()}
        self.y = self.prediction["target"].astype(np.float64)
        self.p = self.prediction["probability"].astype(np.float64)
        n = len(self.y)
        if n == 0 or any(len(self.prediction[key]) != n for key in required):
            raise ValueError("Prediction fields must be nonempty and have identical lengths")
        if not np.isin(self.y, [0, 1]).all() or not np.isfinite(self.p).all() or ((self.p < 0) | (self.p > 1)).any():
            raise ValueError("Prediction requires binary targets and finite probabilities in [0, 1]")
        self.users, self.user_codes = np.unique(self.prediction["user_idx"], return_inverse=True)
        self.n_users = len(self.users)
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold):
            raise ValueError("Threshold must be finite")
        # Match train.metrics' native array >= Python-float scalar semantics.
        # For float32 arrays a scalar such as 5/6 is rounded before comparison;
        # widening the array first can flip events exactly on that boundary.
        pred = self.prediction["probability"] >= self.threshold
        positive = self.y == 1
        columns = (pred & positive, pred & ~positive, ~pred & ~positive, ~pred & positive)
        self.confusion = np.column_stack([
            np.bincount(self.user_codes, weights=column.astype(np.float64), minlength=self.n_users)
            for column in columns
        ])
        self.counts = np.bincount(self.user_codes, minlength=self.n_users).astype(np.float64)
        self.squared_error = np.bincount(self.user_codes, weights=(self.y - self.p) ** 2, minlength=self.n_users)
        bins = np.minimum((self.p * 15).astype(np.int64), 14)
        joint = self.user_codes * 15 + bins
        self.bin_count = np.bincount(joint, minlength=self.n_users * 15).reshape(self.n_users, 15)
        self.bin_target = np.bincount(joint, weights=self.y, minlength=self.n_users * 15).reshape(self.n_users, 15)
        self.bin_probability = np.bincount(joint, weights=self.p, minlength=self.n_users * 15).reshape(self.n_users, 15)
        self.correct_order = ScoreOrder(self.y, self.p, self.user_codes) if with_ranking else None
        # train.metrics computes 1.0 - y_prob in the saved probability dtype.
        # With float32, this subtraction can merge distinct scores into ties;
        # promoting before subtraction would silently change error-class AP.
        # Keep the original arithmetic for score ordering; weighted counts and
        # calibration sums still use float64 for stable bootstrap accumulation.
        incorrect_scores = 1.0 - self.prediction["probability"]
        self.incorrect_order = ScoreOrder(1.0 - self.y, incorrect_scores, self.user_codes) if with_ranking else None

    def metrics(self, user_weights: np.ndarray | None = None, requested: Iterable[str] | None = None) -> dict[str, float]:
        weights = np.ones(self.n_users) if user_weights is None else np.asarray(user_weights, dtype=np.float64)
        if weights.shape != (self.n_users,) or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("User weights must be finite nonnegative weights for the cached user order")
        result = confusion_metrics(*(weights @ self.confusion))
        requested_set = set(REPORT_METRICS if requested is None else requested)
        total = float(weights @ self.counts)
        if "brier" in requested_set:
            result["brier"] = float(weights @ self.squared_error / total)
        if "ece" in requested_set:
            result["ece"] = float(np.abs(weights @ self.bin_target - weights @ self.bin_probability).sum() / total)
        if requested_set.intersection(("auc", "pr_auc")):
            if self.correct_order is None:
                raise ValueError("Ranking cache was not requested")
            result["auc"], result["pr_auc"] = self.correct_order.ranking(weights)
        if "incorrect_pr_auc" in requested_set:
            if self.incorrect_order is None:
                raise ValueError("Ranking cache was not requested")
            _, result["incorrect_pr_auc"] = self.incorrect_order.ranking(weights)
        return {key: result[key] for key in requested_set}


def assert_aligned(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    """Never silently reorder or intersect targets from different protocols."""
    for key in ("event_idx", "user_idx", "target"):
        if key not in left or key not in right or not np.array_equal(left[key], right[key]):
            raise ValueError(f"Paired evaluation alignment failed for {key}; event order and labels must be identical")


def student_draw_counts(rng: np.random.Generator, n_users: int, repetitions: int) -> np.ndarray:
    if n_users < 1 or repetitions < 1:
        raise ValueError("Bootstrap needs at least one user and one repetition")
    # Exactly n_users independent draws, each student has probability 1/n_users.
    return rng.multinomial(n_users, np.full(n_users, 1.0 / n_users), size=repetitions)


def paired_cluster_bootstrap(
    left: list[dict[str, np.ndarray]], right: list[dict[str, np.ndarray]],
    left_thresholds: list[float] | None = None, right_thresholds: list[float] | None = None,
    repetitions: int = 200, seed: int = 20261004,
    requested: Iterable[str] = ("auc", "incorrect_pr_auc", "incorrect_f1", "incorrect_recall", "mcc"),
) -> dict[str, Any]:
    """CI for mean left-minus-right differences across fixed, paired run seeds.

    Thresholds are the validation-selected constants, one per training seed. The
    same student count vector is used for both methods AND all training seeds.
    The resulting CI is conditional on the trained models and fixed data split.
    """
    if len(left) == 0 or len(left) != len(right):
        raise ValueError("Paired bootstrap requires equal nonempty training-seed lists")
    reference = left[0]
    for a, b in zip(left, right):
        assert_aligned(reference, a)
        assert_aligned(reference, b)
    left_thresholds = [0.5] * len(left) if left_thresholds is None else left_thresholds
    right_thresholds = [0.5] * len(right) if right_thresholds is None else right_thresholds
    if len(left_thresholds) != len(left) or len(right_thresholds) != len(right):
        raise ValueError("One fixed threshold is required for each training seed")
    names = tuple(requested)
    ranking = bool(set(names).intersection(("auc", "pr_auc", "incorrect_pr_auc")))
    left_cache = [PredictionCache(p, t, ranking) for p, t in zip(left, left_thresholds)]
    right_cache = [PredictionCache(p, t, ranking) for p, t in zip(right, right_thresholds)]
    draws = student_draw_counts(np.random.default_rng(seed), left_cache[0].n_users, repetitions)
    point_differences = {name: [] for name in names}
    for a, b in zip(left_cache, right_cache):
        am, bm = a.metrics(requested=names), b.metrics(requested=names)
        for name in names:
            point_differences[name].append(am[name] - bm[name])
    differences = {name: np.zeros(repetitions, dtype=np.float64) for name in names}
    for index, counts in enumerate(draws):
        for a, b in zip(left_cache, right_cache):
            am, bm = a.metrics(counts, names), b.metrics(counts, names)
            for name in names:
                differences[name][index] += (am[name] - bm[name]) / len(left_cache)
    intervals = {}
    for name, values in differences.items():
        finite = values[np.isfinite(values)]
        lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (float("nan"), float("nan"))
        intervals[name] = {
            "mean_difference": float(np.mean(point_differences[name])),
            "seed_differences": point_differences[name],
            "ci95": [float(lo), float(hi)], "valid_repetitions": int(len(finite)),
        }
    return {
        "direction": "left_minus_right", "training_seed_count": len(left),
        "test_students": left_cache[0].n_users, "test_events_per_seed": len(left_cache[0].y),
        "repetitions": repetitions, "bootstrap_seed": seed,
        "sampling_unit": "student", "draws_per_repetition": left_cache[0].n_users,
        "seed_aggregation": "mean of paired metric differences; training seeds held fixed",
        "thresholds_fixed_from_validation": {"left": left_thresholds, "right": right_thresholds},
        "metrics": intervals,
    }


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key] = {
            "mean": float(finite.mean()) if len(finite) else float("nan"),
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            "runs": len(values), "finite_runs": len(finite),
        }
    return result


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        result = {key: loaded[key] for key in loaded.files}
    PredictionCache(result, with_ranking=False)  # Fail early on malformed output.
    return result


def _canonical(variant: str) -> str:
    return ALIASES.get(variant, variant)


def load_records(root: Path, stage: str) -> dict[tuple[str, str, int], dict]:
    records = {}
    for path in sorted((root / stage).glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not {"dataset", "variant", "seed", "result"}.issubset(record):
            continue
        key = record["dataset"], _canonical(record["variant"]), int(record["seed"])
        if key in records:
            raise ValueError(f"Duplicate {stage} record: {key}")
        record["_path"] = str(path)
        records[key] = record
    return records


def _threshold_metrics(result: dict, split: str) -> dict[str, dict[str, float]]:
    table = dict(result.get(f"{split}_by_threshold", {}))
    if not table and split in result:
        table["selected"] = result[split]
    for name in THRESHOLDS:
        if name not in table:
            continue
        for rank_metric in RANK_METRICS:
            available = [values[rank_metric] for values in table.values() if rank_metric in values]
            if available and not np.allclose(available, available[0], rtol=1e-8, atol=1e-10, equal_nan=True):
                raise AssertionError(f"Changing only a threshold changed {split} {rank_metric}")
    return table


def _prediction_path(root: Path, record: dict, split: str) -> Path:
    candidates = [root / "predictions" / record["dataset"] / f"{record['variant']}_seed{record['seed']}" / f"{split}.npz"]
    location = record["result"].get("predictions_dir")
    if location:
        base = Path(location)
        candidates.append(base / f"{split}.npz")
        if not base.is_absolute():
            candidates.append(root / base / f"{split}.npz")
    return next((path for path in candidates if path.exists()), candidates[0])


def _validation_auc(record: dict) -> float:
    table = _threshold_metrics(record["result"], "validation")
    values = next(iter(table.values()), record["result"].get("validation", {}))
    return float(values.get("auc", float("nan")))


def _plan(root: Path, protocol_path: Path | None) -> tuple[dict, Path | None]:
    candidates = [protocol_path] if protocol_path else [root / name for name in ("protocol.json", "frozen_plan.json", "experiment_plan.json", "plan.json")]
    path = next((candidate for candidate in candidates if candidate is not None and candidate.exists()), None)
    return (json.loads(path.read_text(encoding="utf-8")), path) if path else ({}, None)


def analyze(root: Path, repetitions: int = 200, prepared_root: Path | None = None, protocol_path: Path | None = None) -> dict:
    screen, final = load_records(root, "screen"), load_records(root, "final")
    protocol, found_protocol = _plan(root, protocol_path)
    frozen_path = root / "frozen_plan.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else {}
    snapshot = protocol.get("config_snapshot", {})
    scope = "smoke" if "smoke" in root.name.lower() or snapshot.get("training", {}).get("epochs", 8) <= 1 else "controlled_experiment"
    datasets = list(protocol.get("datasets", DEFAULT_DATASETS))
    variants_raw = protocol.get("variants", DEFAULT_VARIANTS)
    variants = [_canonical(value if isinstance(value, str) else value["name"]) for value in variants_raw]
    observed_seeds = sorted({key[2] for key in screen.keys() | final.keys()})
    seed_values = protocol.get("seeds", protocol.get("training_seeds"))
    # The intended protocol has three training seeds. With a partially written
    # plan we can display progress but cannot select a winner from just one run.
    seeds = list(map(int, seed_values)) if seed_values is not None else observed_seeds
    required_seed_count = len(seeds) if seed_values is not None else max(3, len(seeds))
    expected = len(datasets) * len(variants) * required_seed_count
    planned_keys = {(d, v, s) for d in datasets for v in variants for s in seeds}
    validation_records = dict(final)
    validation_records.update(screen)  # Screening validation is authoritative.
    completed_validation = {key for key in validation_records if key in planned_keys}
    completed_final = {key for key, value in final.items() if key in planned_keys and value["result"].get("test_by_threshold")}
    selection_ready = bool(seeds) and len(seeds) == required_seed_count and len(completed_validation) == expected
    all_final = selection_ready and len(completed_final) == expected
    pending = sorted(planned_keys - completed_final)
    summary: dict[str, Any] = {
        "root": str(root), "status": "complete" if all_final else "partial",
        "scope": scope, "frozen_plan_path": str(frozen_path) if frozen else None,
        "protocol_path": str(found_protocol) if found_protocol else None,
        "protocol": protocol, "datasets": {},
        "progress": {
            "expected_runs": expected, "validation_runs": len(completed_validation), "test_runs": len(completed_final),
            "required_training_seed_count": required_seed_count, "training_seeds": seeds,
            "validation_selection_ready": selection_ready, "pending_test_runs": [list(key) for key in pending],
            "seed_ids_not_yet_known": max(required_seed_count - len(seeds), 0),
        },
        "bootstrap_method": {
            "unit": "student", "paired": True, "repetitions": repetitions,
            "training_seeds_fixed": True, "student_draws_shared_across_models_and_seeds": True,
            "test_threshold_reselection": False, "interval": "95% percentile, conditional on trained models and data split",
            "score_sorting": "cached once per probability array; exact tie groups",
        },
    }
    if prepared_root is None:
        configured = protocol.get("prepared_root", snapshot.get("prepared_root"))
        if configured:
            prepared_root = Path(configured)
            if not prepared_root.is_absolute():
                prepared_root = root / prepared_root
        else:
            prepared_root = root / "prepared"
    for dataset in datasets:
        block: dict[str, Any] = {"variants": {}, "paired_comparisons": {}, "selection": {"basis": "mean validation ROC-AUC across the fixed training seeds"}}
        metadata = prepared_root / dataset / "metadata.json"
        if metadata.exists():
            block["prepared_metadata"] = json.loads(metadata.read_text(encoding="utf-8"))
            block["prepared_metadata_path"] = str(metadata)
        for variant in variants:
            validation_rows = [validation_records[(dataset, variant, seed)] for seed in seeds if (dataset, variant, seed) in validation_records]
            test_rows = [final[(dataset, variant, seed)] for seed in seeds if (dataset, variant, seed) in completed_final]
            validation_tables = [_threshold_metrics(row["result"], "validation") for row in validation_rows]
            test_tables = [_threshold_metrics(row["result"], "test") for row in test_rows]
            row_summary = {
                "validation_runs": len(validation_rows), "test_runs": len(test_rows),
                "validation_auc": aggregate_metrics([{"auc": _validation_auc(row)} for row in validation_rows]).get("auc"),
                "validation_by_threshold": {
                    name: aggregate_metrics([table[name] for table in validation_tables if name in table])
                    for name in sorted(set().union(*(table.keys() for table in validation_tables)))
                },
                "test_by_threshold": {
                    name: aggregate_metrics([table[name] for table in test_tables if name in table])
                    for name in sorted(set().union(*(table.keys() for table in test_tables)))
                },
                "thresholds": aggregate_metrics([row["result"].get("comparison_thresholds", {}) for row in validation_rows]),
                "training_seeds": [row["seed"] for row in validation_rows],
                "best_epochs": [int(max(row["result"].get("history", []), key=lambda item: item.get("auc", -math.inf))["epoch"]) for row in validation_rows if row["result"].get("history")],
            }
            block["variants"][variant] = row_summary
        if selection_ready:
            def select(candidates: Iterable[str]) -> str | None:
                scored = [(variant, block["variants"][variant]["validation_auc"]["mean"]) for variant in candidates if variant in block["variants"] and block["variants"][variant]["validation_auc"]]
                scored = [(name, auc) for name, auc in scored if math.isfinite(auc)]
                return min(scored, key=lambda item: (-item[1], item[0]))[0] if scored else None
            best_baseline, best_graph = select(("akt", "simplekt")), select(("fchgnn", "enhanced"))
            best_overall = select(("dkt", "akt", "simplekt"))
            frozen_selection = frozen.get("selections", {}).get(dataset)
            if frozen_selection:
                if not frozen.get("frozen_before_test"):
                    raise ValueError("Frozen plan does not attest selection before test")
                for field, selected, eligible in (("best_baseline", best_overall, ("dkt", "akt", "simplekt")), ("best_hypergraph", best_graph, ("fchgnn", "enhanced"))):
                    frozen_name = _canonical(frozen_selection[field]) if frozen_selection.get(field) else None
                    if frozen_name is not None:
                        if frozen_name not in eligible or not selected:
                            raise ValueError(f"Frozen selection has invalid {field} for {dataset}")
                        chosen_mean = block["variants"][frozen_name]["validation_auc"]["mean"]
                        best_mean = block["variants"][selected]["validation_auc"]["mean"]
                        if not np.isclose(chosen_mean, best_mean, rtol=0, atol=1e-10):
                            raise ValueError(f"Frozen {field} does not agree with validation-only selection for {dataset}")
                        if field == "best_baseline":
                            best_overall = frozen_name
                        else:
                            best_graph = frozen_name
                block["selection"]["frozen_before_test"] = True
            block["selection"].update({"best_modern_baseline": best_baseline, "best_graph": best_graph, "best_overall_baseline": best_overall})
        else:
            best_baseline = best_graph = None
            block["selection"]["pending"] = "All planned validation runs are required before selecting models"

        prediction_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        def predictions_for(variant: str) -> list[dict[str, np.ndarray]] | None:
            predictions = []
            for seed in seeds:
                key = dataset, variant, seed
                if key not in completed_final:
                    return None
                path = _prediction_path(root, final[key], "test")
                if not path.exists():
                    block.setdefault("missing_predictions", []).append(str(path))
                    return None
                if (variant, seed) not in prediction_cache:
                    prediction_cache[(variant, seed)] = load_prediction(path)
                predictions.append(prediction_cache[(variant, seed)])
            return predictions if len(predictions) == required_seed_count else None

        def thresholds_for(variant: str, name: str) -> list[float]:
            return [float(validation_records[(dataset, variant, seed)]["result"]["comparison_thresholds"][name]) for seed in seeds]

        def compare(name: str, left_variant: str, right_variant: str, left_policy: str, right_policy: str, names: tuple[str, ...]) -> None:
            if not selection_ready:
                return
            left_p, right_p = predictions_for(left_variant), predictions_for(right_variant)
            if left_p is None or right_p is None:
                return
            result = paired_cluster_bootstrap(
                left_p, right_p, thresholds_for(left_variant, left_policy), thresholds_for(right_variant, right_policy),
                repetitions=repetitions, requested=names,
            )
            result.update({"left_variant": left_variant, "right_variant": right_variant, "left_policy": left_policy, "right_policy": right_policy, "training_seeds": seeds})
            block["paired_comparisons"][name] = result

        if best_baseline and best_graph:
            compare("selected_graph_vs_selected_modern", best_graph, best_baseline, "mcc", "mcc", ("auc", "incorrect_pr_auc", "incorrect_f1", "incorrect_recall", "mcc", "brier", "cost_miss5_false_alarm1"))
            if block["selection"].get("best_overall_baseline") != best_baseline:
                compare("selected_graph_vs_frozen_overall_baseline", best_graph, block["selection"]["best_overall_baseline"], "mcc", "mcc", ("auc", "incorrect_pr_auc", "incorrect_f1", "incorrect_recall", "mcc"))
        if "enhanced" in variants and "nohg" in variants:
            compare("enhanced_vs_nohg", "enhanced", "nohg", "mcc", "mcc", ("auc", "incorrect_pr_auc", "incorrect_f1", "incorrect_recall", "mcc"))
        for variant in variants:
            compare(f"{variant}_mcc_vs_f1", variant, variant, "mcc", "f1", ("incorrect_f1", "incorrect_recall", "mcc"))

        # Constant probability from TRAIN targets; decision class from its prior.
        for variant in variants:
            test_p = predictions_for(variant)
            if test_p is None:
                continue
            majority_rows = []
            priors = []
            for seed, prediction in zip(seeds, test_p):
                counts = validation_records[(dataset, variant, seed)]["result"].get("class_counts", {})
                total = counts.get("correct", 0) + counts.get("incorrect", 0)
                if not total:
                    continue
                prior = counts["correct"] / total
                constant = dict(prediction, probability=np.full(len(prediction["target"]), prior))
                majority_rows.append(PredictionCache(constant).metrics())
                priors.append(prior)
            if majority_rows:
                block["majority_train_prior_baseline"] = {"train_correct_priors": priors, "test": aggregate_metrics(majority_rows), "training_seeds": seeds}
                break
        summary["datasets"][dataset] = block
    return summary


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _format(value: dict | None, digits: int = 4) -> str:
    if not value or value.get("mean") is None or not math.isfinite(value["mean"]):
        return "—"
    return f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"


def _metric(table: dict, name: str) -> str:
    return _format(table.get(name))


def markdown_report(summary: dict) -> str:
    progress = summary["progress"]
    complete = summary["status"] == "complete"
    smoke = summary.get("scope") == "smoke"
    seed_count = progress["required_training_seed_count"]
    lines = [
        "# 新数据集双方向实验报告", "",
        f"状态：{'全部计划运行已完成' if complete else '阶段性结果，计划尚未完成'}。验证运行 {progress['validation_runs']}/{progress['expected_runs']}，测试运行 {progress['test_runs']}/{progress['expected_runs']}。",
        "实验范围：技术烟雾检查，只验证数据、训练、保存和分析链路，不属于正式性能实验。" if smoke else "实验范围：统一预算下的固定学生划分、三个训练种子的本地实验。",
        f"计划包含 {seed_count} 个训练种子，共享固定学生划分；模型与阈值依据验证集决定，测试结果不参与选择。表格为已完成训练种子的均值 ± 样本标准差，单次运行的标准差记为 0，标准差不是置信区间。", "",
        "## 超图模型方向", "",
        "基础 DKT、AKT、simpleKT 和 FC-HGNN 使用第一叶知识点；增强模型和 noHG 使用全部叶知识点。基础与增强的差异同时包含输入信息与模块变化，不能归因为单独的超图贡献。增强与 noHG 保留相同的多知识点融合、时间特征、对比目标和预测头，noHG 仅将超图层数置 0。",
        "AKT/simpleKT 为适配本项目协议的实现，应结合上游来源和适配说明解读。本地 FC-HGNN 是重构与增强候选；本实验使用统一预算检验可行性，不能证明达到论文 SOTA，也不能将结果与其他划分、训练预算或目标定义的论文分数直接比较。", "",
        "## 错误敏感决策与评估偏差", "",
        "标签 1 为正确，标签 0 为错误；错误 AP 使用 1 − 概率，错误 Recall/F1 将错误类视为目标类。固定 0.5、正确类 F1 与 MCC 阈值使用相同的预测概率，因此 AUC、两类 AP、Brier 和 ECE 必须不变。阈值收益是决策变化，不是排序或校准变好。",
        "预设成本情景为漏报一次错误计 5、对正确作答误报计 1：成本 = (5 × 漏错数 + 误报数) / 测试目标数。该比例是研究假设，不能表述为已证实的教学收益。5/6 概率阈值仅在概率校准和该成本假设成立时有决策含义；加权 BCE 概率可能失准，需一并检查 Brier/ECE。", "",
    ]
    for dataset, block in summary["datasets"].items():
        lines.extend([f"## {dataset}", ""])
        metadata = block.get("prepared_metadata", {})
        if metadata:
            lines.append(f"处理后规模：{metadata.get('num_users', '—')} 名学生、{metadata.get('num_items', '—')} 题、{metadata.get('num_concepts', '—')} 知识点、{metadata.get('num_interactions', '—')} 条交互。")
            for split, stats in metadata.get("split_statistics", {}).items():
                rate = stats.get("target_correct_rate")
                rate_text = f"，正确率 {rate:.2%}" if rate is not None else ""
                lines.append(f"{split}：{stats.get('num_users', '—')} 人、{stats.get('num_targets', '—')} 个预测目标{rate_text}。")
            lines.append("")
        selection = block["selection"]
        if selection.get("best_modern_baseline"):
            lines.append(f"验证集跨种子平均 AUC 选择的现代基线：{LABELS.get(selection['best_modern_baseline'], selection['best_modern_baseline'])}；超图候选：{LABELS.get(selection['best_graph'], selection['best_graph'])}；全部基线中的冻结选择：{LABELS.get(selection.get('best_overall_baseline'), selection.get('best_overall_baseline'))}。测试集仅评价这些固定选择。")
        else:
            lines.append("模型选择待全部计划验证运行完成后进行；当前不按测试集挑选最佳模型。")
        lines.extend(["", "| 变体 | 验证/测试次数 | 验证 AUC | 测试 AUC | 错误 AP | Brier | ECE |", "|---|---:|---:|---:|---:|---:|---:|"])
        for variant, row in block["variants"].items():
            test = row["test_by_threshold"].get("mcc", {})
            lines.append(f"| {LABELS.get(variant, variant)} | {row['validation_runs']}/{row['test_runs']} | {_format(row['validation_auc'])} | {_metric(test, 'auc')} | {_metric(test, 'incorrect_pr_auc')} | {_metric(test, 'brier')} | {_metric(test, 'ece')} |")
        lines.extend(["", "| 变体/阈值 | 阈值 | Accuracy | BAcc | MCC | 错误 Recall | 错误 F1 | 干预比例 | 漏错率 | 误报率 | 5:1 成本 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for variant, row in block["variants"].items():
            for name in THRESHOLDS:
                table = row["test_by_threshold"].get(name)
                if not table:
                    continue
                metrics_text = " | ".join(_metric(table, metric) for metric in ("accuracy", "balanced_accuracy", "mcc", "incorrect_recall", "incorrect_f1", "intervention_rate", "incorrect_miss_rate", "false_alarm_rate", "cost_miss5_false_alarm1"))
                lines.append(f"| {LABELS.get(variant, variant)} / {name} | {_format(row['thresholds'].get(name))} | {metrics_text} |")
        additional_rows = []
        relevant = list(dict.fromkeys(name for name in (selection.get("best_graph"), selection.get("best_modern_baseline"), "weighted") if name in block["variants"]))
        for variant in relevant:
            row = block["variants"][variant]
            for name in ("cost_5_to_1", "incorrect_f1", "balanced_accuracy"):
                table = row["test_by_threshold"].get(name)
                if table:
                    values = " | ".join(_metric(table, metric) for metric in ("incorrect_recall", "false_alarm_rate", "intervention_rate", "cost_miss5_false_alarm1"))
                    additional_rows.append(f"| {LABELS.get(variant, variant)} / {name} | {_format(row['thresholds'].get(name))} | {values} |")
        if additional_rows:
            lines.extend(["", "其他预定决策策略（对验证集选定模型与加权对照）：错误 F1 和 BAcc 阈值均由验证集决定；5:1 成本策略固定阈值为 5/6，不按测试成本挑选。这里呈现同一预测概率的不同决策，不另作多策略显著性选择。", "", "| 变体/策略 | 阈值 | 错误 Recall | 误报率 | 干预比例 | 5:1 成本 |", "|---|---:|---:|---:|---:|---:|"])
            lines.extend(additional_rows)
        majority = block.get("majority_train_prior_baseline")
        if majority:
            table = majority["test"]
            lines.extend(["", f"训练集先验常数基线：Accuracy {_metric(table, 'accuracy')}；错误 Recall {_metric(table, 'incorrect_recall')}；MCC {_metric(table, 'mcc')}；错误 AP {_metric(table, 'incorrect_pr_auc')}。该基线的概率仅由训练目标比例确定。"])
        comparisons = block["paired_comparisons"]
        if comparisons:
            lines.extend(["", "学生聚类配对 bootstrap：每次等概率抽取与测试学生数相同的学生，保留所抽学生全部事件；两个方法及所有固定训练种子共享同一次抽样。每次计算各 seed 指标差再平均。阈值按各 seed 的验证集固定，不在测试重选；区间条件于已训练模型和固定划分，不代表重新训练或新平台的不确定性。", "", "| 比较（左 − 右） | 指标 | 平均差 | 95% 区间 | 有效重复 |", "|---|---|---:|---:|---:|"])
            for name, comparison in comparisons.items():
                description = f"{LABELS.get(comparison['left_variant'], comparison['left_variant'])}/{comparison['left_policy']} − {LABELS.get(comparison['right_variant'], comparison['right_variant'])}/{comparison['right_policy']}"
                for metric, values in comparison["metrics"].items():
                    lo, hi = values["ci95"]
                    lines.append(f"| {description} | {metric} | {values['mean_difference']:.5f} | [{lo:.5f}, {hi:.5f}] | {values['valid_repetitions']} |")
        else:
            lines.extend(["", f"配对区间待全部计划验证运行以及对应 {seed_count} 个种子的测试预测齐备后计算。"])
        lines.append("")
    xes_sample = summary.get("protocol", {}).get("config_snapshot", {}).get("datasets", {}).get("xes3g5m", {}).get("sample_users", 5000)
    sample_text = f"固定的 {xes_sample:,} 名学生样本" if xes_sample is not None else "全部可用学生"
    lines.extend([
        "## 解释边界", "",
        f"- XES3G5M 本轮使用{sample_text}，保留作者发布的题目级片段与行内顺序；并非官方 5 折复现。具体纳入人数和预测目标数以处理元数据为准。",
        "- ASSISTments 2017 本轮使用具名技能、排除提示并只保留主问题的作答动作。该口径接近平衡，不能沿用包含提示的原始错误率，也不等于已核验的严格首答。",
        "- NIPS2020 Task 3/4 仅用官方训练表另建学生划分；时间精度为分钟，同分钟按 AnswerId 固定排序属于伪时序，无法保证块内历史标签真实可用。结果属于该约定下的适配实验；不能作为严格时间因果预测结论。",
        "- 多标签与层级祖先必须区分；超图使用题目—叶知识点静态元数据，题目经验难度和类别权重仅从训练学生估计。",
        "- 时间增强版及 noHG 接收目标事件的 next_time_gap，推断需要已知目标时间。XES 时间字段的题目呈现/答案提交语义尚未核实，所以该增强比较属于事件时间已知的离线预测，不应直接解释为答前实时干预表现。",
        "- 不同数据集错误比例不同，错误 AP 不可直接跨平台当作难度或优劣排名。阈值在验证集选择仍会有选择误差；这些测试 bootstrap 区间未涵盖阈值重新拟合的不确定性。",
        "- 若区间跨 0，只表示当前样本和协议下尚未明确区分；不能据此证明两模型等效。200 次重复为本地阶段性区间，正式投稿建议增加重复次数并补充完整数据、更多划分和上游实现复现。", "",
    ])
    return "\n".join(lines)


def plot_summary(summary: dict, output: Path) -> list[str]:
    """Optional standalone static figure; no app or browser dependency."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return []
    datasets = list(summary["datasets"])
    variants = list(DEFAULT_VARIANTS)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * max(1, len(datasets)), 4), squeeze=False)
    plotted = False
    markers = {"fixed_0_5": "o", "f1": "^", "mcc": "s"}
    for ax, dataset in zip(axes[0], datasets):
        rows = summary["datasets"][dataset]["variants"]
        for variant in variants:
            if variant not in rows:
                continue
            points = []
            for name in THRESHOLDS:
                table = rows[variant]["test_by_threshold"].get(name, {})
                recall, alarm = table.get("incorrect_recall"), table.get("false_alarm_rate")
                if recall and alarm:
                    points.append((alarm["mean"], recall["mean"], name))
            if points:
                x, y, names = zip(*points)
                line, = ax.plot(x, y, "-", label=variant, alpha=0.7, linewidth=1)
                for px, py, name in points:
                    ax.scatter(px, py, marker=markers[name], color=line.get_color(), s=30)
                plotted = True
        ax.set(xlabel="False alarm rate on correct responses", ylabel="Recall of incorrect responses", title=dataset, xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=8)
    if not plotted:
        plt.close(fig)
        return []
    fig.suptitle("Error detection decisions (connected points are not an ROC curve)")
    handles = [Line2D([0], [0], marker=markers[name], linestyle="None", color="black", label=label) for name, label in (("fixed_0_5", "Fixed 0.5"), ("f1", "Validation correct-F1"), ("mcc", "Validation MCC"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    path = output / "error_decision_tradeoffs.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [str(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "artifacts/new_datasets_dual")
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be positive")
    root = args.root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "analysis"
    summary = analyze(root, args.bootstrap_reps, args.prepared_root, args.protocol)
    output.mkdir(parents=True, exist_ok=True)
    summary["figures"] = [] if args.no_plots else plot_summary(summary, output)
    (output / "summary.json").write_text(json.dumps(_clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "双方向实验报告.md").write_text(markdown_report(summary) + "\n", encoding="utf-8")
    concise_progress = {key: value for key, value in summary["progress"].items() if key != "pending_test_runs"}
    print(json.dumps({"status": summary["status"], "scope": summary["scope"], **concise_progress, "summary": str(output / "summary.json"), "report": str(output / "双方向实验报告.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
