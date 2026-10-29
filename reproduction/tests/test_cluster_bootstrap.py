import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SPEC = importlib.util.spec_from_file_location("new_dataset_analysis", Path(__file__).resolve().parents[1] / "analyze_new_datasets.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def prediction(probabilities, users=None, target=None):
    probabilities = np.asarray(probabilities, dtype=float)
    n = len(probabilities)
    return {
        "target": np.asarray(target if target is not None else [0, 1] * (n // 2), dtype=np.int8),
        "probability": probabilities,
        "user_idx": np.asarray(users if users is not None else np.repeat(np.arange(n // 2), 2)),
        "event_idx": np.arange(n),
    }


def test_uniform_student_draws_are_not_proportional_to_event_count():
    # A long-sequence user and a short-sequence user each have sampling mass 1/2.
    draws = analysis.student_draw_counts(np.random.default_rng(8), 2, 20_000)
    assert (draws.sum(axis=1) == 2).all()
    np.testing.assert_allclose(draws.mean(axis=0), [1, 1], atol=0.02)
    p = prediction([0.1] * 10 + [0.9, 0.1], users=[0] * 10 + [1, 1], target=[0] * 10 + [0, 1])
    cache = analysis.PredictionCache(p)
    assert cache.metrics(np.array([0, 2]), ["accuracy"])["accuracy"] == 0.0
    assert cache.metrics(np.array([2, 0]), ["accuracy"])["accuracy"] == 1.0


def test_identical_predictions_and_thresholds_have_zero_paired_ci():
    p = prediction([0.2, 0.8, 0.3, 0.6, 0.4, 0.9])
    result = analysis.paired_cluster_bootstrap([p, p, p], [p, p, p], repetitions=80)
    assert result["training_seed_count"] == 3
    assert result["test_events_per_seed"] == 6
    for values in result["metrics"].values():
        assert values["mean_difference"] == 0
        assert values["ci95"] == [0, 0]


def test_perfect_predictions_have_positive_paired_improvement():
    perfect = prediction([0.0, 1.0] * 5)
    reversed_p = prediction([0.9, 0.1] * 5)
    result = analysis.paired_cluster_bootstrap([perfect] * 3, [reversed_p] * 3, repetitions=60)
    for metric in ("auc", "incorrect_f1", "incorrect_recall", "mcc"):
        assert result["metrics"][metric]["mean_difference"] > 0
        assert result["metrics"][metric]["ci95"][0] > 0


@pytest.mark.parametrize("field", ["target", "event_idx", "user_idx"])
def test_alignment_failure_does_not_silently_intersect_or_reorder(field):
    p = prediction([0.2, 0.8, 0.3, 0.9])
    changed = {key: value.copy() for key, value in p.items()}
    changed[field][0], changed[field][1] = changed[field][1], changed[field][0]
    if field == "user_idx":
        changed[field][0] = 20
    with pytest.raises(ValueError, match=field):
        analysis.paired_cluster_bootstrap([p], [changed], repetitions=2)


def test_cached_weighted_ranking_matches_sklearn_with_ties():
    p = prediction([0.2, 0.8, 0.2, 0.8, 0.5, 0.5, 1.0, 0.0], target=[0, 1, 1, 0, 0, 1, 1, 0])
    counts = np.array([2, 0, 1, 3])
    weights = counts[p["user_idx"]]
    values = analysis.PredictionCache(p).metrics(counts)
    assert values["auc"] == pytest.approx(roc_auc_score(p["target"], p["probability"], sample_weight=weights))
    assert values["incorrect_pr_auc"] == pytest.approx(average_precision_score(1 - p["target"], 1 - p["probability"], sample_weight=weights))
    assert values["brier"] == pytest.approx(brier_score_loss(p["target"], p["probability"], sample_weight=weights))


def test_float32_error_complement_keeps_native_ties_like_training_metrics():
    low = np.float32(0.1)
    adjacent = np.nextafter(low, np.float32(1.0))
    scores = np.array([low, adjacent, low, adjacent, .8, .9], dtype=np.float32)
    p = prediction(scores, target=[0, 1, 0, 1, 1, 0])
    p["probability"] = scores
    # Distinct correct probabilities can become equal error scores in float32.
    assert low != adjacent
    assert (np.float32(1.0) - low) == (np.float32(1.0) - adjacent)
    native = average_precision_score(1 - p["target"], 1.0 - scores)
    promoted = average_precision_score(1 - p["target"], 1.0 - scores.astype(np.float64))
    assert native != promoted  # The fixture must exercise the rounding issue.
    actual = analysis.PredictionCache(p).metrics(requested=["incorrect_pr_auc"])
    assert actual["incorrect_pr_auc"] == pytest.approx(native, abs=1e-15)
    counts = np.array([2, 1, 3])
    weights = counts[p["user_idx"]]
    weighted = analysis.PredictionCache(p).metrics(counts, ["incorrect_pr_auc"])
    assert weighted["incorrect_pr_auc"] == pytest.approx(
        average_precision_score(1 - p["target"], 1.0 - scores, sample_weight=weights), abs=1e-15
    )


def test_float32_threshold_boundary_matches_native_scalar_comparison():
    threshold = 5 / 6
    scores = np.array([np.float32(threshold), .2], dtype=np.float32)
    p = prediction(scores, target=[0, 1])
    p["probability"] = scores
    assert (scores >= threshold).tolist() == [True, False]
    assert (scores.astype(np.float64) >= threshold).tolist() == [False, False]
    cached = analysis.PredictionCache(p, threshold, with_ranking=False).metrics(requested=["accuracy", "mcc", "intervention_rate"])
    assert cached == {"accuracy": 0.0, "mcc": -1.0, "intervention_rate": 0.5}


def test_threshold_policies_keep_probability_metrics_constant():
    p = prediction([0.2, 0.8, 0.4, 0.7])
    a = analysis.PredictionCache(p, threshold=0.3).metrics()
    b = analysis.PredictionCache(p, threshold=0.9).metrics()
    for metric in ("auc", "incorrect_pr_auc", "brier", "ece"):
        assert a[metric] == b[metric]
    assert a["incorrect_recall"] != b["incorrect_recall"]
    with pytest.raises(AssertionError, match="auc"):
        analysis._threshold_metrics({"test_by_threshold": {"f1": {"auc": 0.6}, "mcc": {"auc": 0.7}}}, "test")


def _saved_run(root, variant, seed, validation_auc, probabilities, phase="final"):
    p = prediction(probabilities)
    thresholds = {"fixed_0_5": 0.5, "f1": 0.3, "mcc": 0.7}
    tables = {name: analysis.PredictionCache(p, threshold).metrics() for name, threshold in thresholds.items()}
    validation = {name: dict(values, auc=validation_auc) for name, values in tables.items()}
    result = {"validation_by_threshold": validation, "comparison_thresholds": thresholds,
              "class_counts": {"correct": 18, "incorrect": 2}, "history": []}
    if phase == "final":
        result["test_by_threshold"] = tables
        directory = root / "predictions" / "xes3g5m" / f"{variant}_seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "test.npz", **p)
    record = {"dataset": "xes3g5m", "variant": variant, "seed": seed, "result": result}
    directory = root / phase / "xes3g5m"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{variant}_seed{seed}.json").write_text(json.dumps(record))


def test_complete_report_selects_on_validation_and_reads_snapshot_metadata(tmp_path):
    seeds = [1, 2, 3]
    variants = ["dkt", "akt", "simplekt", "fchgnn", "fchgnn_enhanced", "fchgnn_nohg", "fchgnn_weighted"]
    prepared = tmp_path / "outside_prepared"
    (prepared / "xes3g5m").mkdir(parents=True)
    (prepared / "xes3g5m/metadata.json").write_text(json.dumps({"num_users": 6, "num_items": 2, "num_concepts": 1, "num_interactions": 12}))
    protocol = {"datasets": ["xes3g5m"], "variants": variants, "training_seeds": seeds,
                "config_snapshot": {"prepared_root": str(prepared), "training": {"epochs": 8}, "datasets": {"xes3g5m": {"sample_users": 5000}}}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    for variant in variants:
        # AKT wins validation; simpleKT wins test, which must never select it.
        auc = 0.9 if variant in ["akt", "fchgnn_enhanced"] else 0.7
        probabilities = [0.0, 1.0] * 3 if variant == "simplekt" else [0.9, 0.1] * 3
        for seed in seeds:
            _saved_run(tmp_path, variant, seed, auc, probabilities)
    summary = analysis.analyze(tmp_path, repetitions=4)
    assert summary["status"] == "complete"
    assert summary["progress"]["test_runs"] == 21
    block = summary["datasets"]["xes3g5m"]
    assert block["selection"]["best_modern_baseline"] == "akt"
    assert block["selection"]["best_graph"] == "enhanced"
    assert block["prepared_metadata"]["num_users"] == 6
    assert "selected_graph_vs_selected_modern" in block["paired_comparisons"]
    assert "majority_train_prior_baseline" in block
    assert "5,000" in analysis.markdown_report(summary)


def test_partial_protocol_does_not_select_from_completed_subset(tmp_path):
    protocol = {"datasets": ["xes3g5m"], "variants": ["akt", "fchgnn"], "training_seeds": [1, 2, 3]}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    _saved_run(tmp_path, "akt", 1, 0.9, [0.2, 0.8] * 3)
    summary = analysis.analyze(tmp_path, repetitions=2)
    assert summary["status"] == "partial"
    assert summary["progress"]["test_runs"] == 1
    assert summary["progress"]["expected_runs"] == 6
    assert "best_modern_baseline" not in summary["datasets"]["xes3g5m"]["selection"]
    assert "计划尚未完成" in analysis.markdown_report(summary)
