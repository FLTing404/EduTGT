import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fchgnn_repro.train import best_threshold


def test_threshold_search_agrees_with_brute_force_including_ties():
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    p = np.array([0.02, 0.2, 0.2, 0.6, 0.6, 0.91, 0.91, 0.99])
    scorers = {
        "f1": lambda pred: f1_score(y, pred, zero_division=0),
        "incorrect_f1": lambda pred: f1_score(y, pred, pos_label=0, zero_division=0),
        "mcc": lambda pred: matthews_corrcoef(y, pred),
        "balanced_accuracy": lambda pred: balanced_accuracy_score(y, pred),
    }
    for objective, scorer in scorers.items():
        for method in ("grid", "exact"):
            if method == "exact":
                candidates = np.r_[0.0, np.unique(p), np.nextafter(p.max(), np.inf)]
            else:
                candidates = np.linspace(0.1, 0.9, 161) if objective == "f1" else np.linspace(0.05, 0.95, 181)
            expected = candidates[np.argmax([scorer(p >= t) for t in candidates])]
            assert best_threshold(y, p, objective, method) == expected


def test_exact_search_can_choose_all_positive_threshold():
    y = np.array([1, 1, 1, 0])
    p = np.array([0.001, 0.002, 0.003, 0.99])
    t = best_threshold(y, p, "f1", "exact")
    assert t < 0.001
