from __future__ import annotations

import itertools

import numpy as np


def _sequence_predictions(sample: dict[str, np.ndarray], params: tuple[float, float, float, float]) -> tuple[list[float], list[float]]:
    p_init, p_learn, p_guess, p_slip = params
    mastery: dict[int, float] = {}
    predictions: list[float] = []
    targets: list[float] = []
    mask = sample["mask"]
    for concept, observed, next_concept, target in zip(
        sample["concept"][mask], sample["correct"][mask], sample["next_concept"][mask], sample["target"][mask]
    ):
        concept = int(concept)
        prior = mastery.get(concept, p_init)
        likelihood = prior * (1 - p_slip) + (1 - prior) * p_guess
        if observed >= 0.5:
            posterior = prior * (1 - p_slip) / max(likelihood, 1e-8)
        else:
            posterior = prior * p_slip / max(1 - likelihood, 1e-8)
        mastery[concept] = posterior + (1 - posterior) * p_learn
        next_prior = mastery.get(int(next_concept), p_init)
        predictions.append(next_prior * (1 - p_slip) + (1 - next_prior) * p_guess)
        targets.append(float(target))
    return predictions, targets


def fit_bkt(train_samples: list[dict[str, np.ndarray]], valid_samples: list[dict[str, np.ndarray]]) -> tuple[float, float, float, float]:
    del train_samples  # BKT state is updated online; validation chooses stable global priors.
    grids = ([0.1, 0.3, 0.5], [0.05, 0.15, 0.3], [0.1, 0.2, 0.3], [0.05, 0.15, 0.25])
    best = None
    best_loss = float("inf")
    for params in itertools.product(*grids):
        predictions, targets = predict_bkt(valid_samples, params)
        p = np.clip(predictions, 1e-7, 1 - 1e-7)
        y = np.asarray(targets)
        loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        if loss < best_loss:
            best_loss, best = loss, params
    assert best is not None
    return best


def predict_bkt(samples: list[dict[str, np.ndarray]], params: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[float] = []
    targets: list[float] = []
    for sample in samples:
        pred, target = _sequence_predictions(sample, params)
        predictions.extend(pred)
        targets.extend(target)
    return np.asarray(predictions), np.asarray(targets)

