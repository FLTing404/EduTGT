from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .bkt import fit_bkt, predict_bkt
from .data import SequenceDataset, load_graph
from .models import build_model
from .utils import choose_device, read_json, set_seed, write_json


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (y_prob >= lower) & (y_prob < upper if upper < 1.0 else y_prob <= upper)
        if selected.any():
            result += float(selected.mean()) * abs(float(y_true[selected].mean()) - float(y_prob[selected].mean()))
    return result


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int8)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "incorrect_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "incorrect_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "incorrect_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "intervention_rate": float((y_pred == 0).mean()),
        "incorrect_miss_rate": float(((y_true == 0) & (y_pred == 1)).sum() / max(int((y_true == 0).sum()), 1)),
        "false_alarm_rate": float(((y_true == 1) & (y_pred == 0)).sum() / max(int((y_true == 1).sum()), 1)),
        "cost_miss5_false_alarm1": float((5 * ((y_true == 0) & (y_pred == 1)).sum() + ((y_true == 1) & (y_pred == 0)).sum()) / len(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        result["auc"] = float(roc_auc_score(y_true, y_prob))
        result["pr_auc"] = float(average_precision_score(y_true, y_prob))
        result["incorrect_pr_auc"] = float(average_precision_score(1 - y_true, 1.0 - y_prob))
    else:
        result["auc"] = float("nan")
        result["pr_auc"] = float("nan")
        result["incorrect_pr_auc"] = float("nan")
    return result


def _threshold_scores(y_true: np.ndarray, y_prob: np.ndarray, candidates: np.ndarray, objective: str) -> np.ndarray:
    """Exact confusion counts at thresholds, without rescanning labels per candidate."""
    order = np.argsort(y_prob, kind="stable")
    y = np.asarray(y_true, dtype=np.int64)[order]
    p = np.asarray(y_prob)[order]
    prefix = np.concatenate(([0], np.cumsum(y)))
    cut = np.searchsorted(p, candidates, side="left")
    fn = prefix[cut].astype(np.float64)
    tn = cut.astype(np.float64) - fn
    tp = float(y.sum()) - fn
    fp = float(len(y) - y.sum()) - tn
    if objective in {"f1", "incorrect_f1"}:
        hits = tp if objective == "f1" else tn
        denominator = 2.0 * hits + fp + fn
        return np.divide(2.0 * hits, denominator, out=np.zeros_like(hits), where=denominator > 0)
    if objective == "mcc":
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0)
    if objective == "balanced_accuracy":
        recalls = []
        if y.sum():
            recalls.append(tp / float(y.sum()))
        if len(y) - y.sum():
            recalls.append(tn / float(len(y) - y.sum()))
        return np.mean(recalls, axis=0)
    raise ValueError(f"unknown threshold objective: {objective}")


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return best_threshold(y_true, y_prob, "f1")


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str, method: str = "grid") -> float:
    """Choose a threshold using validation data and an imbalance-aware objective."""
    if method == "exact":
        candidates = np.concatenate(([0.0], np.unique(y_prob).astype(np.float64), [np.nextafter(float(np.max(y_prob)), np.inf)]))
    elif method == "grid":
        candidates = np.linspace(0.10, 0.90, 161) if objective == "f1" else np.linspace(0.05, 0.95, 181)
    else:
        raise ValueError(f"unknown threshold search: {method}")
    scores = _threshold_scores(y_true, y_prob, candidates, objective)
    return float(candidates[int(np.argmax(scores))])


def imbalance_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mode: str,
    negative_weight: float = 1.0,
    positive_weight: float = 1.0,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Binary losses where label 0 denotes an incorrect response."""
    if mode == "auc_margin":
        positive = logits[target >= 0.5]
        negative = logits[target < 0.5]
        if positive.numel() and negative.numel():
            # E[(1 - s_pos + s_neg)^2], expanded to avoid a large pairwise matrix.
            pairwise = (
                1.0
                + positive.square().mean()
                + negative.square().mean()
                - 2.0 * positive.mean()
                + 2.0 * negative.mean()
                - 2.0 * positive.mean() * negative.mean()
            )
            calibration = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            return 0.5 * pairwise + 0.5 * calibration
    base = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = torch.where(target < 0.5, negative_weight, positive_weight)
    if mode == "focal":
        probability = torch.sigmoid(logits)
        correct_probability = torch.where(target >= 0.5, probability, 1.0 - probability)
        weights = weights * (1.0 - correct_probability).pow(focal_gamma)
    elif mode != "weighted_bce":
        raise ValueError(f"unknown imbalance loss: {mode}")
    return (base * weights).sum() / weights.sum().clamp_min(1e-12)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, prediction_sink: dict | None = None
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    saved: dict[str, list[np.ndarray]] = {}
    for batch in loader:
        batch = _move(batch, device)
        if hasattr(model, "prediction_details"):
            logits, uncertainty = model.prediction_details(batch)
        else:
            logits, uncertainty = model(batch), None
        mask = batch["mask"]
        target = batch["target"]
        total_loss += float(loss_fn(logits[mask], target[mask]).item())
        total += int(mask.sum().item())
        probabilities.append(torch.sigmoid(logits[mask]).cpu().numpy())
        targets.append(target[mask].cpu().numpy())
        if prediction_sink is not None:
            for key in ("user_idx", "event_idx", "next_item", "next_concept", "correct", "history_position"):
                if key in batch:
                    saved.setdefault(key, []).append(batch[key][mask].cpu().numpy())
        if uncertainty is not None:
            uncertainties.append(uncertainty[mask].cpu().numpy())
    target_array = np.concatenate(targets)
    probability_array = np.concatenate(probabilities)
    if prediction_sink is not None:
        prediction_sink.update({"target": target_array, "probability": probability_array})
        prediction_sink.update({key: np.concatenate(values) for key, values in saved.items()})
    diagnostics: dict[str, float] = {}
    if uncertainties:
        uncertainty_array = np.concatenate(uncertainties)
        errors = (probability_array >= 0.5) != target_array.astype(bool)
        diagnostics["mean_uncertainty"] = float(uncertainty_array.mean())
        diagnostics["mean_uncertainty_correct"] = float(uncertainty_array[~errors].mean()) if (~errors).any() else float("nan")
        diagnostics["mean_uncertainty_error"] = float(uncertainty_array[errors].mean()) if errors.any() else float("nan")
    return total_loss / max(total, 1), target_array, probability_array, diagnostics


@torch.no_grad()
def grouped_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for batch in loader:
        batch = _move(batch, device)
        logits = model(batch)
        mask = batch["mask"]
        targets.append(batch["target"][mask].cpu().numpy())
        probabilities.append(torch.sigmoid(logits[mask]).cpu().numpy())
        groups.append(batch["next_concept"][mask].cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities), np.concatenate(groups)


def shrinkage_group_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    global_threshold: float,
    shrinkage: float,
) -> dict[int, float]:
    """Estimate group MCC thresholds and shrink sparse groups toward the global threshold."""
    thresholds: dict[int, float] = {}
    for group in np.unique(groups):
        selected = groups == group
        group_y = y_true[selected]
        minority_count = int((group_y < 0.5).sum())
        if minority_count == 0 or minority_count == len(group_y):
            thresholds[int(group)] = global_threshold
            continue
        local = best_threshold(group_y, y_prob[selected], "mcc")
        reliability = minority_count / (minority_count + max(shrinkage, 0.0))
        thresholds[int(group)] = float(reliability * local + (1.0 - reliability) * global_threshold)
    return thresholds


def apply_group_thresholds(
    groups: np.ndarray, thresholds: dict[int, float], global_threshold: float
) -> np.ndarray:
    return np.array([thresholds.get(int(group), global_threshold) for group in groups], dtype=np.float64)


def train_neural(name: str, prepared_dir: Path, config: dict, evaluate_test: bool = True) -> dict:
    training = config["training"]
    sequence = config["sequence"]
    metadata = read_json(prepared_dir / "metadata.json")
    datasets = {
        split: SequenceDataset(prepared_dir, split, sequence["max_length"], sequence["min_length"])
        for split in ("train", "valid", "test")
    }
    train_targets = np.concatenate([sample["target"][sample["mask"]] for sample in datasets["train"].samples])
    negative_count = int((train_targets < 0.5).sum())
    positive_count = int((train_targets >= 0.5).sum())
    negative_weight = float(training.get("negative_weight", 1.0))
    positive_weight = float(training.get("positive_weight", 1.0))
    class_balance_beta = training.get("class_balance_beta")
    if training.get("class_weighting") == "inverse_frequency":
        negative_weight = (negative_count + positive_count) / max(2 * negative_count, 1)
        positive_weight = (negative_count + positive_count) / max(2 * positive_count, 1)
    if class_balance_beta is not None:
        beta = float(class_balance_beta)
        raw_negative = (1.0 - beta) / max(1.0 - beta**negative_count, 1e-12)
        raw_positive = (1.0 - beta) / max(1.0 - beta**positive_count, 1e-12)
        normalizer = (negative_count + positive_count) / (
            negative_count * raw_negative + positive_count * raw_positive
        )
        negative_weight = raw_negative * normalizer
        positive_weight = raw_positive * normalizer
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=split == "train",
            num_workers=int(training["num_workers"]),
        )
        for split, dataset in datasets.items()
    }
    edge_item, edge_concept, difficulty = load_graph(prepared_dir)
    kwargs = {
        "num_items": metadata["num_items"],
        "num_concepts": metadata["num_concepts"],
        "embedding_dim": int(training["embedding_dim"]),
        "hidden_dim": int(training["hidden_dim"]),
        "dropout": float(training["dropout"]),
        "hypergraph_layers": int(training["hypergraph_layers"]),
        "num_heads": int(training["num_heads"]),
        "max_length": int(sequence["max_length"]),
        "edge_item": torch.from_numpy(edge_item),
        "edge_concept": torch.from_numpy(edge_concept),
        "difficulty": torch.from_numpy(difficulty),
        "time_aware": bool(training.get("time_aware", False)),
        "multi_concept": bool(training.get("multi_concept", False)),
        "hypergraph_attention": bool(training.get("hypergraph_attention", False)),
        "dynamic_scale_fusion": bool(training.get("dynamic_scale_fusion", False)),
        "contrastive_weight": float(training.get("contrastive_weight", 0.0)),
        "contrastive_temperature": float(training.get("contrastive_temperature", 0.2)),
        "uncertainty_enabled": bool(training.get("uncertainty_enabled", False)),
        "uncertainty_kl_weight": float(training.get("uncertainty_kl_weight", 0.001)),
        "guided_diffusion": bool(training.get("guided_diffusion", False)),
        "diffusion_steps": int(training.get("diffusion_steps", 8)),
        "diffusion_loss_weight": float(training.get("diffusion_loss_weight", 0.05)),
        "diffusion_logit_weight": float(training.get("diffusion_logit_weight", 0.1)),
    }
    kwargs.update({key: training[key] for key in ("n_blocks", "d_ff", "final_fc_dim", "final_fc_dim2", "kq_same", "separate_qa", "l2", "emb_type") if key in training})
    device = choose_device()
    model = build_model(name, **kwargs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(training.get("learning_rate_decay", 1.0))
    )
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    early_stopping_metric = str(training.get("early_stopping_metric", "valid_loss"))
    best_selection = float("inf") if early_stopping_metric == "valid_loss" else -float("inf")
    best_valid = float("inf")
    stale = 0
    history = []
    started = time.time()
    loaded_checkpoint = training.get("load_checkpoint")
    if loaded_checkpoint:
        payload = torch.load(loaded_checkpoint, map_location=device, weights_only=False)
        if payload["model_name"] != name or payload["data_metadata"] != metadata:
            raise ValueError("Checkpoint model or prepared data metadata does not match")
        model.load_state_dict(payload["model_state"])
        best_state = copy.deepcopy(model.state_dict())
        history = payload["history"]
        best_selection = payload["best_selection"]
        best_valid = payload["best_valid_loss"]
    epoch_range = [] if loaded_checkpoint else range(1, int(training["epochs"]) + 1)
    for epoch in epoch_range:
        epoch_started = time.time()
        model.train()
        running = 0.0
        count = 0
        for batch in loaders["train"]:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            mask = batch["mask"]
            imbalance_mode = training.get("imbalance_loss")
            if imbalance_mode:
                logits = model(batch)
                selected_logits = logits[mask]
                selected_target = batch["target"][mask]
                epoch_negative_weight = negative_weight
                epoch_positive_weight = positive_weight
                defer_fraction = float(training.get("defer_reweighting_fraction", 0.0))
                if epoch <= int(training["epochs"]) * defer_fraction:
                    epoch_negative_weight = 1.0
                    epoch_positive_weight = 1.0
                loss = imbalance_loss(
                    selected_logits,
                    selected_target,
                    str(imbalance_mode),
                    epoch_negative_weight,
                    epoch_positive_weight,
                    float(training.get("focal_gamma", 2.0)),
                )
            elif hasattr(model, "training_objective"):
                loss, logits = model.training_objective(batch, epoch, int(training["epochs"]))
            else:
                logits = model(batch)
                loss = loss_fn(logits[mask], batch["target"][mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item()) * int(mask.sum().item())
            count += int(mask.sum().item())
        valid_loss, valid_y, valid_p, _ = evaluate(model, loaders["valid"], device)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": running / max(count, 1),
            "valid_loss": valid_loss,
            **metrics(valid_y, valid_p),
        }
        history.append(row)
        row["elapsed_seconds"] = time.time() - epoch_started
        if training.get("progress", False):
            print(json.dumps({"event": "epoch", "dataset": prepared_dir.name, "model": name, "seed": int(config["seed"]), **row}, ensure_ascii=False), flush=True)
        selection_score = valid_loss if early_stopping_metric == "valid_loss" else float(row[early_stopping_metric])
        improved = selection_score < best_selection if early_stopping_metric == "valid_loss" else selection_score > best_selection
        if improved:
            best_selection = selection_score
            best_valid = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(training["patience"]):
                break
        scheduler.step()
    assert best_state is not None
    model.load_state_dict(best_state)
    if training.get("checkpoint_path") and not loaded_checkpoint:
        checkpoint = Path(training["checkpoint_path"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_name": name, "model_state": model.state_dict(), "data_metadata": metadata, "training_settings": dict(training), "history": history, "best_selection": best_selection, "best_valid_loss": best_valid, "training_elapsed_seconds": time.time() - started}, checkpoint)
    predictions_dir = Path(training["predictions_dir"]) if training.get("predictions_dir") else None
    valid_sink: dict | None = {} if predictions_dir else None
    valid_loss, valid_y, valid_p, valid_diagnostics = evaluate(model, loaders["valid"], device, valid_sink)
    threshold_objective = str(training.get("threshold_objective", "f1"))
    threshold_method = str(training.get("threshold_search", "grid"))
    threshold = best_threshold(valid_y, valid_p, threshold_objective, threshold_method) if training.get("tune_threshold", False) else 0.5
    comparison_thresholds = {
        objective: best_threshold(valid_y, valid_p, objective, threshold_method) for objective in ("f1", "mcc", "balanced_accuracy", "incorrect_f1")
    }
    comparison_thresholds["fixed_0_5"] = 0.5
    comparison_thresholds["cost_5_to_1"] = 5.0 / 6.0
    if predictions_dir:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(predictions_dir / "valid.npz", **valid_sink)
    result = {
        "model": name,
        "device": str(device),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "sequence_counts": {key: len(value) for key, value in datasets.items()},
        "best_valid_loss": best_valid,
        "early_stopping_metric": early_stopping_metric,
        "best_selection_score": best_selection,
        "decision_threshold": threshold,
        "threshold_objective": threshold_objective,
        "threshold_search": threshold_method,
        "training_seed": int(config["seed"]),
        "data_split_seed": int(metadata["seed"]),
        "training_settings": dict(training),
        "checkpoint_path": training.get("checkpoint_path", loaded_checkpoint),
        "predictions_dir": str(predictions_dir) if predictions_dir else None,
        "evaluated_test": bool(evaluate_test),
        "class_counts": {"incorrect": negative_count, "correct": positive_count},
        "class_weights": {"incorrect": negative_weight, "correct": positive_weight},
        "validation": metrics(valid_y, valid_p, threshold),
        "validation_by_threshold": {
            objective: metrics(valid_y, valid_p, candidate) for objective, candidate in comparison_thresholds.items()
        },
        "comparison_thresholds": comparison_thresholds,
        "validation_diagnostics": valid_diagnostics,
        "history": history,
        "elapsed_seconds": time.time() - started + (float(payload.get("training_elapsed_seconds", 0.0)) if loaded_checkpoint else 0.0),
    }
    if evaluate_test:
        test_sink: dict | None = {} if predictions_dir else None
        test_loss, test_y, test_p, test_diagnostics = evaluate(model, loaders["test"], device, test_sink)
        if predictions_dir:
            np.savez_compressed(predictions_dir / "test.npz", **test_sink)
        result["test_loss"] = test_loss
        result["test"] = metrics(test_y, test_p, threshold)
        result["test_by_threshold"] = {
            objective: metrics(test_y, test_p, candidate) for objective, candidate in comparison_thresholds.items()
        }
        result["test_diagnostics"] = test_diagnostics
        if training.get("group_thresholding", False):
            valid_group_y, valid_group_p, valid_groups = grouped_predictions(model, loaders["valid"], device)
            test_group_y, test_group_p, test_groups = grouped_predictions(model, loaders["test"], device)
            shrinkage = float(training.get("group_threshold_shrinkage", 20.0))
            group_thresholds = shrinkage_group_thresholds(
                valid_group_y, valid_group_p, valid_groups, comparison_thresholds["mcc"], shrinkage
            )
            valid_threshold_array = apply_group_thresholds(
                valid_groups, group_thresholds, comparison_thresholds["mcc"]
            )
            test_threshold_array = apply_group_thresholds(
                test_groups, group_thresholds, comparison_thresholds["mcc"]
            )
            result["group_thresholds"] = group_thresholds
            result["group_threshold_shrinkage"] = shrinkage
            result["validation_group_threshold"] = metrics(valid_group_y, valid_group_p, valid_threshold_array)
            result["test_group_threshold"] = metrics(test_group_y, test_group_p, test_threshold_array)
    return result


def train_bkt(prepared_dir: Path, config: dict) -> dict:
    sequence = config["sequence"]
    datasets = {
        split: SequenceDataset(prepared_dir, split, sequence["max_length"], sequence["min_length"])
        for split in ("train", "valid", "test")
    }
    to_numpy = lambda dataset: dataset.samples
    params = fit_bkt(to_numpy(datasets["train"]), to_numpy(datasets["valid"]))
    probability, target = predict_bkt(to_numpy(datasets["test"]), params)
    return {
        "model": "bkt",
        "parameters": {"p_init": params[0], "p_learn": params[1], "p_guess": params[2], "p_slip": params[3]},
        "sequence_counts": {key: len(value) for key, value in datasets.items()},
        "test": metrics(target, probability),
    }


def run_experiments(dataset_name: str, config: dict) -> dict:
    set_seed(int(config["seed"]))
    prepared_root = Path(config.get("prepared_root", Path(config["artifact_root"]) / "prepared"))
    prepared_dir = prepared_root / dataset_name
    output_dir = Path(config["artifact_root"]) / "results" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name in config["training"]["models"]:
        result = train_bkt(prepared_dir, config) if name == "bkt" else train_neural(name, prepared_dir, config)
        all_results[name] = result
        write_json(output_dir / f"{name}.json", result)
    summary_path = output_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    summary.update({name: result["test"] for name, result in all_results.items()})
    write_json(summary_path, summary)
    return summary
