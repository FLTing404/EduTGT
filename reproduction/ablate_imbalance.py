#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fchgnn_repro.train import train_neural
from fchgnn_repro.utils import load_config, set_seed, write_json


STAGES = [
    ("baseline", {"threshold_objective": "f1"}),
    ("threshold_mcc", {"threshold_objective": "mcc"}),
    ("weighted_bce", {"threshold_objective": "mcc", "early_stopping_metric": "incorrect_pr_auc", "imbalance_loss": "weighted_bce", "negative_weight": 4.0}),
    ("focal", {"threshold_objective": "mcc", "early_stopping_metric": "incorrect_pr_auc", "imbalance_loss": "focal", "negative_weight": 4.0, "focal_gamma": 2.0}),
    ("class_balanced_focal", {"threshold_objective": "mcc", "early_stopping_metric": "incorrect_pr_auc", "imbalance_loss": "focal", "class_balance_beta": 0.99999, "focal_gamma": 2.0}),
    ("deferred_class_balanced_focal", {"threshold_objective": "mcc", "early_stopping_metric": "incorrect_pr_auc", "imbalance_loss": "focal", "class_balance_beta": 0.99999, "focal_gamma": 2.0, "defer_reweighting_fraction": 0.5}),
    ("auc_margin", {"threshold_objective": "mcc", "early_stopping_metric": "incorrect_pr_auc", "imbalance_loss": "auc_margin"}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="OULAD class-imbalance ablations")
    parser.add_argument("--config", default=str(ROOT / "configs/tuned.yaml"))
    parser.add_argument("--dataset", choices=["assist12", "ednet", "oulad", "xes3g5m", "assist17", "nips34"], default="oulad")
    parser.add_argument("--output", default=str(ROOT / "artifacts/imbalance_ablations"))
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    base, _ = load_config(args.config)
    prepared_root = Path(base.get("prepared_root", Path(base["artifact_root"]) / "prepared"))
    output = Path(args.output).resolve() / args.dataset
    records = []
    for stage, settings in STAGES:
        config = copy.deepcopy(base)
        config["training"].update(settings)
        config["training"]["models"] = ["fchgnn"]
        config["training"]["epochs"] = args.epochs
        config["training"]["tune_threshold"] = True
        set_seed(int(config["seed"]))
        result = train_neural("fchgnn", prepared_root / args.dataset, config, evaluate_test=False)
        record = {"stage": stage, "settings": settings, "result": result}
        records.append(record)
        write_json(output / f"{stage}.json", record)
        report = {key: result["validation"][key] for key in (
            "auc", "incorrect_pr_auc", "balanced_accuracy", "mcc", "incorrect_precision", "incorrect_recall", "incorrect_f1"
        )}
        print(json.dumps({"stage": stage, "threshold": result["decision_threshold"], **report}, ensure_ascii=False))

    ranked = sorted(records, key=lambda row: (
        row["result"]["validation"]["incorrect_pr_auc"], row["result"]["validation"]["mcc"]
    ), reverse=True)
    write_json(output / "ranking.json", ranked)
    print("best=" + ranked[0]["stage"])


if __name__ == "__main__":
    main()
