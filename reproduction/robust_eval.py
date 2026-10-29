#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fchgnn_repro.train import train_neural
from fchgnn_repro.utils import load_config, set_seed, write_json


METRICS = [
    "accuracy", "balanced_accuracy", "f1", "macro_f1", "mcc", "auc", "pr_auc",
    "incorrect_pr_auc", "incorrect_precision", "incorrect_recall", "incorrect_f1", "brier", "ece",
]


def aggregate(records: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for model in sorted({record["model"] for record in records}):
        rows = [record for record in records if record["model"] == model]
        summary[model] = {"runs": len(rows)}
        for threshold_name in ("f1", "mcc", "balanced_accuracy"):
            block = {}
            for metric in METRICS:
                values = np.array([row["result"]["test_by_threshold"][threshold_name][metric] for row in rows])
                block[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            thresholds = np.array([row["result"]["comparison_thresholds"][threshold_name] for row in rows])
            block["threshold"] = {
                "mean": float(thresholds.mean()),
                "std": float(thresholds.std(ddof=1)) if len(thresholds) > 1 else 0.0,
            }
            summary[model][threshold_name] = block
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed robustness evaluation")
    parser.add_argument("--config", default=str(ROOT / "configs/oulad_threshold_mcc.yaml"))
    parser.add_argument("--dataset", default="oulad", choices=["assist12", "ednet", "oulad", "xes3g5m", "assist17", "nips34"])
    parser.add_argument("--models", nargs="+", default=["dkt", "saint", "fchgnn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260824, 20260825, 20260826, 20260827, 20260828])
    parser.add_argument("--output", default=str(ROOT / "artifacts/robust_eval"))
    args = parser.parse_args()

    base, _ = load_config(args.config)
    prepared_root = Path(base.get("prepared_root", Path(base["artifact_root"]) / "prepared"))
    output = Path(args.output).resolve() / args.dataset
    records = []
    for model in args.models:
        for seed in args.seeds:
            config = copy.deepcopy(base)
            config["seed"] = seed
            config["training"]["models"] = [model]
            set_seed(seed)
            result = train_neural(model, prepared_root / args.dataset, config, evaluate_test=True)
            record = {"dataset": args.dataset, "model": model, "seed": seed, "result": result}
            records.append(record)
            write_json(output / f"{model}_seed{seed}.json", record)
            selected = result["test_by_threshold"]["mcc"]
            print(json.dumps({
                "model": model, "seed": seed, "threshold": result["comparison_thresholds"]["mcc"],
                "auc": selected["auc"], "incorrect_pr_auc": selected["incorrect_pr_auc"],
                "balanced_accuracy": selected["balanced_accuracy"], "mcc": selected["mcc"],
                "incorrect_recall": selected["incorrect_recall"], "incorrect_f1": selected["incorrect_f1"],
            }, ensure_ascii=False))
    summary = aggregate(records)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
