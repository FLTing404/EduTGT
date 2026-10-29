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


CANDIDATES = [
    {"name": "paper_literal", "dropout": 0.8, "learning_rate": 0.00025, "learning_rate_decay": 0.92, "hypergraph_layers": 3},
    {"name": "keep80_lr25", "dropout": 0.2, "learning_rate": 0.00025, "learning_rate_decay": 0.92, "hypergraph_layers": 3},
    {"name": "keep80_lr50", "dropout": 0.2, "learning_rate": 0.0005, "learning_rate_decay": 0.92, "hypergraph_layers": 3},
    {"name": "keep80_lr100", "dropout": 0.2, "learning_rate": 0.001, "learning_rate_decay": 0.92, "hypergraph_layers": 3},
    {"name": "drop50_lr50", "dropout": 0.5, "learning_rate": 0.0005, "learning_rate_decay": 0.92, "hypergraph_layers": 3},
    {"name": "keep80_two_layers", "dropout": 0.2, "learning_rate": 0.0005, "learning_rate_decay": 0.92, "hypergraph_layers": 2},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only FC-HGNN hyperparameter screen")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--dataset", choices=["assist12", "ednet", "oulad"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--candidates", help="Comma-separated candidate names; default runs all")
    args = parser.parse_args()

    base, _ = load_config(args.config)
    output = Path(args.output).resolve()
    prepared = Path(args.prepared_root).resolve() / args.dataset
    requested = set(args.candidates.split(",")) if args.candidates else None
    candidates = [candidate for candidate in CANDIDATES if requested is None or candidate["name"] in requested]
    if not candidates:
        raise SystemExit("No matching candidates")
    results = []
    for candidate in candidates:
        config = copy.deepcopy(base)
        config["training"].update(candidate)
        config["training"]["epochs"] = args.epochs
        config["training"]["patience"] = 3
        config["training"]["tune_threshold"] = True
        set_seed(int(config["seed"]))
        result = train_neural("fchgnn", prepared, config, evaluate_test=False)
        record = {"candidate": candidate, "result": result}
        results.append(record)
        write_json(output / f"{args.dataset}_{candidate['name']}.json", record)
        print(json.dumps({"name": candidate["name"], **result["validation"], "threshold": result["decision_threshold"]}))
    results.sort(key=lambda row: (row["result"]["validation"]["auc"], row["result"]["validation"]["f1"]), reverse=True)
    write_json(output / f"{args.dataset}_ranking.json", results)
    print("best=" + results[0]["candidate"]["name"])


if __name__ == "__main__":
    main()
