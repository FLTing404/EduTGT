#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fchgnn_repro.preprocess import prepare_dataset
from fchgnn_repro.train import run_experiments
from fchgnn_repro.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and reproduce KT baselines and FC-HGNN trends")
    parser.add_argument("command", choices=["prepare", "train", "all"])
    parser.add_argument("--config", default=str(ROOT / "configs/smoke.yaml"))
    parser.add_argument("--dataset", choices=["assist12", "ednet", "oulad", "xes3g5m", "assist17", "nips34", "all"], default="all")
    parser.add_argument("--model", choices=["bkt", "dkt", "saint", "fchgnn", "akt", "simplekt", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    if args.model != "all":
        config = copy.deepcopy(config)
        config["training"]["models"] = [args.model]
    names = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    print(f"config={config_path}")
    if args.command in {"prepare", "all"}:
        for name in names:
            metadata = prepare_dataset(name, config, force=args.force)
            print(f"prepared {name}: {json.dumps({k: v for k, v in metadata.items() if not k.endswith('_ids')}, ensure_ascii=False)}")
    if args.command in {"train", "all"}:
        for name in names:
            summary = run_experiments(name, config)
            print(f"results {name}: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
