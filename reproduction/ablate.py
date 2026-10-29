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
    ("baseline", {}),
    ("time", {"time_aware": True}),
    ("multi_concept", {"multi_concept": True}),
    ("contrastive", {"contrastive_weight": 0.02, "contrastive_temperature": 0.2}),
    ("attentive_hypergraph", {"hypergraph_attention": True, "dynamic_scale_fusion": True}),
    ("uncertainty", {"uncertainty_enabled": True, "uncertainty_kl_weight": 0.001}),
    (
        "guided_diffusion",
        {
            "guided_diffusion": True,
            "diffusion_steps": 8,
            "diffusion_loss_weight": 0.05,
            "diffusion_logit_weight": 0.1,
        },
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cumulative validation-only FC-HGNN ablation")
    parser.add_argument("--config", default=str(ROOT / "configs/tuned.yaml"))
    parser.add_argument("--dataset", choices=["assist12", "ednet", "oulad", "xes3g5m", "assist17", "nips34"], required=True)
    parser.add_argument("--output", default=str(ROOT / "artifacts/ablations"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--through", choices=[name for name, _ in STAGES], default="guided_diffusion")
    args = parser.parse_args()

    base, _ = load_config(args.config)
    prepared_root = Path(base.get("prepared_root", Path(base["artifact_root"]) / "prepared"))
    prepared_dir = prepared_root / args.dataset
    output = Path(args.output).resolve() / args.dataset
    cumulative: dict[str, object] = {
        "time_aware": False,
        "multi_concept": False,
        "contrastive_weight": 0.0,
        "hypergraph_attention": False,
        "dynamic_scale_fusion": False,
        "uncertainty_enabled": False,
        "guided_diffusion": False,
    }
    records = []
    for stage, additions in STAGES:
        cumulative.update(additions)
        config = copy.deepcopy(base)
        config["training"].update(cumulative)
        config["training"]["models"] = ["fchgnn"]
        config["training"]["tune_threshold"] = True
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        set_seed(int(config["seed"]))
        result = train_neural("fchgnn", prepared_dir, config, evaluate_test=False)
        record = {"stage": stage, "settings": copy.deepcopy(cumulative), "result": result}
        records.append(record)
        write_json(output / f"{stage}.json", record)
        print(json.dumps({"stage": stage, **result["validation"]}, ensure_ascii=False))
        if stage == args.through:
            break

    ranked = sorted(
        records,
        key=lambda row: (row["result"]["validation"]["auc"], row["result"]["validation"]["f1"]),
        reverse=True,
    )
    write_json(output / "ranking.json", ranked)
    print("best=" + ranked[0]["stage"])


if __name__ == "__main__":
    main()
