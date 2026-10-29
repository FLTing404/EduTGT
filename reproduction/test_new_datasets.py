#!/usr/bin/env python3
"""Validation-first, checkpointed experiments for both requested research directions."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from fchgnn_repro.preprocess import prepare_dataset
from fchgnn_repro.train import train_neural
from fchgnn_repro.utils import load_config, read_json, set_seed, write_json

VARIANTS = ["dkt", "akt", "simplekt", "fchgnn", "fchgnn_enhanced", "fchgnn_nohg", "fchgnn_weighted"]
SEEDS = [20261004, 20261005, 20261006]


def variant_settings(variant: str, dataset: str) -> tuple[str, dict]:
    model = variant if variant in {"dkt", "akt", "simplekt"} else "fchgnn"
    settings = {"time_aware": False, "multi_concept": False, "contrastive_weight": 0.0,
                "hypergraph_attention": False, "dynamic_scale_fusion": False,
                "uncertainty_enabled": False, "guided_diffusion": False}
    if variant in {"fchgnn_enhanced", "fchgnn_nohg"}:
        settings.update(time_aware=dataset != "nips34", multi_concept=True,
                        hypergraph_attention=True, dynamic_scale_fusion=True,
                        contrastive_weight=0.02, contrastive_temperature=0.2)
    if variant == "fchgnn_nohg":
        settings["hypergraph_layers"] = 0
    if variant == "fchgnn_weighted":
        settings.update(imbalance_loss="weighted_bce", class_weighting="inverse_frequency")
    return model, settings


def record_path(output: Path, phase: str, dataset: str, variant: str, seed: int) -> Path:
    return output / phase / dataset / f"{variant}_seed{seed}.json"


def execute(config: dict, output: Path, dataset: str, variant: str, seed: int, phase: str) -> dict:
    path = record_path(output, phase, dataset, variant, seed)
    if path.exists():
        record = read_json(path)
        if phase == "screen" or record["result"].get("evaluated_test"):
            print(json.dumps({"event": "reuse", "phase": phase, "dataset": dataset, "variant": variant, "seed": seed}), flush=True)
            return record
    run_config = copy.deepcopy(config)
    run_config["seed"] = seed
    model, settings = variant_settings(variant, dataset)
    run_config["training"].update(settings)
    checkpoint = output / "checkpoints" / dataset / f"{variant}_seed{seed}.pt"
    run_config["training"].update(checkpoint_path=str(checkpoint), predictions_dir=str(output / "predictions" / dataset / f"{variant}_seed{seed}"))
    if phase == "final":
        if not record_path(output, "screen", dataset, variant, seed).exists() or not checkpoint.exists():
            raise RuntimeError("Final evaluation requires a completed validation-only training checkpoint")
        run_config["training"]["load_checkpoint"] = str(checkpoint)
    set_seed(seed)
    print(json.dumps({"event": "start", "phase": phase, "dataset": dataset, "variant": variant, "seed": seed}), flush=True)
    result = train_neural(model, Path(config["prepared_root"]) / dataset, run_config, evaluate_test=phase == "final")
    record = {"dataset": dataset, "variant": variant, "model": model, "seed": seed,
              "settings": settings, "result": result}
    write_json(path, record)
    print(json.dumps({"event": "complete", "phase": phase, "dataset": dataset, "variant": variant,
                      "seed": seed, "validation_auc": result["validation"]["auc"],
                      "epochs": len(result["history"]), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return record


def freeze_plan(output: Path, datasets: list[str], variants: list[str], seeds: list[int]) -> dict:
    records = [read_json(record_path(output, "screen", d, v, s)) for d in datasets for v in variants for s in seeds]
    selections = {}
    for dataset in datasets:
        means = {variant: sum(r["result"]["validation"]["auc"] for r in records if r["dataset"] == dataset and r["variant"] == variant) / len(seeds) for variant in variants}
        baselines = [v for v in variants if v in {"dkt", "akt", "simplekt"}]
        graphs = [v for v in variants if v in {"fchgnn", "fchgnn_enhanced"}]
        selections[dataset] = {"validation_auc_mean": means,
                               "best_baseline": max(baselines, key=means.get) if baselines else None,
                               "best_hypergraph": max(graphs, key=means.get) if graphs else None}
    payload = {"selection_uses": "validation mean AUC across fixed training seeds only; no test metrics",
               "datasets": datasets, "variants": variants, "seeds": seeds, "selections": selections,
               "frozen_before_test": True}
    write_json(output / "frozen_plan.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "validate", "test", "all", "sensitivity"])
    parser.add_argument("--config", default=str(ROOT / "configs/new_datasets_dual.yaml"))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    output = Path(config["artifact_root"])
    config.setdefault("prepared_root", str(output / "prepared"))
    datasets = args.datasets or list(config["datasets"])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    protocol = {"date": "2026-10-04", "config": str(config_path), "config_snapshot": config,
                "fixed_split_seed": config["seed"], "training_seeds": args.seeds,
                "datasets": datasets, "variants": args.variants,
                "selection": "all validation-only training completes before frozen selection and any final test",
                "sota_scope": "controlled feasibility comparison to author-official AKT/simpleKT; not a published SOTA reproduction",
                "information": "base models use representative first leaf KC; enhanced and noHG both receive all KC metadata, FM, difficulty, contrastive/time; noHG only removes propagation layers",
                "primary_loss": "BCE (AKT retains official Rasch L2); weighted FC-HGNN is separate train-frequency loss control",
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src").rglob("*.py")}}
    if (output / "protocol.json").exists():
        old = read_json(output / "protocol.json")
        if old["config_snapshot"] != config:
            raise RuntimeError("Existing experiment root uses a different configuration; choose a separate artifact_root")
    else:
        write_json(output / "protocol.json", protocol)
    if args.phase in {"prepare", "all"}:
        for dataset in datasets:
            metadata = prepare_dataset(dataset, config)
            print(json.dumps({"event": "prepared", "dataset": dataset,
                              **{k: v for k, v in metadata.items() if not k.endswith("_ids")}}, ensure_ascii=False), flush=True)
        if args.phase == "prepare":
            return
    if args.phase in {"validate", "all"}:
        for dataset in datasets:
            for variant in args.variants:
                for seed in args.seeds:
                    execute(config, output, dataset, variant, seed, "screen")
    if args.phase in {"test", "all"}:
        freeze_plan(output, datasets, args.variants, args.seeds)
        for dataset in datasets:
            for variant in args.variants:
                for seed in args.seeds:
                    execute(config, output, dataset, variant, seed, "final")
        write_json(output / "completion.json", {"completed": True, "datasets": datasets,
                   "variants": args.variants, "seeds": args.seeds,
                   "final_runs": len(datasets) * len(args.variants) * len(args.seeds)})
    if args.phase == "sensitivity":
        sensitivity = copy.deepcopy(config)
        sensitivity["prepared_root"] = str(output / "sensitivity_prepared")
        sensitivity["datasets"]["nips34"]["tie_order"] = "random"
        prepare_dataset("nips34", sensitivity)
        for variant in ("akt", "fchgnn"):
            execute(sensitivity, output / "tie_sensitivity", "nips34", variant, args.seeds[0], "screen")
        freeze_plan(output / "tie_sensitivity", ["nips34"], ["akt", "fchgnn"], [args.seeds[0]])
        for variant in ("akt", "fchgnn"):
            execute(sensitivity, output / "tie_sensitivity", "nips34", variant, args.seeds[0], "final")


if __name__ == "__main__":
    main()
