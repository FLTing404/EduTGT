#!/usr/bin/env python3
"""Export test AUC and controlled hypergraph-ablation figures from saved analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("dkt", "akt", "simplekt", "fchgnn", "enhanced", "nohg", "weighted")
LABELS = ("DKT", "AKT", "simpleKT", "FC base", "FC enhanced", "FC noHG", "FC weighted")


def plot(summary: dict, output: Path) -> list[str]:
    if summary["status"] != "complete":
        raise ValueError("Final findings figures require all planned test evaluations")
    output.mkdir(parents=True, exist_ok=True)
    datasets = list(summary["datasets"])
    seeds = summary["progress"]["required_training_seed_count"]
    prefix = "TECHNICAL SMOKE ONLY: " if summary.get("scope") == "smoke" else ""
    fig, axes = plt.subplots(1, len(datasets), figsize=(13, 4.5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        rows = summary["datasets"][dataset]["variants"]
        values = [rows[v]["test_by_threshold"]["mcc"]["auc"] for v in VARIANTS]
        ax.bar(np.arange(len(values)), [x["mean"] for x in values], yerr=[x["std"] or 0 for x in values], capsize=3,
               color=["#7b9cb5"] * 3 + ["#3c7c65", "#639f7c", "#a7b5a9", "#b98e56"])
        ax.set_xticks(np.arange(len(values)), LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(dataset)
        ax.set_ylabel("Test ROC AUC")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    caption = f"Fixed student split; mean +/- sample SD across {seeds} training seeds" if seeds > 1 else "Fixed student split; one training seed, SD unavailable"
    fig.suptitle(prefix + caption)
    fig.tight_layout()
    auc = output / "test_auc_by_model.png"
    fig.savefig(auc, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    for ax, metric in zip(axes[0], ("auc", "incorrect_pr_auc")):
        for y, dataset in enumerate(datasets):
            values = summary["datasets"][dataset]["paired_comparisons"]["enhanced_vs_nohg"]["metrics"][metric]
            lo, hi = values["ci95"]
            color = "#28734b" if lo > 0 else "#ab463b" if hi < 0 else "#67737d"
            ax.plot([lo, hi], [y, y], color=color, linewidth=2)
            ax.plot(values["mean_difference"], y, "o", color=color)
            ax.annotate(f"{values['mean_difference']:+.5f} [{lo:+.5f}, {hi:+.5f}]", ((lo + hi) / 2, y), xytext=(0, 9),
                        textcoords="offset points", fontsize=8, ha="center",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 1})
        ax.axvline(0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_yticks(range(len(datasets)), datasets)
        ax.set_ylim(-0.6, len(datasets) - 0.4)
        ax.margins(x=0.5)
        ax.set_xlabel(f"Enhanced minus noHG: {metric}")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(prefix + "Matched inputs; paired student bootstrap 95% CI, conditional on trained models")
    fig.tight_layout()
    graph = output / "hypergraph_contribution.png"
    fig.savefig(graph, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [str(auc), str(graph)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "artifacts/new_datasets_dual")
    args = parser.parse_args()
    directory = args.root.resolve() / "analysis"
    summary = json.loads((directory / "summary.json").read_text())
    print(json.dumps({"figures": plot(summary, directory)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
