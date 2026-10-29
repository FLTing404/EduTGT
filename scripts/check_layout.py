"""Check the repository layout and preserved result files without training."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    python_count = 0
    for path in ROOT.rglob("*.py"):
        if any(part in {".git", ".local", ".venv", "venv", "env", "__pycache__"}
               for part in path.relative_to(ROOT).parts):
            continue
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        python_count += 1

    paths = load_module("edutgt_paths_check", ROOT / "EduTGT/paths.py")
    downstream = load_module(
        "downstream_paths_check", ROOT / "downstream_pass_prediction/src/paths_downstream.py"
    )
    check(paths.ROOT == ROOT / "EduTGT", "Incorrect main code directory")
    check(paths.OUTPUT_ROOT == ROOT / "results/link_prediction", "Incorrect main output directory")
    check(paths.RESULT_DIR == paths.OUTPUT_ROOT / "batches", "Incorrect batch output directory")
    check(paths.BASELINE_RESULTS_DIR == paths.OUTPUT_ROOT / "baselines", "Incorrect baseline output directory")
    check(downstream.REPO_ROOT == ROOT, "Incorrect downstream repository root")
    check(downstream.CODE_ROOT == paths.ROOT, "Downstream imports point to the wrong code directory")
    check(downstream.PRETRAIN_DIR == paths.PRETRAIN_DIR, "Pretrained model paths disagree")
    check(downstream.OUTPUT_ROOT == ROOT / "results/pass_prediction", "Incorrect downstream output directory")
    check(downstream.processed_dir("data_AAA") == paths.PROCESSED_DATA_DIR / "data_AAA",
          "Downstream processed-data path disagrees with main code")

    for relative in ["results/summary/对比试验.csv", "results/summary/消融实验.csv",
                     "results/link_prediction/pretrain", "results/link_prediction/batches",
                     "results/pass_prediction", "results/archive/legacy", "results/archive/2026-03-26",
                     "EduTGT/data/scripts/preprocess_oulad_for_contratgt.py"]:
        check((ROOT / relative).exists(), f"Missing repository entry: {relative}")
    for old in ["result", "EduTGT/outputs", "EduTGT/baseline/results", "downstream_pass_prediction/outputs"]:
        check(not (ROOT / old).exists(), f"Obsolete output directory: {old}")

    manifest_count = 0
    optional_data_missing = 0
    for path in (ROOT / "downstream_pass_prediction/data").glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        check((ROOT / manifest["labels_csv"]).is_file(), f"Missing labels in {path}")
        check((ROOT / manifest["stats_json"]).is_file(), f"Missing metadata in {path}")
        for key in ["processed_dir", "ml_csv", "content_csv", "node_map_csv", "stats_json", "student_info_csv"]:
            relative = manifest[key]
            check(relative.startswith("EduTGT/data/"), f"Incorrect {key} in {path}")
            if not (ROOT / relative).exists():
                optional_data_missing += 1
        manifest_count += 1

    result_count = 0
    with (ROOT / "results/files.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            path = (ROOT / row["path"]).resolve()
            check(path.is_relative_to(ROOT / "results"), f"Result path outside results/: {path}")
            check(path.is_file(), f"Missing preserved result: {row['path']}")
            check(path.stat().st_size == int(row["bytes"]), f"Result size changed: {row['path']}")
            check(sha256(path) == row["sha256"], f"Result content changed: {row['path']}")
            result_count += 1

    print(f"PASS: {python_count} Python files, main/downstream paths, "
          f"{manifest_count} downstream manifests, {result_count} preserved result files.")
    print(f"Optional local data paths absent: {optional_data_missing} (large OULAD files are not tracked).")


if __name__ == "__main__":
    main()
