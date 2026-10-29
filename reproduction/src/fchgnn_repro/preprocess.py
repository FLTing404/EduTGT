from __future__ import annotations

import ast
import io
import json
import random
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import write_json


COMMON_COLUMNS = ["user_raw", "item_raw", "concepts_raw", "correct", "timestamp"]
SPLIT_NAMES = ("train", "valid", "test")


def _random_user_splits(users: Iterable[str], split: dict[str, float], seed: int) -> dict[str, str]:
    """Assign all eligible students before any sampling, without reading labels."""
    users = sorted(set(map(str, users)))
    order = np.arange(len(users))
    np.random.default_rng(seed).shuffle(order)
    n_train = int(len(users) * float(split["train"]))
    n_valid = int(len(users) * float(split["valid"]))
    groups = (order[:n_train], order[n_train : n_train + n_valid], order[n_train + n_valid :])
    return {users[int(index)]: name for name, group in zip(SPLIT_NAMES, groups) for index in group}


def _sample_fixed_splits(assignments: dict[str, str], size: int | None, seed: int) -> set[str]:
    """Allocate a total user budget proportionally across already fixed splits."""
    groups = {name: sorted(user for user, value in assignments.items() if value == name) for name in SPLIT_NAMES}
    total = len(assignments)
    if size is None or int(size) >= total:
        return set(assignments)
    size = int(size)
    if size < 3:
        raise ValueError("sample_users must be at least 3 for three student-disjoint splits")
    exact = {name: size * len(values) / total for name, values in groups.items()}
    quotas = {name: min(len(groups[name]), int(exact[name])) for name in SPLIT_NAMES}
    for name in sorted(SPLIT_NAMES, key=lambda key: exact[key] - quotas[key], reverse=True):
        if sum(quotas.values()) < size and quotas[name] < len(groups[name]):
            quotas[name] += 1
    # Small smoke budgets should still contain each non-empty split.
    for name in SPLIT_NAMES:
        if groups[name] and quotas[name] == 0:
            donor = max(SPLIT_NAMES, key=lambda key: quotas[key])
            quotas[donor] -= 1
            quotas[name] = 1
    selected: set[str] = set()
    for index, name in enumerate(SPLIT_NAMES):
        selected.update(_stable_sample(groups[name], quotas[name], seed + index))
    return selected


def _fixed_student_selection(
    frame: pd.DataFrame, sample_users: int | None, split: dict[str, float], seed: int
) -> pd.DataFrame:
    counts = frame["user_raw"].value_counts()
    eligible = counts[counts >= 3].index.astype(str)
    assignments = _random_user_splits(eligible, split, seed)
    selected = _sample_fixed_splits(assignments, sample_users, seed)
    attrs = frame.attrs.copy()
    attrs.setdefault("filter_statistics", {})["eligible_students_before_sampling"] = len(eligible)
    attrs["full_split_user_counts"] = dict(Counter(assignments.values()))
    frame = frame[frame["user_raw"].isin(selected)].copy()
    frame["split_raw"] = frame["user_raw"].map(assignments)
    frame.attrs.update(attrs)
    return frame


def _stable_sample(values: Iterable[str], size: int | None, seed: int) -> set[str]:
    values = sorted(set(map(str, values)))
    if size is None or size >= len(values):
        return set(values)
    return set(random.Random(seed).sample(values, size))


def _prepare_assist(data_root: Path, sample_users: int | None, seed: int) -> pd.DataFrame:
    archive = data_root / "ASSISTments2012_2013/raw/2012-2013-data-with-predictions-4-final.zip"
    member = "2012-2013-data-with-predictions-4-final.csv"
    usecols = ["user_id", "problem_id", "skill_id", "correct", "start_time"]
    counts: Counter[str] = Counter()
    with zipfile.ZipFile(archive) as zf, zf.open(member) as raw:
        for chunk in pd.read_csv(raw, usecols=usecols, dtype=str, chunksize=250_000):
            valid = chunk["skill_id"].notna() & chunk["skill_id"].ne("")
            counts.update(chunk.loc[valid, "user_id"].dropna().astype(str))
    eligible = {user for user, count in counts.items() if count >= 3}
    selected = _stable_sample(eligible, sample_users, seed)
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(archive) as zf, zf.open(member) as raw:
        for chunk in pd.read_csv(raw, usecols=usecols, dtype=str, chunksize=250_000):
            mask = (
                chunk["user_id"].isin(selected)
                & chunk["skill_id"].notna()
                & chunk["skill_id"].ne("")
                & chunk["correct"].isin(["0", "1"])
            )
            part = chunk.loc[mask].rename(
                columns={
                    "user_id": "user_raw",
                    "problem_id": "item_raw",
                    "skill_id": "concepts_raw",
                }
            )
            part["correct"] = part["correct"].astype(np.int8)
            part["timestamp"] = pd.to_datetime(part["start_time"], errors="coerce").astype("int64")
            frames.append(part[COMMON_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def _prepare_ednet(data_root: Path, sample_users: int | None, seed: int) -> pd.DataFrame:
    archive = data_root / "EdNet/raw/EdNet-KT1.zip"
    questions = pd.read_csv(data_root / "EdNet/raw/questions.csv", dtype=str)
    qmeta = questions.set_index("question_id")
    answer = qmeta["correct_answer"].to_dict()
    tags = qmeta["tags"].fillna("unknown").to_dict()
    with zipfile.ZipFile(archive) as zf:
        members = sorted(name for name in zf.namelist() if name.endswith(".csv"))
        candidates = members.copy()
        random.Random(seed).shuffle(candidates)
        frames: list[pd.DataFrame] = []
        for member in candidates:
            with zf.open(member) as raw:
                part = pd.read_csv(raw, dtype=str)
            if len(part) < 3:
                continue
            part = part[part["question_id"].isin(answer)].copy()
            if len(part) < 3:
                continue
            part["user_raw"] = Path(member).stem
            part["item_raw"] = part["question_id"]
            part["concepts_raw"] = part["question_id"].map(tags)
            part["correct"] = (part["user_answer"] == part["question_id"].map(answer)).astype(np.int8)
            part["timestamp"] = pd.to_numeric(part["timestamp"], errors="coerce").fillna(0).astype("int64")
            frames.append(part[COMMON_COLUMNS])
            if sample_users is not None and len(frames) >= sample_users:
                break
    if not frames:
        raise RuntimeError("EdNet selection produced no usable user sequences")
    return pd.concat(frames, ignore_index=True)


def _prepare_oulad(data_root: Path, sample_users: int | None, seed: int) -> pd.DataFrame:
    archive = data_root / "OULAD/raw/anonymisedData.zip"
    with zipfile.ZipFile(archive) as zf:
        assessments = pd.read_csv(zf.open("assessments.csv"))
        responses = pd.read_csv(zf.open("studentAssessment.csv"))
    merged = responses.merge(assessments, on="id_assessment", how="inner", validate="many_to_one")
    merged = merged[merged["score"].notna()].copy()
    merged["user_raw"] = (
        merged["id_student"].astype(str)
        + ":"
        + merged["code_module"].astype(str)
        + ":"
        + merged["code_presentation"].astype(str)
    )
    counts = merged["user_raw"].value_counts()
    eligible = counts[counts >= 3].index.astype(str)
    selected = _stable_sample(eligible, sample_users, seed)
    merged = merged[merged["user_raw"].isin(selected)].copy()
    merged["item_raw"] = (
        merged["code_module"].astype(str)
        + ":"
        + merged["code_presentation"].astype(str)
        + ":"
        + merged["id_assessment"].astype(str)
    )
    merged["concepts_raw"] = merged["code_module"].astype(str) + ":" + merged["code_presentation"].astype(str)
    merged["correct"] = (merged["score"].astype(float) >= 40.0).astype(np.int8)
    merged["timestamp"] = merged["date_submitted"].fillna(merged["date"]).fillna(0).astype("int64")
    return merged[COMMON_COLUMNS]


def _prepare_xes3g5m(data_root: Path, sample_users: int | None, seed: int) -> pd.DataFrame:
    archive = data_root / "XES3G5M/raw/XES3G5M.tar.gz"
    sources = {
        "train_valid_sequences_quelevel.csv": "train_valid",
        "test_quelevel.csv": "test",
    }
    sequence_frames: list[pd.DataFrame] = []
    # Streaming avoids extracting or modifying the downloaded release and skips images.
    with tarfile.open(archive, "r|gz") as release:
        for member in release:
            source = sources.get(Path(member.name).name)
            if source is None or not member.isfile():
                continue
            table = pd.read_csv(io.BytesIO(release.extractfile(member).read()), dtype=str)
            table["source_raw"] = source
            table["source_row"] = np.arange(len(table), dtype=np.int64)
            sequence_frames.append(table)
    if len(sequence_frames) != 2:
        raise ValueError("XES3G5M release must contain both official question-level split files")
    sequences = pd.concat(sequence_frames, ignore_index=True)
    sequences["split_raw"] = np.where(
        sequences["source_raw"].eq("test"), "test", np.where(sequences["fold"].eq("0"), "valid", "train")
    )
    if sequences.groupby("uid")["split_raw"].nunique().max() != 1:
        raise ValueError("XES3G5M official split assigns a student to multiple train/valid/test partitions")
    if not sequences.loc[sequences["source_raw"].eq("train_valid"), "fold"].isin(["0", "1", "2", "3", "4"]).all():
        raise ValueError("Unexpected XES3G5M training fold identifier")
    assignments = sequences.drop_duplicates("uid").set_index("uid")["split_raw"].to_dict()
    selected = _sample_fixed_splits(assignments, sample_users, seed)
    source_sequence_count = len(sequences)
    sequences = sequences[sequences["uid"].isin(selected)]
    records: list[tuple] = []
    dropped_padding = 0
    timestamp_anomalies = 0
    for row in sequences.to_dict("records"):
        columns = {name: str(row[name]).split(",") for name in ("questions", "concepts", "responses", "timestamps")}
        masks = str(row.get("selectmasks", "")).split(",") if row["source_raw"] == "train_valid" else ["1"] * len(columns["questions"])
        lengths = {len(value) for value in columns.values()}
        if len(lengths) != 1 or len(masks) != len(columns["questions"]):
            raise ValueError(f"Misaligned XES3G5M sequence fields for uid={row['uid']}")
        sequence_id = f"{row['source_raw']}:{row['source_row']}"
        for order, (question, concept, response, timestamp) in enumerate(zip(*columns.values())):
            if question == "-1" or concept == "-1" or response not in {"0", "1"}:
                dropped_padding += 1
                continue
            value = int(timestamp) if timestamp not in {"", "nan", "-1"} else 0
            # Seven released tokens date to 2001 amid 2020/2021 interactions.
            # Keep author event order; mask unreliable gaps instead of sorting by these dates.
            valid_time = 1_262_304_000_000 <= value <= 1_924_992_000_000
            timestamp_anomalies += int(not valid_time)
            records.append((
                row["uid"], question, concept.replace("_", ";"), int(response), value,
                sequence_id, order, row["split_raw"], valid_time, masks[order] == "1",
            ))
    frame = pd.DataFrame(records, columns=COMMON_COLUMNS + ["sequence_raw", "order_raw", "split_raw", "time_valid", "target_valid"])
    frame.attrs.update({
        "protocol": {
            "source": "official question_level/train_valid_sequences_quelevel.csv and test_quelevel.csv",
            "split": "official test students retained; official training fold 0 valid; folds 1-4 train",
            "sequence": "each released row is an independent author segment; no cross-segment transitions",
            "target_mask": "released selectmasks shifted to next-response targets; official test has all valid targets",
            "ordering": "author token order, regardless of timestamp anomalies",
            "concepts": "all underscore-separated question concepts preserved in hypergraph; first concept used by single-concept baselines",
            "time": "milliseconds; dates outside 2010-2030 or nonpositive/backward/>365-day gaps masked to zero",
            "sampling": "student budget allocated proportionally after fixed official split; null means full release",
            "graph": "question-concept metadata uses all selected splits; labels used only for training item difficulty",
        },
        "filter_statistics": {
            "source_sequences": source_sequence_count,
            "selected_source_sequences": int(len(sequences)),
            "dropped_padding_or_invalid_label_tokens": dropped_padding,
            "selected_timestamp_anomaly_tokens": timestamp_anomalies,
            "eligible_students_before_sampling": len(assignments),
        },
        "full_split_user_counts": dict(Counter(assignments.values())),
    })
    return frame


def _prepare_assist17(
    data_root: Path, sample_users: int | None, seed: int, split: dict[str, float]
) -> pd.DataFrame:
    archive = data_root / "ASSISTments2017/raw/anonymized_full_release_competition_dataset.zip"
    columns = ["studentId", "problemId", "skill", "correct", "original", "hint", "startTime", "action_num"]
    with zipfile.ZipFile(archive) as release:
        members = [name for name in release.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError("ASSISTments2017 archive must contain one interaction CSV")
        raw = pd.read_csv(release.open(members[0]), usecols=columns, dtype=str)
    valid_skill = raw["skill"].notna() & raw["skill"].str.strip().ne("") & raw["skill"].str.lower().ne("noskill")
    original = raw["original"].eq("1")
    no_hint = raw["hint"].eq("0")
    binary = raw["correct"].isin(["0", "1"])
    keep = valid_skill & original & no_hint & binary
    frame = raw.loc[keep].rename(columns={"studentId": "user_raw", "problemId": "item_raw", "skill": "concepts_raw"}).copy()
    frame["correct"] = frame["correct"].astype(np.int8)
    frame["timestamp"] = pd.to_numeric(frame["startTime"], errors="coerce").fillna(0).astype(np.int64)
    frame["action_num"] = pd.to_numeric(frame["action_num"], errors="coerce").fillna(-1).astype(np.int64)
    frame = frame.sort_values(["user_raw", "timestamp", "action_num"], kind="mergesort")
    frame["order_raw"] = frame.groupby("user_raw", sort=False).cumcount()
    frame.attrs.update({
        "protocol": {
            "source": "anonymized_full_release_competition_dataset.csv",
            "split": "eligible students randomly assigned by seed to configured 70/15/15 before sampling",
            "filters": "known skill, binary label, original=1, hint=0",
            "ordering": "startTime epoch seconds, action_num numeric, then stable source-row order",
            "label_limitation": "original=1 excludes scaffolds and hint=0 excludes help actions; repeated original attempts may remain, so this is not a strict first-attempt protocol",
            "sampling": "student budget allocated proportionally after fixed student split; null means full eligible data",
            "graph": "question-concept metadata uses all selected splits; labels used only for training item difficulty",
        },
        "filter_statistics": {
            "source_interactions": int(len(raw)),
            "missing_or_noskill_interactions": int((~valid_skill).sum()),
            "nonoriginal_interactions": int((~original).sum()),
            "hint_or_unknown_hint_interactions": int((~no_hint).sum()),
            "invalid_label_interactions": int((~binary).sum()),
            "retained_before_user_filter": int(keep.sum()),
            "repeated_student_problem_events": int(frame.duplicated(["user_raw", "item_raw"]).sum()),
        },
    })
    return _fixed_student_selection(frame[COMMON_COLUMNS + ["order_raw"]], sample_users, split, seed)


def _leaf_subjects(value: str, parents: dict[str, str]) -> str:
    tags = [str(tag) for tag in ast.literal_eval(value)]
    ancestors: set[str] = set()
    for tag in tags:
        visited: set[str] = set()
        current = parents.get(tag)
        while current is not None and current not in visited:
            ancestors.add(current)
            visited.add(current)
            current = parents.get(current)
    return ";".join(dict.fromkeys(tag for tag in tags if tag not in ancestors))


def _prepare_nips34(
    data_root: Path, sample_users: int | None, seed: int, split: dict[str, float], tie_order: str = "answer_id"
) -> pd.DataFrame:
    root = data_root / "NIPS2020_Task34/raw/data"
    raw = pd.read_csv(root / "train_data/train_task_3_4.csv", dtype=str)
    dates = pd.read_csv(root / "metadata/answer_metadata_task_3_4.csv", usecols=["AnswerId", "DateAnswered"], dtype=str)
    questions = pd.read_csv(root / "metadata/question_metadata_task_3_4.csv", dtype=str)
    subjects = pd.read_csv(root / "metadata/subject_metadata.csv", dtype=str, encoding="utf-8-sig")
    parents = {
        row.SubjectId.removesuffix(".0"): row.ParentId.removesuffix(".0") for row in subjects.itertuples()
        if pd.notna(row.ParentId) and row.ParentId not in {"NULL", ""}
    }
    questions["concepts_raw"] = questions["SubjectId"].map(lambda value: _leaf_subjects(value, parents))
    frame = raw.merge(dates, on="AnswerId", how="left", validate="many_to_one")
    frame = frame.merge(questions[["QuestionId", "concepts_raw"]], on="QuestionId", how="left", validate="many_to_one")
    parsed_dates = pd.to_datetime(frame["DateAnswered"], errors="coerce")
    keep = frame["IsCorrect"].isin(["0", "1"]) & parsed_dates.notna() & frame["concepts_raw"].notna() & frame["concepts_raw"].ne("")
    frame = frame.loc[keep].rename(columns={"UserId": "user_raw", "QuestionId": "item_raw", "IsCorrect": "correct"}).copy()
    frame["correct"] = frame["correct"].astype(np.int8)
    frame["timestamp"] = parsed_dates.loc[keep].astype("int64")
    frame["answer_order"] = pd.to_numeric(frame["AnswerId"], errors="raise").astype(np.int64)
    ties = frame.duplicated(["user_raw", "timestamp"], keep=False)
    if tie_order == "random":
        frame["tie_order"] = np.random.default_rng(seed).random(len(frame))
        sort_columns = ["user_raw", "timestamp", "tie_order", "answer_order"]
    elif tie_order == "answer_id":
        sort_columns = ["user_raw", "timestamp", "answer_order"]
    else:
        raise ValueError("nips34 tie_order must be 'answer_id' or 'random'")
    frame = frame.sort_values(sort_columns, kind="mergesort")
    frame["order_raw"] = frame.groupby("user_raw", sort=False).cumcount()
    frame.attrs.update({
        "protocol": {
            "source": "only official train_data/train_task_3_4.csv; official test files excluded",
            "split": "eligible students randomly assigned by seed to configured 70/15/15 before sampling",
            "concepts": "SubjectId ancestors removed; all deepest available subject leaves retained",
            "ordering": f"DateAnswered then {'seeded random tie order' if tie_order == 'random' else 'numeric AnswerId'}; minute-resolution tied events have a surrogate order, not a verified true order",
            "tie_order": tie_order,
            "ordering_limitation": "same-minute tie-order sensitivity requires separate analysis before chronology-sensitive scientific claims",
            "time": "datetime64 nanoseconds; within-minute gaps are zero",
            "sampling": "student budget allocated proportionally after fixed student split; null means full eligible data",
            "graph": "question-concept metadata uses all selected splits; labels used only for training item difficulty",
        },
        "filter_statistics": {
            "source_interactions": int(len(raw)),
            "missing_date_concept_or_nonbinary_label": int((~keep).sum()),
            "retained_before_user_filter": int(keep.sum()),
            "same_student_minute_tied_events": int(ties.sum()),
            "same_student_minute_tied_event_rate": float(ties.mean()),
        },
    })
    return _fixed_student_selection(frame[COMMON_COLUMNS + ["order_raw"]], sample_users, split, seed)


def _finalize(
    frame: pd.DataFrame, output_dir: Path, split: dict[str, float], seed: int,
    max_length: int = 100, min_length: int = 3,
) -> dict:
    attributes = frame.attrs.copy()
    frame = frame.dropna(subset=["user_raw", "item_raw", "concepts_raw"]).copy()
    frame["user_raw"] = frame["user_raw"].astype(str)
    frame["item_raw"] = frame["item_raw"].astype(str)
    frame["concepts_raw"] = frame["concepts_raw"].astype(str)
    if frame.empty:
        raise ValueError("Dataset preprocessing produced no usable interactions")
    if "sequence_raw" not in frame:
        frame["sequence_raw"] = frame["user_raw"]
    frame["sequence_raw"] = frame["sequence_raw"].astype(str)
    if "order_raw" in frame:
        frame = frame.sort_values(["user_raw", "sequence_raw", "order_raw"], kind="mergesort")
    else:
        frame = frame.sort_values(["user_raw", "sequence_raw", "timestamp", "item_raw"], kind="mergesort")
        frame["order_raw"] = frame.groupby(["user_raw", "sequence_raw"], sort=False).cumcount()
    frame = frame.reset_index(drop=True)

    users = sorted(frame["user_raw"].unique().tolist())
    items = sorted(frame["item_raw"].unique().tolist())
    concepts = sorted({tag for value in frame["concepts_raw"] for tag in value.split(";") if tag})
    user_map = {value: idx for idx, value in enumerate(users)}
    item_map = {value: idx for idx, value in enumerate(items)}
    concept_map = {value: idx for idx, value in enumerate(concepts)}
    frame["user_idx"] = frame["user_raw"].map(user_map).astype("int32")
    frame["item_idx"] = frame["item_raw"].map(item_map).astype("int32")
    frame["concept_idx"] = frame["concepts_raw"].map(lambda value: concept_map[value.split(";")[0]]).astype("int32")
    sequence_pairs = pd.MultiIndex.from_frame(frame[["user_raw", "sequence_raw"]])
    frame["sequence_idx"] = pd.factorize(sequence_pairs, sort=False)[0].astype("int32")
    frame["event_order"] = frame["order_raw"].astype("int64")
    frame["event_idx"] = np.arange(len(frame), dtype=np.int64)

    edge_pairs = set()
    for item_raw, tags_raw in frame[["item_raw", "concepts_raw"]].drop_duplicates().itertuples(index=False):
        for tag in tags_raw.split(";"):
            if tag:
                edge_pairs.add((item_map[item_raw], concept_map[tag]))
    edges = sorted(edge_pairs)

    if "split_raw" in frame:
        if not frame["split_raw"].isin(SPLIT_NAMES).all():
            raise ValueError("Unknown or missing fixed split label")
        if frame.groupby("user_raw")["split_raw"].nunique().max() != 1:
            raise ValueError("A student cannot belong to multiple data splits")
        assignments = frame.drop_duplicates("user_raw").set_index("user_raw")["split_raw"].to_dict()
    else:
        assignments = _random_user_splits(users, split, seed)
        frame["split_raw"] = frame["user_raw"].map(assignments)
    splits = {
        name: np.asarray(sorted(user_map[user] for user, value in assignments.items() if value == name), dtype=np.int32)
        for name in SPLIT_NAMES
    }

    # Empirical item difficulty must be estimated from training users only.
    # Items unseen in training retain the neutral prior 0.5.
    train_frame = frame[frame["user_idx"].isin(set(splits["train"].tolist()))]
    item_accuracy = train_frame.groupby("item_idx", sort=False)["correct"].mean()
    difficulty = np.full(len(items), 0.5, dtype=np.float32)
    for item_idx, accuracy in item_accuracy.items():
        difficulty[int(item_idx)] = float(np.clip(1.0 - accuracy, 0.0, 1.0))

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["user_idx", "item_idx", "concept_idx", "correct", "timestamp", "sequence_idx", "event_order", "event_idx"]
    columns += [name for name in ("time_valid", "target_valid") if name in frame]
    frame[columns].to_parquet(output_dir / "interactions.parquet", index=False)
    np.savez_compressed(output_dir / "splits.npz", **splits)
    np.save(output_dir / "difficulty.npy", difficulty)
    np.savez_compressed(
        output_dir / "hypergraph.npz",
        item=np.asarray([pair[0] for pair in edges], dtype=np.int64),
        concept=np.asarray([pair[1] for pair in edges], dtype=np.int64),
    )
    metadata = {
        "schema_version": 2,
        "num_interactions": int(len(frame)),
        "num_users": len(users),
        "num_items": len(items),
        "num_concepts": len(concepts),
        "num_sequences": int(frame["sequence_idx"].nunique()),
        "num_hypergraph_incidents": len(edges),
        "correct_rate": float(frame["correct"].mean()),
        "user_ids": users,
        "item_ids": items,
        "concept_ids": concepts,
        "seed": seed,
        "split_user_counts": {name: int(len(value)) for name, value in splits.items()},
        "protocol": attributes.get("protocol", {"split": "random student split", "ordering": "timestamp then item"}),
        "filter_statistics": attributes.get("filter_statistics", {}),
        "full_split_user_counts": attributes.get("full_split_user_counts", {name: int(len(value)) for name, value in splits.items()}),
        "target_protocol": {"max_length": int(max_length), "min_length": int(min_length), "event_id": "prepared row ordinal, targets use the next interaction's event_idx"},
    }
    split_statistics = {}
    for name in SPLIT_NAMES:
        part = frame[frame["split_raw"].eq(name)]
        target_labels: list[np.ndarray] = []
        for _, group in part.groupby("sequence_idx", sort=False):
            if len(group) < min_length:
                continue
            labels = group["correct"].to_numpy()
            enabled = group["target_valid"].to_numpy(bool) if "target_valid" in group else np.ones(len(group), dtype=bool)
            for start in range(0, len(group) - 1, max_length):
                end = min(start + max_length + 1, len(group))
                if end - start >= min_length:
                    target_labels.append(labels[start + 1 : end][enabled[start + 1 : end]])
        targets = np.concatenate(target_labels) if target_labels else np.asarray([], dtype=np.int8)
        positives = int(targets.sum())
        split_statistics[name] = {
            "num_users": int(len(splits[name])), "num_sequences": int(part["sequence_idx"].nunique()),
            "num_interactions": int(len(part)), "correct_interactions": int(part["correct"].sum()),
            "incorrect_interactions": int(len(part) - part["correct"].sum()),
            "num_targets": int(len(targets)), "correct_targets": positives,
            "incorrect_targets": int(len(targets) - positives),
            "target_correct_rate": float(targets.mean()) if len(targets) else None,
        }
    metadata["split_statistics"] = split_statistics
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def prepare_dataset(name: str, config: dict, force: bool = False) -> dict:
    data_root = Path(config["data_root"])
    prepared_root = Path(config.get("prepared_root", Path(config["artifact_root"]) / "prepared"))
    output_dir = prepared_root / name
    metadata_path = output_dir / "metadata.json"
    sequence = config.get("sequence", {})
    fingerprint = {
        "schema_version": 2,
        "dataset": name,
        "data_root": str(data_root.resolve()),
        "seed": int(config["seed"]),
        "sample_users": config["datasets"][name].get("sample_users"),
        "tie_order": config["datasets"][name].get("tie_order", "answer_id") if name == "nips34" else None,
        "max_length": int(sequence.get("max_length", 100)),
        "min_length": int(sequence.get("min_length", 3)),
        "split": config["split"],
    }
    if metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("preparation_config") == fingerprint:
            return metadata
    seed = int(config["seed"])
    sample_users = config["datasets"][name].get("sample_users")
    if name == "assist12":
        frame = _prepare_assist(data_root, sample_users, seed)
    elif name == "ednet":
        frame = _prepare_ednet(data_root, sample_users, seed)
    elif name == "oulad":
        frame = _prepare_oulad(data_root, sample_users, seed)
    elif name == "xes3g5m":
        frame = _prepare_xes3g5m(data_root, sample_users, seed)
    elif name == "assist17":
        frame = _prepare_assist17(data_root, sample_users, seed, config["split"])
    elif name == "nips34":
        frame = _prepare_nips34(data_root, sample_users, seed, config["split"], config["datasets"][name].get("tie_order", "answer_id"))
    else:
        raise ValueError(f"Unknown dataset: {name}")
    metadata = _finalize(frame, output_dir, config["split"], seed, int(sequence.get("max_length", 100)), int(sequence.get("min_length", 3)))
    metadata["preparation_config"] = fingerprint
    write_json(metadata_path, metadata)
    return metadata
