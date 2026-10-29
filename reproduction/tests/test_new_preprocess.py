import io
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fchgnn_repro.data import SequenceDataset, _normalized_time_gaps
from fchgnn_repro.preprocess import (
    COMMON_COLUMNS,
    _finalize,
    _leaf_subjects,
    _prepare_nips34,
    _prepare_assist17,
    _prepare_xes3g5m,
    _random_user_splits,
    _sample_fixed_splits,
    prepare_dataset,
)

SPLIT = {"train": 0.7, "valid": 0.15, "test": 0.15}


def _xes_archive(root: Path, train: pd.DataFrame, test: pd.DataFrame):
    path = root / "XES3G5M/raw/XES3G5M.tar.gz"
    path.parent.mkdir(parents=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, frame in (("train_valid_sequences_quelevel.csv", train), ("test_quelevel.csv", test)):
            content = frame.to_csv(index=False).encode()
            member = tarfile.TarInfo(f"XES3G5M/question_level/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def test_official_xes_split_segment_order_multilabel_and_shifted_mask(tmp_path):
    train = pd.DataFrame([
        {"fold": "0", "uid": "valid", "questions": "a,b,c", "concepts": "1_2,2,3", "responses": "1,0,1", "timestamps": "1600000000000,990000000000,1600000002000", "selectmasks": "1,0,1"},
        {"fold": "1", "uid": "train", "questions": "d,e,f", "concepts": "1,2,3", "responses": "1,1,0", "timestamps": "1600000000000,1600000001000,1600000002000", "selectmasks": "1,1,1"},
        {"fold": "1", "uid": "train", "questions": "g,h,i", "concepts": "1,2,3", "responses": "0,0,1", "timestamps": "1600000003000,1600000004000,1600000005000", "selectmasks": "1,1,1"},
    ])
    test = pd.DataFrame([
        {"fold": "-1", "uid": "test", "questions": "j,k,l", "concepts": "1,2,3", "responses": "0,1,0", "timestamps": "1600000000000,1600000001000,1600000002000"},
    ])
    _xes_archive(tmp_path, train, test)
    frame = _prepare_xes3g5m(tmp_path, None, seed=5)
    prepared = tmp_path / "prepared/xes3g5m"
    meta = _finalize(frame, prepared, SPLIT, 5, max_length=10)
    splits = np.load(prepared / "splits.npz")
    assert {meta["user_ids"][index] for index in splits["test"]} == {"test"}
    assert {meta["user_ids"][index] for index in splits["valid"]} == {"valid"}
    train_data = SequenceDataset(prepared, "train", max_length=10)
    assert len(train_data) == 2
    assert sum(int(sample["mask"].sum()) for sample in train_data.samples) == 4
    # No prediction from the last event of segment 1 to the first event of segment 2.
    target_items = {int(item) for sample in train_data.samples for item in sample["next_item"][sample["mask"]]}
    assert meta["item_ids"].index("g") not in target_items
    valid_data = SequenceDataset(prepared, "valid", max_length=10)
    assert len(valid_data) == 1
    sample = valid_data.samples[0]
    assert sample["mask"][:2].tolist() == [False, True]
    assert sample["next_item"][:2].tolist() == [meta["item_ids"].index("b"), meta["item_ids"].index("c")]
    assert sample["next_time_gap"][:2].tolist() == [0.0, 0.0]
    assert np.all(sample["user_idx"][:2] == meta["user_ids"].index("valid"))
    assert len(set(sample["event_idx"][:2])) == 2
    assert sample["history_position"][:2].tolist() == [1, 2]
    assert sample["history_position"][2:].tolist() == [-1] * 8
    graph = np.load(prepared / "hypergraph.npz")
    a_index = meta["item_ids"].index("a")
    assert set(graph["concept"][graph["item"] == a_index]) == {meta["concept_ids"].index("1"), meta["concept_ids"].index("2")}
    # An unseen validation item retains a neutral difficulty despite its label.
    assert np.load(prepared / "difficulty.npy")[a_index] == 0.5
    assert meta["split_statistics"]["valid"]["num_targets"] == 1


def test_reject_student_in_multiple_fixed_splits(tmp_path):
    frame = pd.DataFrame([
        ("student", "a", "1", 1, 1, "train"),
        ("student", "b", "1", 0, 2, "test"),
    ], columns=COMMON_COLUMNS + ["split_raw"])
    with pytest.raises(ValueError, match="multiple data splits"):
        _finalize(frame, tmp_path, SPLIT, 1)


def test_sampling_happens_after_fixed_student_split():
    assignments = _random_user_splits([str(i) for i in range(100)], SPLIT, 7)
    small = _sample_fixed_splits(assignments, 20, 7)
    large = _sample_fixed_splits(assignments, 40, 7)
    assert len(small) == 20 and len(large) == 40
    for user in small | large:
        assert assignments[user] in {"train", "valid", "test"}
    assert len({user for user in small if assignments[user] == "train"}) == 14
    assert len({user for user in small if assignments[user] == "valid"}) == 3
    assert len({user for user in small if assignments[user] == "test"}) == 3


def test_nips_leaf_concepts_same_minute_order_and_no_official_test(tmp_path):
    root = tmp_path / "NIPS2020_Task34/raw/data"
    (root / "train_data").mkdir(parents=True)
    (root / "metadata").mkdir()
    (root / "test_data").mkdir()
    rows = []
    for user in range(20):
        for answer in (3, 1, 2):
            rows.append((answer, str(user), user * 3 + answer, answer % 2))
    pd.DataFrame(rows, columns=["QuestionId", "UserId", "AnswerId", "IsCorrect"]).to_csv(root / "train_data/train_task_3_4.csv", index=False)
    pd.DataFrame({"AnswerId": range(1, 61), "DateAnswered": ["2020-01-01 12:00:00.000"] * 60}).to_csv(root / "metadata/answer_metadata_task_3_4.csv", index=False)
    pd.DataFrame({"QuestionId": [1, 2, 3], "SubjectId": ["[3, 10, 11, 12]", "[3, 10, 11]", "[3, 10, 12]"]}).to_csv(root / "metadata/question_metadata_task_3_4.csv", index=False)
    pd.DataFrame({"SubjectId": [3, 10, 11, 12], "ParentId": [None, 3, 10, 10]}).to_csv(root / "metadata/subject_metadata.csv", index=False)
    # This deliberately malformed official test file must never be read.
    (root / "test_data/test_private_task_4.csv").write_text("forbidden official test data")
    frame = _prepare_nips34(tmp_path, None, 3, SPLIT)
    assert frame.loc[frame["user_raw"].eq("0"), "item_raw"].tolist() == ["1", "2", "3"]
    assert frame.loc[frame["item_raw"].eq("1"), "concepts_raw"].iloc[0] == "11;12"
    assert frame.attrs["filter_statistics"]["same_student_minute_tied_event_rate"] == 1.0
    assert frame.groupby("user_raw")["split_raw"].nunique().eq(1).all()
    randomized = _prepare_nips34(tmp_path, None, 3, SPLIT, tie_order="random")
    assert randomized.attrs["protocol"]["tie_order"] == "random"
    assert randomized[["user_raw", "split_raw"]].drop_duplicates().sort_values("user_raw").to_numpy().tolist() == frame[["user_raw", "split_raw"]].drop_duplicates().sort_values("user_raw").to_numpy().tolist()
    assert randomized["item_raw"].tolist() != frame["item_raw"].tolist()
    assert _leaf_subjects("[3, 10, 11, 12]", {"10": "3", "11": "10", "12": "10"}) == "11;12"


def test_time_units_and_masked_xes_anomalies():
    seconds = _normalized_time_gaps(np.asarray([1000, 1010]), "assist17")
    milliseconds = _normalized_time_gaps(np.asarray([1000000, 1010000]), "xes3g5m")
    assert seconds[1] == pytest.approx(milliseconds[1])
    assert _normalized_time_gaps(np.asarray([1600000000000, 990000000000, 1600000002000]), "xes3g5m", np.asarray([True, False, True])).tolist() == [0.0, 0.0, 0.0]


def test_prepared_cache_checks_protocol_config(tmp_path, monkeypatch):
    calls = []

    def source(data_root, sample_users, seed, split, tie_order):
        calls.append((sample_users, seed, tie_order))
        rows = []
        for user, partition in (("a", "train"), ("b", "valid"), ("c", "test")):
            rows.extend((user, str(item), "concept", item % 2, item, partition) for item in range(4))
        return pd.DataFrame(rows, columns=COMMON_COLUMNS + ["split_raw"])

    monkeypatch.setattr("fchgnn_repro.preprocess._prepare_nips34", source)
    config = {"data_root": str(tmp_path), "artifact_root": str(tmp_path / "artifacts"), "seed": 10,
              "split": SPLIT, "sequence": {"max_length": 100, "min_length": 3},
              "datasets": {"nips34": {"sample_users": 3}}}
    prepare_dataset("nips34", config)
    prepare_dataset("nips34", config)
    assert len(calls) == 1
    for mutation in (lambda: config["datasets"]["nips34"].update(tie_order="random"),
                     lambda: config["sequence"].update(max_length=2),
                     lambda: config.update(seed=11),
                     lambda: config["datasets"]["nips34"].update(sample_users=None)):
        mutation()
        prepare_dataset("nips34", config)
    assert len(calls) == 5


def test_legacy_parquet_produces_target_identity(tmp_path):
    prepared = tmp_path / "assist12"
    prepared.mkdir()
    pd.DataFrame({"user_idx": [0, 0, 0], "item_idx": [0, 1, 2], "concept_idx": [0, 0, 0],
                  "correct": [1, 0, 1], "timestamp": [10, 20, 30]}).to_parquet(prepared / "interactions.parquet", index=False)
    np.savez(prepared / "splits.npz", train=np.asarray([0]))
    dataset = SequenceDataset(prepared, "train", 5)
    assert dataset.samples[0]["event_idx"][:2].tolist() == [1, 2]
    assert dataset.samples[0]["user_idx"][:2].tolist() == [0, 0]
    assert dataset.samples[0]["history_position"][:2].tolist() == [1, 2]


def test_assist17_filters_help_scaffolds_and_uses_numeric_action_order(tmp_path):
    rows = []
    for user in range(20):
        for action, item in ((10, "second"), (2, "first"), (20, "last")):
            rows.append((str(user), item, "skill", "1", "1", "0", "1000", action))
    rows.extend([
        ("0", "hint", "skill", "0", "1", "1", "1000", 1),
        ("0", "scaffold", "skill", "0", "0", "0", "1000", 1),
        ("0", "unknown", "noskill", "0", "1", "0", "1000", 1),
    ])
    raw = pd.DataFrame(rows, columns=["studentId", "problemId", "skill", "correct", "original", "hint", "startTime", "action_num"])
    archive = tmp_path / "ASSISTments2017/raw/anonymized_full_release_competition_dataset.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as release:
        release.writestr("anonymized_full_release_competition_dataset.csv", raw.to_csv(index=False))
    frame = _prepare_assist17(tmp_path, None, 11, SPLIT)
    assert frame.loc[frame["user_raw"].eq("0"), "item_raw"].tolist() == ["first", "second", "last"]
    assert frame.attrs["filter_statistics"]["retained_before_user_filter"] == 60
    assert len(frame) == 60
    assert frame.groupby("user_raw")["split_raw"].nunique().eq(1).all()
