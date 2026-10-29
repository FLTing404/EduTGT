#!/usr/bin/env python3
"""Read-only statistics over the downloaded original Eedi Task 3/4 CSVs."""
import ast
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
train = pd.read_csv(ROOT / "raw/data/train_data/train_task_3_4.csv")
answer = pd.read_csv(ROOT / "raw/data/metadata/answer_metadata_task_3_4.csv")
question = pd.read_csv(ROOT / "raw/data/metadata/question_metadata_task_3_4.csv")
student = pd.read_csv(ROOT / "raw/data/metadata/student_metadata_task_3_4.csv")
subject = pd.read_csv(ROOT / "raw/data/metadata/subject_metadata.csv", encoding="utf-8-sig")

correct_counts = train["IsCorrect"].value_counts().sort_index()
user_counts = train.groupby("UserId").size()
dates = train[["AnswerId", "UserId"]].merge(answer[["AnswerId", "DateAnswered"]], on="AnswerId", how="left", validate="one_to_one")
dates["DateAnswered"] = pd.to_datetime(dates["DateAnswered"], errors="coerce")
valid_dates = dates["DateAnswered"].dropna()
all_dates = pd.to_datetime(answer["DateAnswered"], errors="coerce")
tag_lists = question["SubjectId"].map(ast.literal_eval)
parent_by_id = {int(row.SubjectId): None if pd.isna(row.ParentId) else int(row.ParentId) for row in subject.itertuples(index=False)}


def ancestors(cid):
    result = set()
    while parent_by_id.get(cid) is not None:
        cid = parent_by_id[cid]
        if cid in result:
            raise ValueError("Cycle in subject hierarchy")
        result.add(cid)
    return result


def leaf_tags(tags):
    tagged_ancestors = set().union(*(ancestors(cid) for cid in tags))
    return [cid for cid in tags if cid not in tagged_ancestors]


leaf_lists = tag_lists.map(leaf_tags)
tag_counts = tag_lists.map(len)
leaf_counts = leaf_lists.map(len)
known_tags = set(parent_by_id)
referenced_tags = set().union(*(set(tags) for tags in tag_lists))
leaf_referenced = set().union(*(set(tags) for tags in leaf_lists))
qids = set(train["QuestionId"].unique())
uids = set(train["UserId"].unique())
training = {
    "rows": len(train),
    "users": int(train["UserId"].nunique()),
    "questions": int(train["QuestionId"].nunique()),
    "unique_answer_ids": int(train["AnswerId"].nunique()),
    "duplicate_user_question_rows": int(train.duplicated(["UserId", "QuestionId"]).sum()),
    "correct": int(correct_counts.get(1, 0)),
    "incorrect": int(correct_counts.get(0, 0)),
    "correct_rate": float((train["IsCorrect"] == 1).mean()),
    "incorrect_rate": float((train["IsCorrect"] == 0).mean()),
    "label_values": [int(x) for x in correct_counts.index],
    "correctness_vs_answer_choice_mismatches": int(((train["AnswerValue"] == train["CorrectAnswer"]).astype(int) != train["IsCorrect"]).sum()),
    "answers_per_user_min": int(user_counts.min()),
    "answers_per_user_median": float(user_counts.median()),
    "answers_per_user_mean": float(user_counts.mean()),
    "answers_per_user_max": int(user_counts.max()),
    "date_answered_min": valid_dates.min().isoformat() if len(valid_dates) else None,
    "date_answered_max": valid_dates.max().isoformat() if len(valid_dates) else None,
    "date_answered_missing_rows": int(dates["DateAnswered"].isna().sum()),
    "same_user_same_minute_extra_rows": int(dates.dropna(subset=["DateAnswered"]).duplicated(["UserId", "DateAnswered"]).sum()),
    "missing_question_metadata_ids": len(qids - set(question["QuestionId"])),
    "missing_student_metadata_ids": len(uids - set(student["UserId"])),
}
metadata_stats = {
    "answer_metadata_rows": len(answer),
    "question_metadata_rows": len(question),
    "student_metadata_rows": len(student),
    "subject_metadata_rows": len(subject),
    "answer_metadata_date_min": all_dates.min().isoformat(),
    "answer_metadata_date_max": all_dates.max().isoformat(),
    "answer_metadata_missing_date_rows": int(all_dates.isna().sum()),
    "raw_subject_tags_unique": len(referenced_tags),
    "raw_subject_tag_count_min": int(tag_counts.min()),
    "raw_subject_tag_count_max": int(tag_counts.max()),
    "raw_subject_tag_count_histogram": {str(k): int(v) for k, v in tag_counts.value_counts().sort_index().items()},
    "questions_with_multiple_raw_subject_tags": int((tag_counts > 1).sum()),
    "leaf_subject_tags_unique": len(leaf_referenced),
    "leaf_subject_tag_count_histogram": {str(k): int(v) for k, v in leaf_counts.value_counts().sort_index().items()},
    "questions_with_multiple_leaf_subject_tags": int((leaf_counts > 1).sum()),
    "referenced_subject_ids_missing_from_tree": sorted(referenced_tags - known_tags),
    "subject_levels": {str(k): int(v) for k, v in subject["Level"].value_counts().sort_index().items()},
    "subject_tag_semantics": "SubjectId is a list containing mathematics topics at several hierarchy levels. Raw multiple tags include ancestors; leaf counts remove any tag that is an ancestor of another tag on the same question.",
}
files = []
for path in sorted((ROOT / "raw/data").rglob("*.csv")):
    table = pd.read_csv(path, encoding="utf-8-sig")
    item = {"member": path.relative_to(ROOT).as_posix(), "rows": len(table), "columns": list(table.columns), "bytes": path.stat().st_size}
    if "IsCorrect" in table:
        item["is_correct_value_counts"] = {str(k): int(v) for k, v in table["IsCorrect"].value_counts(dropna=False).sort_index().items()}
    files.append(item)
stats = {
    "dataset": "NIPS2020 Task 3/4 original CSV subset",
    "analysis_method": "Full read-only CSV scan; training counts use train_task_3_4.csv; training dates join answer metadata via AnswerId; no processed datasets are saved",
    "training": training,
    "metadata": metadata_stats,
    "files": files,
    "cautions": [
        "Official competition Task 3/4 is question-quality assessment and active question selection, rather than a canonical next-event KT benchmark.",
        "The release keeps only the latest answer per (UserId, QuestionId); it omits earlier repeated attempts.",
        "DateAnswered is rounded to minute precision; tied timestamps cannot reconstruct a unique within-minute sequence.",
        "Task 1/2 and Task 3/4 IDs are anonymized separately and must not be joined.",
        "SubjectId includes a hierarchy route; multiple raw topic IDs should not automatically be treated as independent leaf knowledge components.",
        "Competition public/private test files preserve their original labels and masks. Use an explicit student/time split for a new KT experiment.",
        "The current project training pipeline has not been changed to consume this dataset.",
    ],
}
(ROOT / "metadata/stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"training": training, "metadata": metadata_stats}, ensure_ascii=False, indent=2))
