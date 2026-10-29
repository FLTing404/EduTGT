"""is_pass 映射与标签表逻辑。"""
from __future__ import annotations

from typing import Iterable, Set

import pandas as pd

PASS_SET = {"Pass", "Distinction"}
FAIL_SET = {"Fail", "Withdrawn"}


def final_result_to_is_pass(final_result: str) -> int:
    s = str(final_result).strip()
    if s in PASS_SET:
        return 1
    if s in FAIL_SET:
        return 0
    raise ValueError(f"未约定的 final_result: {final_result!r}（请补充 PASS_SET/FAIL_SET）")


def load_labels_from_student_info(
    student_info_path,
    code_module: str,
    presentations: Iterable[str],
    student_ids_in_graph: Set[int],
    student_to_nid: dict,
) -> pd.DataFrame:
    df = pd.read_csv(student_info_path)
    pres_set = set(presentations)
    m = (df["code_module"].astype(str) == str(code_module)) & (
        df["code_presentation"].astype(str).isin(pres_set)
    )
    sub = df.loc[m, ["id_student", "code_module", "code_presentation", "final_result"]].copy()
    sub["id_student"] = sub["id_student"].astype(int)
    sub = sub[sub["id_student"].isin(student_ids_in_graph)]
    sub["ml_node_id"] = sub["id_student"].map(lambda s: student_to_nid[int(s)])
    sub["is_pass"] = sub["final_result"].map(final_result_to_is_pass)
    return sub[["id_student", "ml_node_id", "code_module", "code_presentation", "is_pass"]]
