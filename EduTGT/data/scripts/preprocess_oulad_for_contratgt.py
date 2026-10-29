#!/usr/bin/env python3
"""
OULAD -> ContraTGT/EduTGT 预处理：学生 ↔ VLE 站点 (presentation, id_site) 二部图。

- --module AAA -> data/processed/data_AAA/（多 presentation）
- --module AAA-2013J -> data/processed/data_AAA_2013J/（单实例）
- --module ALL -> 依次 AAA…GGG
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 允许的课程实例 (code_module, code_presentation)
# ---------------------------------------------------------------------------
ALLOWED_COURSES: List[Tuple[str, str]] = [
    ("AAA", "2013J"),
    ("AAA", "2014J"),
    ("BBB", "2013J"),
    ("BBB", "2014J"),
    ("BBB", "2013B"),
    ("BBB", "2014B"),
    ("CCC", "2014J"),
    ("CCC", "2014B"),
    ("DDD", "2013J"),
    ("DDD", "2014J"),
    ("DDD", "2013B"),
    ("DDD", "2014B"),
    ("EEE", "2013J"),
    ("EEE", "2014J"),
    ("EEE", "2014B"),
    ("FFF", "2013J"),
    ("FFF", "2014J"),
    ("FFF", "2013B"),
    ("FFF", "2014B"),
    ("GGG", "2013J"),
    ("GGG", "2014J"),
    ("GGG", "2014B"),
]

REQUIRED_FILES = [
    "courses.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv",
]

STUDENT_INFO_COLS = [
    "code_module",
    "code_presentation",
    "id_student",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
]
STUDENT_REG_COLS = [
    "code_module",
    "code_presentation",
    "id_student",
    "date_registration",
    "date_unregistration",
]
STUDENT_VLE_COLS = [
    "code_module",
    "code_presentation",
    "id_student",
    "id_site",
    "date",
    "sum_click",
]
VLE_COLS = [
    "id_site",
    "code_module",
    "code_presentation",
    "activity_type",
    "week_from",
    "week_to",
]
COURSES_COLS = ["code_module", "code_presentation", "module_presentation_length"]

# 合并多 presentation 时，ts = pres_block_index * TS_PRES_BLOCK + date（date 为各课相对日，不可直接混排）
TS_PRES_BLOCK = 1000


def _require_cols(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少列: {missing}，实际列: {list(df.columns)}")


def _sanitize_token(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def parse_module_arg(s: str) -> Tuple[str, str, str]:
    """
    返回 (kind, module, presentation)：
    - kind=='all'：依次处理 ALLOWED_COURSES 中每个不同 module（AAA→…→GGG），presentation 为 ''
    - kind=='module'：整模块，presentation 为 ''
    - kind=='pres'：单实例 MODULE-PRESENTATION，presentation 为 code_presentation（如 2013J）
    """
    s = s.strip()
    if not s:
        raise ValueError("--module 不能为空")
    u = s.upper().replace(" ", "")
    if u == "ALL":
        return ("all", "", "")
    m = re.fullmatch(r"([A-Z]{3})-(\w+)", u)
    if m:
        mod, pres = m.group(1), m.group(2)
        if (mod, pres) not in ALLOWED_COURSES:
            raise ValueError(
                f"({mod}, {pres}) 不在 ALLOWED_COURSES 中；请核对模块与学期代码"
            )
        return ("pres", mod, pres)
    if re.fullmatch(r"[A-Z]{3}", u):
        mod = u
        have = [cp for cm, cp in ALLOWED_COURSES if cm == mod]
        if not have:
            raise ValueError(f"模块 {mod} 在 ALLOWED_COURSES 中无 presentation")
        return ("module", mod, "")
    raise ValueError(
        f"无法解析 --module {s!r}；使用 AAA、AAA-2013J 或 ALL"
    )


def modules_in_allowed_order() -> List[str]:
    """ALLOWED_COURSES 中首次出现的 code_module 顺序（AAA, BBB, …）。"""
    out: List[str] = []
    for cm, _ in ALLOWED_COURSES:
        if cm not in out:
            out.append(cm)
    return out


def allowed_presentations_for_module(code_module: str) -> List[Tuple[str, str]]:
    return [(cm, cp) for cm, cp in ALLOWED_COURSES if cm == code_module.upper()]


def load_raw_tables(raw_dir: str) -> Dict[str, pd.DataFrame]:
    paths = {f: os.path.join(raw_dir, f) for f in REQUIRED_FILES}
    for p in paths.values():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺少文件: {p}")
    courses = pd.read_csv(paths["courses.csv"], dtype=str)
    student_info = pd.read_csv(paths["studentInfo.csv"], dtype=str)
    student_reg = pd.read_csv(paths["studentRegistration.csv"], dtype=str)
    student_vle = pd.read_csv(paths["studentVle.csv"], dtype=str)
    vle = pd.read_csv(paths["vle.csv"], dtype=str)
    _require_cols(courses, COURSES_COLS, "courses.csv")
    _require_cols(student_info, STUDENT_INFO_COLS, "studentInfo.csv")
    _require_cols(student_reg, STUDENT_REG_COLS, "studentRegistration.csv")
    _require_cols(student_vle, STUDENT_VLE_COLS, "studentVle.csv")
    _require_cols(vle, VLE_COLS, "vle.csv")
    return {
        "courses": courses,
        "student_info": student_info,
        "student_reg": student_reg,
        "student_vle": student_vle,
        "vle": vle,
    }


def one_hot_matrix(
    series: pd.Series, categories: List[str], name: str
) -> Tuple[np.ndarray, List[str]]:
    s = series.fillna("__MISSING__").astype(str)
    cat_to_i = {c: i for i, c in enumerate(categories)}
    unk = len(categories)
    n = len(s)
    k = len(categories) + 1
    out = np.zeros((n, k), dtype=np.float64)
    col_names = [f"{name}_{_sanitize_token(c)}" for c in categories] + [
        f"{name}__UNK__"
    ]
    for i, v in enumerate(s):
        j = cat_to_i.get(v, unk)
        out[i, j] = 1.0
    return out, col_names


def fit_categories(values: pd.Series) -> List[str]:
    s = values.fillna("__MISSING__").astype(str)
    return sorted(s.unique().tolist())


def standardize_block(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """列 z-score；全常数列 -> 置 0。"""
    mu = x.mean(axis=0)
    sig = x.std(axis=0)
    sig[sig == 0] = 1.0
    z = (x - mu) / sig
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z, mu, sig


def _parse_week_val(v) -> float:
    s = str(v).strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _aggregate_vlesite_edges_one_pres(
    tables: Dict[str, pd.DataFrame], code_m: str, code_p: str
) -> pd.DataFrame:
    """
    学生–站点日边：键 (id_student, code_module, code_presentation, id_site, date)，
    total_clicks = sum(sum_click)；无 vle 匹配的行丢弃并告警。
    """
    sv = tables["student_vle"]
    sv_f = sv[
        (sv["code_module"] == code_m) & (sv["code_presentation"] == code_p)
    ].copy()
    if sv_f.empty:
        return pd.DataFrame()

    vle = tables["vle"]
    vle_f = vle[
        (vle["code_module"] == code_m) & (vle["code_presentation"] == code_p)
    ].copy()
    if vle_f.empty:
        return pd.DataFrame()

    sv_f["id_student"] = pd.to_numeric(sv_f["id_student"], errors="coerce")
    sv_f["id_site"] = sv_f["id_site"].astype(str).str.strip()
    sv_f["date"] = pd.to_numeric(sv_f["date"], errors="coerce")
    sv_f["sum_click"] = pd.to_numeric(sv_f["sum_click"], errors="coerce")
    vle_f["id_site"] = vle_f["id_site"].astype(str).str.strip()
    vle_map = vle_f[
        ["id_site", "activity_type", "week_from", "week_to"]
    ].drop_duplicates(subset=["id_site"], keep="first")
    dup_act = vle_f.groupby("id_site")["activity_type"].nunique().reset_index(name="n")
    if (dup_act["n"] > 1).any():
        bad = dup_act.loc[dup_act["n"] > 1, "id_site"].head(5).tolist()
        raise ValueError(
            f"vle.csv 存在同一 id_site 多种 activity_type，示例 id_site={bad}"
        )

    n_before = len(sv_f)
    merged = sv_f.merge(vle_map, on="id_site", how="left")
    dropped = int(merged["activity_type"].isna().sum())
    if dropped:
        print(
            f"[WARN] {code_m} {code_p}: {dropped}/{n_before} 行 studentVle 无 vle 匹配，已丢弃",
            file=sys.stderr,
        )
    merged = merged[merged["activity_type"].notna()]
    merged = merged[merged["sum_click"].fillna(0) > 0]
    if merged.empty:
        return pd.DataFrame()

    gcols = [
        "id_student",
        "code_module",
        "code_presentation",
        "id_site",
        "date",
    ]
    out = merged.groupby(gcols, as_index=False).agg(
        total_clicks=("sum_click", "sum"),
        activity_type=("activity_type", "first"),
    )
    out = out[out["total_clicks"] > 0].copy()
    return out


def build_unified_features_vlesite(
    student_ids: np.ndarray,
    code_m: str,
    site_keys: List[Tuple[str, str]],
    pres_ordered: List[str],
    edges_df: pd.DataFrame,
    student_info_df: pd.DataFrame,
    student_reg_df: pd.DataFrame,
    courses_full: pd.DataFrame,
    vle_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    站点节点 (presentation, id_site)：activity one-hot + module/presentation one-hot + length_z +
    week_from/week_to z-score；学生行与 module 图一致，右侧补零。
    """
    info_all = student_info_df[
        (student_info_df["code_module"] == code_m)
        & (student_info_df["code_presentation"].isin(pres_ordered))
    ].copy()
    reg_all = student_reg_df[
        (student_reg_df["code_module"] == code_m)
        & (student_reg_df["code_presentation"].isin(pres_ordered))
    ].copy()
    if info_all.empty or reg_all.empty:
        raise ValueError(f"studentInfo/Registration 无 {code_m} 与 {pres_ordered} 的记录")

    info_all["id_student"] = info_all["id_student"].astype(str).str.strip()
    reg_all["id_student"] = reg_all["id_student"].astype(str).str.strip()

    def canonical_pres(sid: int) -> str:
        ps = edges_df.loc[edges_df["id_student"] == sid, "code_presentation"].unique()
        ps = sorted(str(p) for p in ps)
        if not ps:
            raise RuntimeError(f"学生 {sid} 无边却出现在节点集")
        return ps[0]

    rows = []
    for sid in student_ids:
        sid_s = str(int(sid)).strip()
        cp = canonical_pres(int(sid))
        inf = info_all[
            (info_all["id_student"] == sid_s)
            & (info_all["code_presentation"] == cp)
        ].drop_duplicates("id_student", keep="first")
        rg = reg_all[
            (reg_all["id_student"] == sid_s)
            & (reg_all["code_presentation"] == cp)
        ].drop_duplicates("id_student", keep="first")
        if inf.empty or rg.empty:
            raise KeyError(
                f"学生 {sid} 缺少 ({code_m},{cp}) 的 studentInfo 或 studentRegistration"
            )
        rows.append((sid_s, inf.iloc[0], rg.iloc[0]))

    cat_specs = [
        ("gender", "gender"),
        ("region", "region"),
        ("highest_education", "highest_education"),
        ("imd_band", "imd_band"),
        ("age_band", "age_band"),
        ("disability", "disability"),
    ]
    stud_cat_blocks = []
    stud_cat_names: List[str] = []
    for col, prefix in cat_specs:
        cats = fit_categories(info_all[col])
        vals = pd.Series([r[1][col] for r in rows])
        M, names = one_hot_matrix(vals, cats, prefix)
        stud_cat_blocks.append(M)
        stud_cat_names.extend(names)

    def to_num_info(series_list):
        arr = np.zeros((len(series_list),), dtype=np.float64)
        for i, ser in enumerate(series_list):
            v = ser["num_of_prev_attempts"]
            arr[i] = float(v) if str(v).strip() != "" else np.nan
        return arr.reshape(-1, 1)

    def to_num_credits(series_list):
        arr = np.zeros((len(series_list),), dtype=np.float64)
        for i, ser in enumerate(series_list):
            v = ser["studied_credits"]
            arr[i] = float(v) if str(v).strip() != "" else np.nan
        return arr.reshape(-1, 1)

    def to_num_reg(series_list, key):
        arr = np.zeros((len(series_list),), dtype=np.float64)
        for i, ser in enumerate(series_list):
            v = ser[key]
            vs = str(v).strip()
            arr[i] = np.nan if vs == "" else float(vs)
        return arr.reshape(-1, 1)

    num_prev = to_num_info([r[1] for r in rows])
    num_cred = to_num_credits([r[1] for r in rows])
    num_dreg = to_num_reg([r[2] for r in rows], "date_registration")
    num_dunr = to_num_reg([r[2] for r in rows], "date_unregistration")

    stud_num = np.hstack([num_prev, num_cred, num_dreg, num_dunr])
    stud_num_names = [
        "num_of_prev_attempts",
        "studied_credits",
        "date_registration",
        "date_unregistration",
    ]
    stud_num_z, _, _ = standardize_block(
        np.nan_to_num(stud_num, nan=np.nanmedian(stud_num, axis=0))
    )

    stud_cat = (
        np.hstack(stud_cat_blocks) if stud_cat_blocks else np.zeros((len(rows), 0))
    )
    stud_block = np.hstack([stud_cat, stud_num_z])
    stud_block_w = stud_block.shape[1]

    mod_cats = sorted(courses_full["code_module"].astype(str).unique().tolist())
    pres_cats = sorted(courses_full["code_presentation"].astype(str).unique().tolist())
    lengths = courses_full["module_presentation_length"].astype(float).values.reshape(
        -1, 1
    )
    len_mu = float(np.mean(lengths))
    len_sig = float(np.std(lengths)) or 1.0

    vle_sub = vle_df[
        (vle_df["code_module"] == code_m)
        & (vle_df["code_presentation"].isin([sk[0] for sk in site_keys]))
    ].copy()
    vle_sub["id_site"] = vle_sub["id_site"].astype(str).str.strip()
    act_vals: List[str] = []
    week_from_list: List[float] = []
    week_to_list: List[float] = []
    for cp, sid in site_keys:
        sub = vle_sub[
            (vle_sub["code_presentation"] == cp)
            & (vle_sub["id_site"] == str(sid).strip())
        ]
        if sub.empty:
            raise ValueError(f"vle.csv 无站点 ({code_m},{cp},{sid})")
        r0 = sub.iloc[0]
        act_vals.append(str(r0["activity_type"]))
        week_from_list.append(_parse_week_val(r0["week_from"]))
        week_to_list.append(_parse_week_val(r0["week_to"]))

    act_cats = sorted(set(act_vals))
    act_mat, act_names = one_hot_matrix(pd.Series(act_vals), act_cats, "vle_activity")

    site_feat_rows = []
    for k, (cp, _) in enumerate(site_keys):
        row_len = courses_full[
            (courses_full["code_module"] == code_m)
            & (courses_full["code_presentation"] == cp)
        ]["module_presentation_length"].astype(float)
        if row_len.empty:
            raise ValueError(f"courses.csv 无 {code_m} {cp} 长度")
        length_val = float(row_len.iloc[0])
        length_z = (length_val - len_mu) / len_sig
        mod_oh = one_hot_matrix(pd.Series([code_m]), mod_cats, "module")[0]
        pres_oh = one_hot_matrix(pd.Series([cp]), pres_cats, "presentation")[0]
        row = np.hstack(
            [
                act_mat[k : k + 1, :],
                mod_oh,
                pres_oh,
                np.array([[length_z]], dtype=np.float64),
            ]
        )
        site_feat_rows.append(row)
    site_core = np.vstack(site_feat_rows)
    weeks = np.array([week_from_list, week_to_list], dtype=np.float64).T
    for j in range(weeks.shape[1]):
        col = weeks[:, j]
        m = float(np.nanmedian(col))
        if not np.isfinite(m):
            m = 0.0
        col = np.where(np.isnan(col), m, col)
        weeks[:, j] = col
    weeks_z, _, _ = standardize_block(weeks)
    site_tail = np.hstack([site_core, weeks_z])

    _, mod_names = one_hot_matrix(pd.Series([code_m]), mod_cats, "module")
    _, pres_names = one_hot_matrix(
        pd.Series([site_keys[0][0]]), pres_cats, "presentation"
    )
    site_tail_names = (
        act_names + mod_names + pres_names + ["module_presentation_length_z", "vle_week_from_z", "vle_week_to_z"]
    )
    site_tail_w = site_tail.shape[1]

    z_pad = np.zeros((len(rows), site_tail_w), dtype=np.float64)
    student_rows = np.hstack([stud_block, z_pad])

    z_stud = np.zeros((len(site_keys), stud_block_w), dtype=np.float64)
    site_rows = np.hstack([z_stud, site_tail])

    feat_names = stud_cat_names + stud_num_names + site_tail_names
    if student_rows.shape[1] != site_rows.shape[1]:
        raise RuntimeError("特征拼接维度不一致")
    all_feats = np.vstack([student_rows, site_rows])
    return all_feats, feat_names


def process_module_vlesite_merge(
    tables: Dict[str, pd.DataFrame],
    code_m: str,
    out_dir: str,
    pairs: Optional[List[Tuple[str, str]]] = None,
    file_tag: Optional[str] = None,
) -> None:
    if pairs is None:
        pairs = allowed_presentations_for_module(code_m)
    parts: List[pd.DataFrame] = []
    for cm, cp in pairs:
        ed = _aggregate_vlesite_edges_one_pres(tables, cm, cp)
        if not ed.empty:
            parts.append(ed)
    if not parts:
        raise ValueError(f"模块 {code_m}：所有 presentation 均无有效站点边")

    edges_df = pd.concat(parts, ignore_index=True)
    edges_df["id_site"] = edges_df["id_site"].astype(str).str.strip()

    stud_set = np.sort(edges_df["id_student"].dropna().unique().astype(np.int64))
    if stud_set.size == 0:
        raise ValueError("无学生节点")

    pres_ordered = sorted(edges_df["code_presentation"].astype(str).unique().tolist())
    pres_to_idx = {p: j for j, p in enumerate(pres_ordered)}
    sid_to_local = {int(s): i + 1 for i, s in enumerate(stud_set)}

    site_tuples = edges_df[["code_presentation", "id_site"]].drop_duplicates()
    site_keys = sorted(
        zip(
            site_tuples["code_presentation"].astype(str),
            site_tuples["id_site"].astype(str),
        ),
        key=lambda x: (x[0], x[1]),
    )
    site_to_nid = {sk: int(stud_set.size + 1 + j) for j, sk in enumerate(site_keys)}
    n_sites = len(site_keys)

    u = edges_df["id_student"].map(lambda x: sid_to_local[int(x)]).astype(np.int64)
    i = edges_df.apply(
        lambda r: site_to_nid[(str(r["code_presentation"]), str(r["id_site"]))],
        axis=1,
    ).astype(np.int64)
    d_int = edges_df["date"].astype(np.int64)
    blk = edges_df["code_presentation"].map(lambda x: pres_to_idx[str(x)]).astype(
        np.int64
    )
    ts = blk * TS_PRES_BLOCK + d_int

    order = np.lexsort((i.values, u.values, ts.values))
    edges_df = edges_df.iloc[order].reset_index(drop=True)
    u = u.iloc[order].to_numpy()
    i = i.iloc[order].to_numpy()
    ts = ts.iloc[order].to_numpy()

    if not np.all(ts[:-1] <= ts[1:]):
        raise RuntimeError("ts 非单调非递减")

    act_cats = sorted(edges_df["activity_type"].fillna("__MISSING__").astype(str).unique().tolist())
    for ac in act_cats:
        col = f"activity_type_{_sanitize_token(ac)}"
        edges_df[col] = (
            edges_df["activity_type"].fillna("__MISSING__").astype(str) == ac
        ).astype(np.float64)

    n_e = len(edges_df)
    idx = np.arange(1, n_e + 1, dtype=np.int64)
    label = np.ones(n_e, dtype=np.int64)
    n_nodes = int(stud_set.size + n_sites)
    tag = file_tag if file_tag is not None else code_m

    os.makedirs(out_dir, exist_ok=True)
    ml_path = os.path.join(out_dir, f"ml_{tag}.csv")
    with open(ml_path, "w", encoding="utf-8") as f:
        f.write(",u,i,ts,label,idx\n")
        for row in range(n_e):
            f.write(
                f"{row},{int(u[row])},{int(i[row])},{int(ts[row])},"
                f"{int(label[row])},{int(idx[row])}\n"
            )

    attr_cols = ["total_clicks"] + sorted(
        c for c in edges_df.columns if c.startswith("activity_type_")
    )
    edge_attr = edges_df[attr_cols].copy()
    for c in attr_cols:
        edge_attr[c] = pd.to_numeric(edge_attr[c], errors="coerce").fillna(0)
    edge_attr.insert(0, "idx", idx)
    edge_attr.to_csv(os.path.join(out_dir, "edge_attr.csv"), index=False)

    feats, feat_names = build_unified_features_vlesite(
        stud_set,
        code_m,
        site_keys,
        pres_ordered,
        edges_df,
        tables["student_info"],
        tables["student_reg"],
        tables["courses"],
        tables["vle"],
    )
    if feats.shape[0] != n_nodes:
        raise RuntimeError(f".content 行数 {feats.shape[0]} != 节点数 {n_nodes}")

    np.savetxt(
        os.path.join(out_dir, f"{tag}.content"),
        feats,
        delimiter=",",
        fmt="%.12g",
    )

    rows_map = [(sid_to_local[s], f"student::{s}") for s in sid_to_local]
    for sk, nid in site_to_nid.items():
        cp, sid = sk
        rows_map.append((nid, f"site::{code_m}_{cp}_{sid}"))
    rows_map.sort(key=lambda x: x[0])
    pd.DataFrame(rows_map, columns=["node_id", "raw_id"]).to_csv(
        os.path.join(out_dir, "node_map.csv"), index=False
    )

    val_time, test_time = [
        float(x) for x in list(np.quantile(ts.astype(np.float64), [0.10, 0.20]))
    ]
    train_mask = ts <= val_time
    val_mask = (ts > val_time) & (ts <= test_time)
    test_mask = ts > test_time

    def write_idx(path: str, mask: np.ndarray) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for v in idx[mask]:
                f.write(f"{int(v)}\n")

    write_idx(os.path.join(out_dir, "train_idx.txt"), train_mask)
    write_idx(os.path.join(out_dir, "val_idx.txt"), val_mask)
    write_idx(os.path.join(out_dir, "test_idx.txt"), test_mask)

    edge_u_full, edge_i_full = u, i
    total_nodes = set(np.unique(np.stack([edge_u_full, edge_i_full])))
    train_nodes = set(
        np.unique(np.stack([edge_u_full[train_mask], edge_i_full[train_mask]]))
    )
    new_nodes = total_nodes - train_nodes
    nn_edge = np.array(
        [
            (a in new_nodes or b in new_nodes)
            for a, b in zip(edge_u_full, edge_i_full)
        ]
    )
    write_idx(os.path.join(out_dir, "nn_val_idx.txt"), val_mask & nn_edge)
    write_idx(os.path.join(out_dir, "nn_test_idx.txt"), test_mask & nn_edge)

    used = set(np.unique(np.stack([u, i])))
    expected = set(range(1, n_nodes + 1))
    if used != expected:
        raise RuntimeError(f"节点编号不连续: 缺 {expected - used} 多 {used - expected}")

    per_student_days = (
        edges_df.groupby("id_student")["date"].nunique().astype(float).values
    )
    mean_days = float(np.mean(per_student_days)) if len(per_student_days) else 0.0

    stats = {
        "code_module": code_m,
        "graph": "vlesite",
        "code_presentations_merged": pres_ordered,
        "ts_encoding": f"ts = pres_index * {TS_PRES_BLOCK} + date（相对日）",
        "num_students": int(stud_set.size),
        "num_site_nodes": n_sites,
        "num_nodes": n_nodes,
        "num_edges": int(n_e),
        "time_min": int(ts.min()),
        "time_max": int(ts.max()),
        "mean_active_days_per_student": mean_days,
        "split_quantiles_ts": {"val_time": val_time, "test_time": test_time},
        "split_note": "与 ContraTGT utils.Dataset 一致",
        "feature_dim": int(feats.shape[1]),
        "feature_names": feat_names,
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(
        f"[{tag}] 学生={stats['num_students']} 站点节点={n_sites} "
        f"总节点={n_nodes} 边={n_e} ts∈[{stats['time_min']},{stats['time_max']}] "
        f"presentations={pres_ordered}"
    )


def main() -> None:
    # data/scripts/ -> 项目根 EduTGT/EduTGT
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_raw = os.path.join(_root, "data", "raw", "OULAD")
    default_out = os.path.join(_root, "data", "processed")

    p = argparse.ArgumentParser(description="OULAD -> ContraTGT 预处理")
    p.add_argument(
        "--raw_dir",
        type=str,
        default=default_raw,
        help="OULAD CSV 目录（含 courses.csv 等）",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default=default_out,
        help="输出根目录（默认 data/processed，其下创建 data_{module}/）",
    )
    p.add_argument(
        "--module",
        type=str,
        required=True,
        help="整模块 AAA、单实例 AAA-2013J，或 ALL（按模块 AAA→…→GGG）；均为学生–VLE 站点图",
    )
    args = p.parse_args()

    kind, mod, pres = parse_module_arg(args.module)
    tables = load_raw_tables(args.raw_dir)

    def out_subdir_whole(code_m: str) -> str:
        return os.path.join(args.out_root, f"data_{code_m}")

    if kind == "all":
        for code_m in modules_in_allowed_order():
            process_module_vlesite_merge(tables, code_m, out_subdir_whole(code_m))
    elif kind == "pres":
        pairs = [(mod, pres)]
        sub = os.path.join(args.out_root, f"data_{mod}_{pres}")
        process_module_vlesite_merge(
            tables,
            mod,
            sub,
            pairs=pairs,
            file_tag=f"{mod}_{pres}",
        )
    else:
        process_module_vlesite_merge(tables, mod, out_subdir_whole(mod))

    print("预处理完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        raise
