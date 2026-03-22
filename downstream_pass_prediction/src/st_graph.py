"""加载与 main.py 一致的时序图、重映射特征及 TimeEncode。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from repo_path import ensure_repo_on_path

ensure_repo_on_path()

from sampling import get_adj_list, get_interaction_list, init_offset  # noqa: E402
from utils import Dataset, TimeEncode, normalize_features  # noqa: E402


@dataclass
class GraphBundle:
    edge: Dict[str, Any]
    num_nodes: int
    features: np.ndarray
    indim: int
    node_l: np.ndarray
    ts_l: np.ndarray
    idx_l: np.ndarray
    offset_l: np.ndarray
    interaction_list: Any
    time_encode: TimeEncode
    time_information: TimeEncode


def build_old_to_new_map(ml_path: str, edge_idx: torch.Tensor) -> Dict[int, int]:
    """用 ml 原始 u,v 与 Dataset 重映射后的 edge['idx'] 对齐，得到 old_ml_id -> remapped_id。"""
    df = pd.read_csv(ml_path)
    m: Dict[int, int] = {}
    n = min(len(df), edge_idx.shape[0])
    for i in range(n):
        ou, ov = int(df["u"].iloc[i]), int(df["i"].iloc[i])
        ru, rv = int(edge_idx[i, 0]), int(edge_idx[i, 1])
        m[ou] = ru
        m[ov] = rv
    return m


def load_graph_bundle(ml_path: str, content_path: str) -> GraphBundle:
    edge, num_nodes, _nodes_list, _node_time, _tr, _te, _va, _nn1, _nn2 = Dataset(
        file=str(ml_path)
    )
    df = pd.read_csv(ml_path)
    feat_df = pd.read_csv(content_path, header=None)
    feat_csr = sp.csr_matrix(feat_df.values.astype(np.float64))
    feat_t = normalize_features(feat_csr)
    if sp.issparse(feat_t):
        feat_t = torch.from_numpy(feat_t.toarray()).float()
    else:
        feat_t = torch.from_numpy(np.asarray(feat_t)).float()
    _pad = (4 - (feat_t.shape[1] % 4)) % 4
    if _pad:
        feat_t = torch.cat(
            [feat_t, torch.zeros(feat_t.shape[0], _pad, dtype=feat_t.dtype)], dim=1
        )
    raw = feat_t.cpu().numpy()
    indim = raw.shape[1]

    old_to_new = build_old_to_new_map(ml_path, edge["idx"])
    features = np.zeros((num_nodes, indim), dtype=np.float64)
    for old_id, new_id in old_to_new.items():
        if old_id >= 1 and old_id <= raw.shape[0]:
            features[new_id] = raw[old_id - 1]

    adj_list = get_adj_list(edge)
    node_l, ts_l, idx_l, offset_l = init_offset(adj_list)
    interaction_list = get_interaction_list(edge)
    time_encode = TimeEncode(time_dim=indim, time_encoding="concat")
    time_information = TimeEncode(time_dim=indim, time_encoding="sum")

    return GraphBundle(
        edge=edge,
        num_nodes=num_nodes,
        features=features,
        indim=indim,
        node_l=node_l,
        ts_l=ts_l,
        idx_l=idx_l,
        offset_l=offset_l,
        interaction_list=interaction_list,
        time_encode=time_encode,
        time_information=time_information,
    )


def build_anchor_lists(
    ml_path: str,
    labels_df: pd.DataFrame,
    presentations: List[str],
    t_star_relative_day: int | None,
    block: int = 1000,
) -> Tuple[List[Tuple[int, int, int, int, int]], List[int]]:
    """
    每个标签行一条锚点边（原始 ml 坐标 u,i,ts,idx）及 is_pass；并行返回 id_student。
    取该生在该 presentation 下、满足 t* 的最后一条边。
    """
    df = pd.read_csv(ml_path)
    anchors: List[Tuple[int, int, int, int, int]] = []
    students: List[int] = []
    for _, row in labels_df.iterrows():
        ml_nid = int(row["ml_node_id"])
        cp = str(row["code_presentation"])
        y = int(row["is_pass"])
        blk = presentations.index(cp)
        sub = df[(df["u"] == ml_nid) & (df["ts"] // block == blk)]
        if t_star_relative_day is not None:
            sub = sub[(sub["ts"] % block) <= int(t_star_relative_day)]
        if sub.empty:
            continue
        sub = sub.sort_values("ts")
        r = sub.iloc[-1]
        anchors.append((int(r["u"]), int(r["i"]), int(r["ts"]), int(r["idx"]), y))
        students.append(int(row["id_student"]))
    return anchors, students


def pack_anchors(
    edge_idx: torch.Tensor,
    ml_df: pd.DataFrame,
    anchors: List[Tuple[int, int, int, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """锚点对应 Dataset 重映射后的边行 (B,4) 与标签 (B,)。"""
    keys = list(
        zip(
            ml_df["u"].astype(int),
            ml_df["i"].astype(int),
            ml_df["ts"].astype(int),
            ml_df["idx"].astype(int),
        )
    )
    key_to_row = {k: i for i, k in enumerate(keys)}
    rows: List[List[int]] = []
    ys: List[float] = []
    for ou, ov, ts, jdx, y in anchors:
        ri = key_to_row.get((ou, ov, ts, jdx))
        if ri is None:
            raise RuntimeError(f"锚点边未在 ml 中找到: {(ou, ov, ts, jdx)}")
        rows.append([int(edge_idx[ri, j]) for j in range(4)])
        ys.append(float(y))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(ys, dtype=torch.float32)
