"""
整模块 OULAD 图：课程实例 (presentation) 关系矩阵 + 训练期按 W 偏置负样本 dst。
与 preprocess_oulad_for_contratgt 中 ts = pres_index * TS_PRES_BLOCK + date 一致。
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

TS_PRES_BLOCK = 1000


def ml_compact_uniq_raw(ml_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """与 utils.Dataset 相同的节点压缩映射；返回 uniq_raw, u_compact, v_compact, ts。"""
    with open(ml_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    edges = torch.tensor(
        [[float(r) for r in row.split(",")] for row in lines[1:]], dtype=torch.long
    )
    uv = edges[:, [1, 2]]
    uniq_vals, inv = uv.unique(return_inverse=True)
    cu = inv[:, 0].cpu().numpy()
    cv = inv[:, 1].cpu().numpy()
    ts = edges[:, 3].numpy()
    return uniq_vals.cpu().numpy(), cu, cv, ts


def _parse_site_presentation(raw_id: str, presentations: List[str]) -> int:
    if not raw_id.startswith("site::"):
        return -1
    body = raw_id.split("::", 1)[1]
    for p in presentations:
        if f"_{p}_" in body:
            return presentations.index(p)
    return -1


def load_node_raw_to_pres(
    node_map_path: str, presentations: List[str]
) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with open(node_map_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row["node_id"])
            rid = row["raw_id"].strip()
            pi = _parse_site_presentation(rid, presentations)
            out[nid] = pi
    return out


def jaccard_pres_matrix_from_node_map(
    node_map_path: str, presentations: List[str]
) -> np.ndarray:
    """按各 presentation 下站点 id_site 集合（字符串）算 Jaccard，对角为 1。"""
    P = len(presentations)
    sets: List[set] = [set() for _ in range(P)]
    with open(node_map_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["raw_id"].strip()
            if not rid.startswith("site::"):
                continue
            body = rid.split("::", 1)[1]
            parts = body.split("_")
            if len(parts) < 3:
                continue
            site_key = parts[-1]
            for i, p in enumerate(presentations):
                if f"_{p}_" in rid:
                    sets[i].add(site_key)
                    break
    W = np.eye(P, dtype=np.float64)
    for i in range(P):
        for j in range(i + 1, P):
            a, b = sets[i], sets[j]
            if not a and not b:
                inter, union = 0.0, 1.0
            else:
                inter = len(a & b)
                uni = len(a | b)
                v = inter / uni if uni else 0.0
            W[i, j] = W[j, i] = v
    return W


def load_stats_presentations(stats_path: str) -> List[str]:
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    pres = stats.get("code_presentations_merged")
    if not pres:
        raise ValueError(f"{stats_path} 缺少 code_presentations_merged")
    return list(pres)


def build_pres_relation_dict(data_dir: str) -> Dict[str, Any]:
    stats_path = os.path.join(data_dir, "stats.json")
    node_map_path = os.path.join(data_dir, "node_map.csv")
    presentations = load_stats_presentations(stats_path)
    W = jaccard_pres_matrix_from_node_map(node_map_path, presentations)
    return {
        "ts_pres_block": TS_PRES_BLOCK,
        "presentations": presentations,
        "W": W.tolist(),
        "description": "站点 id_site 集合的 Jaccard，对角为 1",
    }


def save_pres_relation_json(data_dir: str, path: Optional[str] = None) -> str:
    d = build_pres_relation_dict(data_dir)
    if path:
        out = path if os.path.isabs(path) else os.path.join(data_dir, path)
    else:
        out = os.path.join(data_dir, "pres_relation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    return out


def load_pres_relation_json(path: str) -> Tuple[List[str], np.ndarray, int]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    presentations = list(d["presentations"])
    W = np.asarray(d["W"], dtype=np.float64)
    blk = int(d.get("ts_pres_block", TS_PRES_BLOCK))
    return presentations, W, blk


def resolve_pres_relation_path(ml_path: str, arg_path: Optional[str]) -> Optional[str]:
    data_dir = os.path.dirname(os.path.abspath(ml_path))
    if arg_path:
        p = arg_path if os.path.isabs(arg_path) else os.path.normpath(os.path.join(data_dir, arg_path))
        if os.path.isfile(p):
            return p
        return None
    auto = os.path.join(data_dir, "pres_relation.json")
    if os.path.isfile(auto):
        return auto
    return None


def build_dst_pres_arrays(
    train_v_compact: np.ndarray,
    uniq_raw: np.ndarray,
    raw_to_pres: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """train 中出现过的 dst 紧凑 id 及其 presentation 索引（站点；-1 未知）。"""
    dst_list = np.unique(train_v_compact.astype(np.int64))
    pres_arr = np.full(len(dst_list), -1, dtype=np.int64)
    for j, d in enumerate(dst_list):
        raw = int(uniq_raw[int(d)])
        pres_arr[j] = int(raw_to_pres.get(raw, -1))
    return dst_list, pres_arr


class PresBiasedDstSampler:
    """
    负样本 dst：以 (1-mix)*P(dst|W[p,:]) + mix*Uniform 采样；
    p 为当前正边 ts 对应的 presentation 块索引。
    """

    def __init__(
        self,
        dst_list: np.ndarray,
        dst_pres: np.ndarray,
        W: np.ndarray,
        mix_uniform: float = 0.35,
    ):
        self.dst_list = np.asarray(dst_list, dtype=np.int64)
        self.dst_pres = np.asarray(dst_pres, dtype=np.int64)
        self.W = np.asarray(W, dtype=np.float64)
        self.mix_uniform = float(mix_uniform)
        self.n_pres = W.shape[0]
        n = len(self.dst_list)
        self._probs = np.zeros((self.n_pres, n), dtype=np.float64)
        uni = np.ones(n, dtype=np.float64) / max(n, 1)
        for p in range(self.n_pres):
            w_row = np.zeros(n, dtype=np.float64)
            for j in range(n):
                q = int(self.dst_pres[j])
                if q < 0:
                    w_row[j] = 1.0
                else:
                    w_row[j] = max(float(self.W[p, q]), 1e-8)
            s = float(w_row.sum())
            if s <= 0:
                biased = uni.copy()
            else:
                biased = w_row / s
            m = self.mix_uniform
            self._probs[p] = m * uni + (1.0 - m) * biased
            self._probs[p] /= self._probs[p].sum()

    def sample(self, batch_ts: torch.Tensor) -> np.ndarray:
        ts = batch_ts.detach().cpu().numpy().astype(np.int64)
        pres = ts // TS_PRES_BLOCK
        pres = np.clip(pres, 0, self.n_pres - 1)
        node_sum = len(pres)
        out = np.zeros(node_sum, dtype=np.int64)
        for i in range(node_sum):
            p = int(pres[i])
            out[i] = np.random.choice(self.dst_list, p=self._probs[p])
        return out


def make_train_pres_biased_sampler(
    train_idx_v: torch.Tensor,
    ml_path: str,
    data_dir: str,
    relation_path: str,
    mix_uniform: float,
) -> PresBiasedDstSampler:
    presentations, W, blk = load_pres_relation_json(relation_path)
    if blk != TS_PRES_BLOCK:
        raise ValueError(f"pres_relation ts_pres_block={blk} 与 course_pres.TS_PRES_BLOCK 不一致")
    uniq_raw, _, cv, _ = ml_compact_uniq_raw(ml_path)
    node_map_path = os.path.join(data_dir, "node_map.csv")
    raw_to_pres = load_node_raw_to_pres(node_map_path, presentations)
    train_v = train_idx_v.detach().cpu().numpy().astype(np.int64)
    dst_list, dst_pres = build_dst_pres_arrays(train_v, uniq_raw, raw_to_pres)
    return PresBiasedDstSampler(dst_list, dst_pres, W, mix_uniform=mix_uniform)


def sample_fake_dst(sampler, node_sum: int, batch_ts: torch.Tensor):
    if isinstance(sampler, PresBiasedDstSampler):
        return sampler.sample(batch_ts)
    return sampler.sample(node_sum)
