"""
增强负采样：模块约束、时序共现、度感知；与 utils.RandEdgeSampler 接口兼容（sample(size) 或 sample(i_pos)）。
"""
import numpy as np
from collections import defaultdict


def _edges_idx_np(edges_idx):
    if hasattr(edges_idx, 'numpy'):
        return edges_idx.numpy()
    return np.asarray(edges_idx)


def build_pseudo_module_from_edges(edges_idx, jaccard_threshold=0.1):
    """
    从全量边构建课程伪模块：按 Jaccard 相似度聚类（两门课选课学生重叠多则同模块）。
    edges_idx: (N, 4) [u, i, ts, idx]，i 为课程（目标节点）
    Returns: course_id -> list of same-module course ids (不含自身)
    """
    idx = _edges_idx_np(edges_idx)
    dst_set = np.unique(idx[:, 1])
    course_students = defaultdict(set)
    for row in idx:
        u, i = int(row[0]), int(row[1])
        course_students[i].add(u)
    # 简单聚类：Jaccard(c_i, c_j) >= threshold 则同组，用并查集
    parent = {c: c for c in dst_set}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for i, ci in enumerate(dst_set):
        for cj in dst_set[i + 1:]:
            si, sj = course_students[ci], course_students[cj]
            if not si or not sj:
                continue
            jaccard = len(si & sj) / len(si | sj)
            if jaccard >= jaccard_threshold:
                union(ci, cj)
    # 每组内课程列表
    module_to_courses = defaultdict(list)
    for c in dst_set:
        module_to_courses[find(c)].append(c)
    course_related = {}
    for c in dst_set:
        group = module_to_courses[find(c)]
        same = [x for x in group if x != c]
        course_related[c] = np.array(same, dtype=np.int64) if same else np.array([], dtype=np.int64)
    return course_related, dst_set


def build_temporal_cooccurrence(edges_idx, time_bin_ratio=0.1):
    """
    从全量边构建 (学生, 时间桶) -> 该学生在该时间附近选过的课程列表。
    time_bin_ratio: 时间分成多少份（按分位数），相邻桶视为“相同时段”。
    Returns: (u, bin_id) -> list of course ids
    """
    idx = _edges_idx_np(edges_idx)
    ts = idx[:, 2]
    n = len(ts)
    bins = np.quantile(ts, np.linspace(0, 1, max(2, int(1 / time_bin_ratio) + 1)))
    u_bin_to_courses = defaultdict(set)
    for row in idx:
        u, i, t = int(row[0]), int(row[1]), float(row[2])
        bin_id = np.searchsorted(bins, t, side='right') - 1
        bin_id = max(0, min(bin_id, len(bins) - 2))
        u_bin_to_courses[(u, bin_id)].add(i)
    # 转为 array 并支持相邻桶
    u_bin_to_courses_list = {k: np.array(list(v), dtype=np.int64) for k, v in u_bin_to_courses.items()}
    return u_bin_to_courses_list, bins


def build_degree_weights(edges_idx, dst_nodes):
    """目标节点度（作为负样本的权重：度越小权重越大，冷门课更常被采为负样本）。"""
    idx = _edges_idx_np(edges_idx)
    deg = defaultdict(int)
    for row in idx:
        deg[int(row[1])] += 1
    dst = np.asarray(dst_nodes, dtype=np.int64)
    counts = np.array([max(1, deg[c]) for c in dst])
    weights = 1.0 / np.sqrt(counts)
    weights /= weights.sum()
    return weights, dst


class EnhancedNegSampler:
    """
    组合负采样：以概率 p_module 从同伪模块、p_temporal 从相同时段、p_degree 按度加权随机；
    其余用均匀随机。接口 sample(n) 或 sample(i_pos) 均可（i_pos 用于需要正样本课程信息的策略）。
    """

    def __init__(self, edges_idx, train_dst_nodes, seed=None,
                 p_module=0.25, p_temporal=0.25, p_degree=0.25,
                 jaccard_threshold=0.08, time_bin_ratio=0.1):
        self.rng = np.random.default_rng(seed)
        self.train_dst = np.unique(np.asarray(train_dst_nodes, dtype=np.int64))
        self.all_idx = _edges_idx_np(edges_idx)

        # 1) 伪模块
        self.course_related, _ = build_pseudo_module_from_edges(self.all_idx, jaccard_threshold=jaccard_threshold)
        self.p_module = p_module

        # 2) 时序共现 (u, ts) -> 需在 sample 时传入 u, ts
        self.u_bin_courses, self.time_bins = build_temporal_cooccurrence(self.all_idx, time_bin_ratio=time_bin_ratio)
        self.time_bin_ratio = time_bin_ratio
        self.p_temporal = p_temporal

        # 3) 度感知
        self.deg_weights, self.deg_dst = build_degree_weights(self.all_idx, self.train_dst)
        self.p_degree = p_degree

        self.p_rand = max(0., 1.0 - p_module - p_temporal - p_degree)

    def _bin_id(self, ts):
        if np.isscalar(ts):
            return max(0, min(np.searchsorted(self.time_bins, ts, side='right') - 1, len(self.time_bins) - 2))
        return np.clip(np.searchsorted(self.time_bins, ts, side='right') - 1, 0, len(self.time_bins) - 2)

    def sample(self, n, u_batch=None, ts_batch=None, i_pos=None):
        """
        n: batch 大小
        u_batch: (n,) 当前边的学生 id，用于时序共现
        ts_batch: (n,) 当前边的时间戳
        i_pos: (n,) 正样本课程 id，用于同模块采样
        """
        if u_batch is None:
            u_batch = np.zeros(n, dtype=np.int64)
        if ts_batch is None:
            ts_batch = np.zeros(n)
        if i_pos is None:
            i_pos = self.rng.choice(self.train_dst, size=n)
        u_batch = np.asarray(u_batch).ravel()[:n]
        ts_batch = np.asarray(ts_batch).ravel()[:n]
        i_pos = np.asarray(i_pos).ravel()[:n]

        neg = np.empty(n, dtype=np.int64)
        for b in range(n):
            r = self.rng.random()
            u, ts, i = u_batch[b], ts_batch[b], int(i_pos[b])
            if r < self.p_module and i in self.course_related and len(self.course_related[i]) > 0:
                cand = self.course_related[i]
                neg[b] = self.rng.choice(cand)
            elif r < self.p_module + self.p_temporal:
                bin_id = self._bin_id(ts) if np.isscalar(ts) else self._bin_id(ts)[0]
                cand = []
                for db in [-1, 0, 1]:
                    key = (int(u), bin_id + db)
                    if key in self.u_bin_courses:
                        cand.extend(self.u_bin_courses[key].tolist())
                cand = [c for c in cand if c != i]
                if cand:
                    neg[b] = self.rng.choice(np.unique(cand))
                else:
                    neg[b] = self._random_exclude(i)
            elif r < self.p_module + self.p_temporal + self.p_degree:
                neg[b] = self._degree_sample_exclude(i)
            else:
                neg[b] = self._random_exclude(i)
        return neg

    def _random_exclude(self, i):
        cand = self.train_dst[self.train_dst != i]
        if len(cand) == 0:
            return self.rng.choice(self.train_dst)
        return self.rng.choice(cand)

    def _degree_sample_exclude(self, i):
        mask = self.deg_dst != i
        if mask.sum() == 0:
            return self.rng.choice(self.train_dst)
        w = self.deg_weights[mask]
        w /= w.sum()
        return self.rng.choice(self.deg_dst[mask], p=w)
