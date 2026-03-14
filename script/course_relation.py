"""
课程关系模块：从训练边学习课程共现关系，用于带约束的负采样增强。
两门课程「相关」= 被同一学生选过（共现）。
"""
import numpy as np
from collections import defaultdict


def build_course_relation(train_idx, dst_nodes=None, top_k=50, min_cooccur=1):
    """
    从训练边构建课程-课程共现关系。
    train_idx: (N, 4) 训练边 [u, i, ts, idx]，i 为课程（目标节点）
    dst_nodes: 可选，目标节点集合；若未提供则从 train_idx 提取
    top_k: 每门课程保留最多 top_k 个最相关课程
    min_cooccur: 最少共现次数

    Returns:
        course_related: dict, course_id -> np.array of related course ids (按共现强度排序)
        dst_nodes: np.array
    """
    if dst_nodes is None:
        dst_nodes = np.unique(train_idx[:, 1]).astype(np.int64)
    dst_set = set(dst_nodes.tolist())

    # 按学生聚合：student -> list of courses
    student_courses = defaultdict(set)
    for row in train_idx:
        u, i = int(row[0]), int(row[1])
        if i in dst_set:
            student_courses[u].add(i)

    # 课程共现矩阵：course_i 与 course_j 的共现次数
    course_cooccur = defaultdict(lambda: defaultdict(int))
    for u, courses in student_courses.items():
        courses = list(courses)
        for i in range(len(courses)):
            for j in range(i + 1, len(courses)):
                ci, cj = courses[i], courses[j]
                course_cooccur[ci][cj] += 1
                course_cooccur[cj][ci] += 1

    # 对每门课程，取 top_k 个最相关课程
    course_related = {}
    for c in dst_nodes:
        c = int(c)
        related = [(j, cnt) for j, cnt in course_cooccur[c].items() if cnt >= min_cooccur and j != c]
        related.sort(key=lambda x: -x[1])
        related = related[:top_k]
        course_related[c] = np.array([r[0] for r in related], dtype=np.int64) if related else np.array([], dtype=np.int64)

    return course_related, dst_nodes


class CourseAwareNegSampler:
    """
    带课程关系约束的负采样器：以概率 p_related 从「相关课程」中采样负样本（更难），
    否则从全量目标节点随机采样。
    """

    def __init__(self, course_related, dst_nodes, p_related=0.6, seed=None):
        self.course_related = course_related
        self.dst_nodes = np.asarray(dst_nodes, dtype=np.int64)
        self.p_related = p_related
        self.rng = np.random.default_rng(seed)

    def sample(self, i_pos):
        """
        i_pos: (n,) 正边的目标节点（课程）
        Returns: (n,) 负样本目标节点，保证 neg[b] != i_pos[b]
        """
        n = len(i_pos)
        neg = np.empty(n, dtype=np.int64)
        for b in range(n):
            i = int(i_pos[b])
            if self.rng.random() < self.p_related and i in self.course_related and len(self.course_related[i]) > 0:
                cand = self.course_related[i]
                cand = cand[cand != i]
                if len(cand) > 0:
                    neg[b] = self.rng.choice(cand)
                else:
                    neg[b] = self._random_exclude(i)
            else:
                neg[b] = self._random_exclude(i)
        return neg

    def _random_exclude(self, i):
        cand = self.dst_nodes[self.dst_nodes != i]
        if len(cand) == 0:
            return self.rng.choice(self.dst_nodes)
        return self.rng.choice(cand)
