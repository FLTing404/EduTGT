"""
复用主流程 utils.Dataset 的节点重映射、时间分位划分与边表。
不读取 train_idx.txt（主模型 main.py 同样由 Dataset 内分位数切分，与 README_oulad 一致）。
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

# 项目根目录（EduTGT/EduTGT）
_BASELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_BASELINE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import Dataset, RandEdgeSampler, normalize_features  # noqa: E402


@dataclass
class BaselineDataBundle:
    """供各 baseline 使用的统一数据结构。"""

    data_dir: str
    data_name: str
    ml_path: str
    content_path: str
    num_nodes: int
    feat_dim: int
    features: np.ndarray  # [N, F] float32，已行归一化 + 4 倍数 pad（与 main 一致）
    # 全图边（与 Dataset 输出一致，列 u,v,t,idx）
    edge_idx: torch.Tensor  # [E, 4]
    labels: torch.Tensor
    train_mask: torch.Tensor  # [E] bool
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    val_time: float
    test_time: float
    train_data: Dict[str, torch.Tensor]
    val_data: Dict[str, torch.Tensor]
    test_data: Dict[str, torch.Tensor]
    train_rand_sampler: RandEdgeSampler
    # 与 utils.Dataset / ContraTGT 一致：验证/测试时段内「至少一端为训练未出现节点」的边
    nn_val_mask: torch.Tensor  # [E] bool
    nn_test_mask: torch.Tensor  # [E] bool
    nn_val_data: Dict[str, torch.Tensor]
    nn_test_data: Dict[str, torch.Tensor]
    # 全图按时间排序后的边下标（用于时序模型按时间流处理）
    perm_chrono: np.ndarray
    stats: Dict[str, Any]


def _resolve_paths(data_dir: str) -> Tuple[str, str, str, str]:
    root = _PROJECT_ROOT
    if not os.path.isabs(data_dir):
        data_dir = os.path.normpath(os.path.join(root, data_dir))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir 不存在: {data_dir}")
    ml_files = sorted(glob.glob(os.path.join(data_dir, "ml_*.csv")))
    if not ml_files:
        raise FileNotFoundError(f"{data_dir} 下未找到 ml_*.csv")
    content_files = sorted(glob.glob(os.path.join(data_dir, "*.content")))
    if not content_files:
        raise FileNotFoundError(f"{data_dir} 下未找到 *.content")
    ml_path = ml_files[0]
    stem = os.path.basename(ml_path)
    stem = stem[3:-4] if stem.startswith("ml_") and stem.endswith(".csv") else ""
    preferred = [f for f in content_files if stem and stem in os.path.basename(f)]
    content_path = preferred[0] if preferred else content_files[0]
    data_name = os.path.basename(os.path.normpath(data_dir))
    return data_dir, ml_path, content_path, data_name


def load_baseline_bundle(data_dir: str, log_edge_attr: bool = True) -> BaselineDataBundle:
    data_dir, ml_path, content_path, data_name = _resolve_paths(data_dir)

    edge, num_nodes, nodes_list, node_time, train_data, test_data, val_data, nn_test, nn_val = Dataset(
        file=ml_path
    )

    ts = edge["idx"][:, 2].float().numpy()
    val_time, test_time = [float(x) for x in list(np.quantile(ts, [0.10, 0.20]))]
    train_mask = torch.from_numpy(ts <= val_time)
    val_mask = torch.from_numpy((ts > val_time) & (ts <= test_time))
    test_mask = torch.from_numpy(ts > test_time)

    # 一致性：与 Dataset 内张量比较
    assert int(train_mask.sum()) == len(train_data["idx"])
    assert int(val_mask.sum()) == len(val_data["idx"])
    assert int(test_mask.sum()) == len(test_data["idx"])

    # 互斥
    assert not (train_mask & val_mask).any()
    assert not (train_mask & test_mask).any()
    assert not (val_mask & test_mask).any()

    # new-node 边掩码：与 utils.Dataset（ContraTGT / EduTGT 同源）一致
    ei_np = edge["idx"].numpy()
    train_e = ei_np[train_mask.numpy()]
    train_node_set = set(train_e[:, 0].tolist()) | set(train_e[:, 1].tolist())
    total_node_set = set(ei_np[:, 0].tolist()) | set(ei_np[:, 1].tolist())
    new_node_set = total_node_set - train_node_set
    u_col, v_col = ei_np[:, 0], ei_np[:, 1]
    is_new_node_edge = np.array(
        [(int(u_col[i]) in new_node_set or int(v_col[i]) in new_node_set) for i in range(ei_np.shape[0])],
        dtype=bool,
    )
    is_nn_t = torch.from_numpy(is_new_node_edge)
    nn_val_mask = is_nn_t & val_mask
    nn_test_mask = is_nn_t & test_mask
    assert int(nn_val_mask.sum()) == len(nn_val["idx"]), "nn_val 边数应与 Dataset 一致"
    assert int(nn_test_mask.sum()) == len(nn_test["idx"]), "nn_test 边数应与 Dataset 一致"

    # 特征（与 main.py 一致）
    features = pd.read_csv(content_path, header=None)
    feat_csr = sp.csr_matrix(features.values.astype(np.float64))
    features = normalize_features(feat_csr)
    arr = features.toarray() if sp.issparse(features) else np.asarray(features)
    if arr.shape[0] != num_nodes:
        raise ValueError(f".content 行数 {arr.shape[0]} != 图节点数 {num_nodes}")
    pad = (4 - (arr.shape[1] % 4)) % 4
    if pad:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], pad), dtype=np.float64)], axis=1)
    arr = arr.astype(np.float32)

    edge_attr_path = os.path.join(data_dir, "edge_attr.csv")
    if log_edge_attr:
        if os.path.isfile(edge_attr_path):
            print(f"[data_adapter] 检测到 edge_attr.csv，当前 baseline 模型不使用，已忽略。")
        else:
            print(f"[data_adapter] 无 edge_attr.csv，跳过。")

    perm_chrono = np.argsort(edge["idx"][:, 2].numpy())

    # ml 首列校验（可选）
    with open(ml_path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if len(header) < 6:
        print(f"[data_adapter] 警告: ml 表头列数 {len(header)}，期望至少 6 列 (含 u,i,ts,label,idx)")

    stats = {
        "num_edges": int(edge["idx"].size(0)),
        "num_train": int(train_mask.sum()),
        "num_val": int(val_mask.sum()),
        "num_test": int(test_mask.sum()),
        "num_nn_val": int(nn_val_mask.sum()),
        "num_nn_test": int(nn_test_mask.sum()),
        "val_time": val_time,
        "test_time": test_time,
        "ts_min": int(ts.min()),
        "ts_max": int(ts.max()),
    }

    return BaselineDataBundle(
        data_dir=data_dir,
        data_name=data_name,
        ml_path=ml_path,
        content_path=content_path,
        num_nodes=num_nodes,
        feat_dim=arr.shape[1],
        features=arr,
        edge_idx=edge["idx"],
        labels=edge["labels"],
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        val_time=val_time,
        test_time=test_time,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_rand_sampler=RandEdgeSampler(train_data["idx"][:, 1].numpy()),
        nn_val_mask=nn_val_mask,
        nn_test_mask=nn_test_mask,
        nn_val_data=nn_val,
        nn_test_data=nn_test,
        perm_chrono=perm_chrono,
        stats=stats,
    )


def sanity_check_nodes(edge_idx: torch.Tensor, num_nodes: int) -> None:
    mx = int(edge_idx[:, :2].max().item())
    mn = int(edge_idx[:, :2].min().item())
    if mn < 0 or mx >= num_nodes:
        raise ValueError(f"节点编号应在 [0,{num_nodes-1}]，实际 [{mn},{mx}]")
