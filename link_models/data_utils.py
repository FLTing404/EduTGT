"""
链路预测用数据加载：ml_oulad.csv + oulad.content，与 script/utils.get_data_paths 约定一致。
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_link_data_paths(data_dir, code_root):
    """解析 data_dir 得到边表、节点特征路径与 data_name。"""
    if not data_dir or not data_dir.strip():
        return None, None, None
    d = data_dir.strip()
    if os.path.isabs(d):
        data_path = d
    else:
        data_path = os.path.join(code_root, 'data', 'all_data', d)
    edges_file = os.path.join(data_path, 'ml_oulad.csv')
    feature_file = os.path.join(data_path, 'oulad.content')
    if not os.path.exists(edges_file) or not os.path.exists(feature_file):
        return None, None, None
    data_name = os.path.basename(os.path.normpath(data_path))
    return edges_file, feature_file, data_name


def load_edges_and_features(edges_file, feature_file):
    """加载边表（含 u, i, ts, label）与节点特征，节点 0-indexed。"""
    df = pd.read_csv(edges_file)
    # 列名可能是 id,u,i,ts,label,idx 或无 header
    if 'u' not in df.columns and len(df.columns) >= 5:
        df.columns = ['id', 'u', 'i', 'ts', 'label', 'idx'][:len(df.columns)]
    u = df['u'].values.astype(np.int64)
    i = df['i'].values.astype(np.int64)
    ts = df['ts'].values.astype(np.float64)
    num_nodes = int(max(u.max(), i.max()))
    # 0-indexed
    u0 = np.clip(u - 1, 0, num_nodes - 1)
    i0 = np.clip(i - 1, 0, num_nodes - 1)
    features = []
    with open(feature_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                features.append([float(x) for x in line.split(',')])
    x = np.array(features, dtype=np.float32)
    if len(x) < num_nodes:
        pad = np.zeros((num_nodes - len(x), x.shape[1]), dtype=np.float32)
        x = np.vstack([x, pad])
    elif len(x) > num_nodes:
        x = x[:num_nodes]
    return u0, i0, ts, num_nodes, x, df


def split_edges_by_time(u, i, ts, val_ratio=0.15, test_ratio=0.15, seed=42):
    """按时间划分边为 train/val/test。"""
    n = len(u)
    order = np.argsort(ts)
    u, i, ts = u[order], i[order], ts[order]
    t = ts
    val_t = np.quantile(t, 1 - val_ratio - test_ratio)
    test_t = np.quantile(t, 1 - test_ratio)
    train_mask = t <= val_t
    val_mask = (t > val_t) & (t <= test_t)
    test_mask = t > test_t
    return (
        (u[train_mask], i[train_mask], ts[train_mask]),
        (u[val_mask], i[val_mask], ts[val_mask]),
        (u[test_mask], i[test_mask], ts[test_mask]),
    )


def split_edges_by_ui(u, i, ts, val_ratio=0.1, test_ratio=0.2, seed=42):
    """按 (u,i) 划分：同一 (u,i) 只出现在一个集合，防泄露。"""
    ui = np.stack([u, i], axis=1)
    unique_ui, idx = np.unique(ui, axis=0, return_inverse=True)
    n_ui = len(unique_ui)
    np.random.seed(seed)
    perm = np.random.permutation(n_ui)
    n_val = int(n_ui * val_ratio)
    n_test = int(n_ui * test_ratio)
    n_train = n_ui - n_val - n_test
    train_ui = set(map(tuple, unique_ui[perm[:n_train]]))
    val_ui = set(map(tuple, unique_ui[perm[n_train:n_train + n_val]]))
    test_ui = set(map(tuple, unique_ui[perm[n_train + n_val:]]))
    train_mask = np.array([tuple(ui[j]) in train_ui for j in range(len(u))])
    val_mask = np.array([tuple(ui[j]) in val_ui for j in range(len(u))])
    test_mask = np.array([tuple(ui[j]) in test_ui for j in range(len(u))])
    return (
        (u[train_mask], i[train_mask], ts[train_mask]),
        (u[val_mask], i[val_mask], ts[val_mask]),
        (u[test_mask], i[test_mask], ts[test_mask]),
    )


def split_edges_by_ratio(u, i, ts, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8, seed=42):
    """
    按边随机划分，与论文 Section V-A 一致：1:1:8 => train 10%, val 10%, test 80%。
    train_ratio + val_ratio + test_ratio 应为 1.0。
    """
    n = len(u)
    np.random.seed(seed)
    perm = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return (
        (u[train_idx], i[train_idx], ts[train_idx]),
        (u[val_idx], i[val_idx], ts[val_idx]),
        (u[test_idx], i[test_idx], ts[test_idx]),
    )


def get_inductive_mask(u_train, i_train, u_test, i_test):
    """
    测试集中哪些边是 Inductive（至少一个端点在训练集中未出现）。
    u_train, i_train: 训练边端点；u_test, i_test: 测试边端点。
    返回：长度为 len(u_test) 的 bool 数组，True 表示该测试边为 inductive。
    """
    train_nodes = set(u_train.tolist()) | set(i_train.tolist())
    u_t, i_t = u_test, i_test
    return np.array([(a not in train_nodes or b not in train_nodes) for a, b in zip(u_t, i_t)])


def list_available_data(code_root):
    """列出 data/all_data 下包含 ml_oulad.csv 的子目录。"""
    all_data = os.path.join(code_root, 'data', 'all_data')
    if not os.path.exists(all_data):
        return []
    out = []
    for name in os.listdir(all_data):
        path = os.path.join(all_data, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, 'ml_oulad.csv')):
            out.append(name)
    return sorted(out)
