"""
课程通过/不通过：统一 (u,i) 划分、标签、原始特征（单向量 + 序列）。
与 main_passfail 一致：按边 1:1:8 划分，每个 (u,i) 归属到其「首条边（按时间）」所在集合。
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(SCRIPT_DIR)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
if os.path.join(CODE_ROOT, 'script') not in sys.path:
    sys.path.insert(0, os.path.join(CODE_ROOT, 'script'))

from utils import Dataset, normalize_features


def get_ui_splits_and_labels(edges_file, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8, random_state=42):
    """
    从边级划分得到 (u,i) 级划分与标签。
    规则：每个 (u,i) 归属到其「时间最早的那条边」所在的集合（train/val/test）。
    返回：
        train_ui, val_ui, test_ui: list of (u,int) 元组，顺序稳定
        y_train, y_val, y_test: 一维 np.array，0/1 标签
        edge: 完整边 dict（edge['idx'], edge['labels']），供后续建特征用
    """
    edge, _, _, _, train_data, test_data, val_data, _, _ = Dataset(
        file=edges_file, split_by_ui=False,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, random_state=random_state)
    idx = edge['idx']  # (N, 4): u, i, ts, idx_col
    labels = edge['labels']

    train_set = set()
    for i in range(train_data['idx'].size(0)):
        row = train_data['idx'][i]
        train_set.add((int(row[0]), int(row[1]), int(row[2])))
    val_set = set()
    for i in range(val_data['idx'].size(0)):
        row = val_data['idx'][i]
        val_set.add((int(row[0]), int(row[1]), int(row[2])))
    test_set = set()
    for i in range(test_data['idx'].size(0)):
        row = test_data['idx'][i]
        test_set.add((int(row[0]), int(row[1]), int(row[2])))

    # (u,i) -> 最早 ts, label
    ui_first_ts = {}
    ui_label = {}
    for i in range(edge['idx'].size(0)):
        u, i_node, ts = int(edge['idx'][i, 0]), int(edge['idx'][i, 1]), int(edge['idx'][i, 2])
        lab = int(edge['labels'][i].item())
        key = (u, i_node)
        if key not in ui_first_ts or ts < ui_first_ts[key]:
            ui_first_ts[key] = ts
            ui_label[key] = lab

    train_ui, val_ui, test_ui = [], [], []
    for (u, i_node), first_ts in ui_first_ts.items():
        tup = (u, i_node, first_ts)
        if tup in train_set:
            train_ui.append((u, i_node))
        elif tup in val_set:
            val_ui.append((u, i_node))
        else:
            test_ui.append((u, i_node))

    y_train = np.array([ui_label[k] for k in train_ui], dtype=np.int64)
    y_val = np.array([ui_label[k] for k in val_ui], dtype=np.int64)
    y_test = np.array([ui_label[k] for k in test_ui], dtype=np.int64)

    return train_ui, val_ui, test_ui, y_train, y_val, y_test, edge


def _ui_edge_stats(edge):
    """(u,i) -> (mean_ts, max_ts, count, list of (ts, label) for sequence)."""
    idx = edge['idx']
    lab = edge['labels']
    from collections import defaultdict
    ui_edges = defaultdict(list)
    for i in range(idx.size(0)):
        u, i_node = int(idx[i, 0]), int(idx[i, 1])
        ts = float(idx[i, 2])
        ui_edges[(u, i_node)].append((ts, int(lab[i].item())))
    out = {}
    for k, arr in ui_edges.items():
        arr = sorted(arr, key=lambda x: x[0])
        ts_list = [x[0] for x in arr]
        out[k] = {
            'mean_ts': np.mean(ts_list),
            'max_ts': np.max(ts_list),
            'count': len(ts_list),
            'seq': arr,
        }
    return out


def build_raw_features_ui(edge, features, train_ui, val_ui, test_ui, seq_max_len=64):
    """
    原始特征：单向量 per (u,i) + 序列 per (u,i)。
    features: 已归一化的节点特征矩阵，行数 = num_nodes，列数 = feat_dim；节点 id 与 edge['idx'] 一致（0-based）。
    返回:
        X_train, X_val, X_test: (n, feat_dim) 单向量，feat_dim = 2*node_dim + 3（mean_ts, max_ts, count）
        X_train_seq, X_val_seq, X_test_seq: list of (seq_len, feat_dim_seq)，每帧 [feat_u, feat_i, ts]
        seq_lengths_train, seq_lengths_val, seq_lengths_test: 实际长度（padding 前）
    """
    feat_dim_node = features.shape[1]
    ui_stats = _ui_edge_stats(edge)

    def one_vector(ui_list):
        X = []
        for (u, i_node) in ui_list:
            st = ui_stats.get((u, i_node), {'mean_ts': 0, 'max_ts': 0, 'count': 0})
            f_u = features[u] if u < features.shape[0] else np.zeros(feat_dim_node)
            f_i = features[i_node] if i_node < features.shape[0] else np.zeros(feat_dim_node)
            vec = np.concatenate([f_u, f_i, [st['mean_ts'], st['max_ts'], float(st['count'])]])
            X.append(vec)
        return np.array(X, dtype=np.float32)

    def seq_vectors(ui_list):
        X_seqs = []
        lengths = []
        for (u, i_node) in ui_list:
            st = ui_stats.get((u, i_node), {'seq': []})
            arr = st['seq'][:seq_max_len]
            f_u = features[u] if u < features.shape[0] else np.zeros(feat_dim_node)
            f_i = features[i_node] if i_node < features.shape[0] else np.zeros(feat_dim_node)
            frames = []
            for ts, _ in arr:
                frames.append(np.concatenate([f_u, f_i, [ts]]))
            lengths.append(len(frames))
            if len(frames) < seq_max_len:
                pad = np.zeros((seq_max_len - len(frames), feat_dim_node * 2 + 1), dtype=np.float32)
                frames = np.vstack(frames).astype(np.float32) if frames else np.zeros((0, feat_dim_node * 2 + 1), dtype=np.float32)
                frames = np.vstack([frames, pad]) if len(frames) > 0 else pad
            else:
                frames = np.array(frames[:seq_max_len], dtype=np.float32)
            X_seqs.append(frames)
        return X_seqs, lengths

    X_train = one_vector(train_ui)
    X_val = one_vector(val_ui)
    X_test = one_vector(test_ui)
    X_train_seq, seq_lengths_train = seq_vectors(train_ui)
    X_val_seq, seq_lengths_val = seq_vectors(val_ui)
    X_test_seq, seq_lengths_test = seq_vectors(test_ui)

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'X_train_seq': X_train_seq, 'X_val_seq': X_val_seq, 'X_test_seq': X_test_seq,
        'seq_lengths_train': seq_lengths_train, 'seq_lengths_val': seq_lengths_val, 'seq_lengths_test': seq_lengths_test,
        'seq_max_len': seq_max_len,
        'feat_dim': X_train.shape[1],
        'feat_dim_seq': feat_dim_node * 2 + 1,
    }


def load_features_and_build_raw(feature_file, edge, train_ui, val_ui, test_ui, seq_max_len=64):
    """加载节点特征并构建原始特征（单向量 + 序列）。"""
    features = pd.read_csv(feature_file, header=None)
    features = normalize_features(features)
    if hasattr(features, 'values'):
        features = features.values
    features = np.array(features, dtype=np.float32)
    return build_raw_features_ui(edge, features, train_ui, val_ui, test_ui, seq_max_len=seq_max_len)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_abc_0.01')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    from utils import get_data_paths
    edges_file, feature_file, data_name = get_data_paths(args)
    train_ui, val_ui, test_ui, y_train, y_val, y_test, edge = get_ui_splits_and_labels(
        edges_file, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8, random_state=args.seed)
    print('train_ui', len(train_ui), 'val_ui', len(val_ui), 'test_ui', len(test_ui))
    raw = load_features_and_build_raw(feature_file, edge, train_ui, val_ui, test_ui)
    print('X_train shape', raw['X_train'].shape, 'X_train_seq len', len(raw['X_train_seq']))
