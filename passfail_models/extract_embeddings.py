"""
预训练表示提取：加载 EduTGT 预训练模型，对每条边前向得到 pair embedding，按 (u,i) 聚合成单向量与序列。
与 passfail_data 的 (u,i) 划分一致，严格按 train/val/test 边划分。
"""
import os
import sys
import math
import copy
import numpy as np
import torch
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(SCRIPT_DIR)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
if os.path.join(CODE_ROOT, 'script') not in sys.path:
    sys.path.insert(0, os.path.join(CODE_ROOT, 'script'))

from utils import get_data_paths, Dataset, get_edges, init_seeds, normalize_features
from sampling import get_adj_list, init_offset, get_interaction_list, get_neighbor_list, get_unique_node_sequence
from model import SpatialTemporal
import pandas as pd

from passfail_data import get_ui_splits_and_labels


def collect_edge_embeddings(data, batch_size, model, features, time_encode, time_information,
                            node_l, ts_l, idx_l, offset_l, edges, interaction_list, idx_list_sorted,
                            ctx_sample, tmp_sample, rand_sampler, device, indim):
    """对 data 中每条边前向得到 (u, i, ts, embed_vec, label)。embed_vec 为 (embed_dim,) numpy。"""
    model.eval()
    num_instance = len(data['idx'])
    num_batch = math.ceil(num_instance / batch_size)
    edge_list = []
    with torch.no_grad():
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)
            batch_data = get_edges(s_idx, e_idx, data)
            node_sum = len(batch_data['idx'])
            batch_ngh_node, batch_ngh_ts, batch_ngh_idx, batch_ngh_mask = get_neighbor_list(
                node_l, ts_l, idx_l, offset_l,
                batch_data['idx'][:, 0], batch_data['idx'][:, 2], num_sample=ctx_sample)
            to_ngh_node, to_ngh_ts, to_ngh_idx, to_ngh_mask = get_neighbor_list(
                node_l, ts_l, idx_l, offset_l,
                batch_data['idx'][:, 1], batch_data['idx'][:, 2], num_sample=ctx_sample)
            con_seq_fea = np.empty((node_sum, ctx_sample, indim))
            con_to_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(batch_ngh_node):
                for idj, j in enumerate(i):
                    con_seq_fea[idx, idj, :] = features[j]
            for idx, i in enumerate(to_ngh_node):
                for idj, j in enumerate(i):
                    con_to_fea[idx, idj, :] = features[j]
            con_seq_fea = torch.FloatTensor(np.array(con_seq_fea))
            ts_batch = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ngh_ts)
            con_seq_fea = time_information(con_seq_fea, ts_batch)
            context_feature = time_encode(con_seq_fea, torch.tensor(batch_ngh_ts)).to(device)
            batch_ngh_mask = torch.LongTensor(np.array(batch_ngh_mask)).to(device)
            con_to_fea = torch.FloatTensor(np.array(con_to_fea))
            ts_to = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ngh_ts)
            con_to_fea = time_information(con_to_fea, ts_to)
            to_con_feature = time_encode(con_to_fea, torch.tensor(to_ngh_ts)).to(device)
            to_ngh_mask = torch.LongTensor(np.array(to_ngh_mask)).to(device)
            batch_node_seq, batch_node_seq_mask, batch_ts = get_unique_node_sequence(
                batch_data, edges, tmp_sample, interaction_list, flag=True, idx_list_sorted=idx_list_sorted)
            temp_seq_fea = np.empty((node_sum, tmp_sample, indim))
            for idx, i in enumerate(batch_node_seq):
                for idj, j in enumerate(i):
                    temp_seq_fea[idx, idj, :] = features[j]
            temp_seq_fea = torch.FloatTensor(np.array(temp_seq_fea))
            ts_tmp = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ts)
            temp_seq_fea = time_information(temp_seq_fea, ts_tmp)
            temporal_feature = time_encode(temp_seq_fea, torch.tensor(batch_ts)).to(device)
            batch_node_seq_mask = torch.LongTensor(np.array(batch_node_seq_mask)).to(device)
            to_node_seq, to_node_seq_mask, to_ts = get_unique_node_sequence(
                batch_data, edges, tmp_sample, interaction_list, flag=False, idx_list_sorted=idx_list_sorted)
            to_seq_fea = np.empty((node_sum, tmp_sample, indim))
            for idx, i in enumerate(to_node_seq):
                for idj, j in enumerate(i):
                    to_seq_fea[idx, idj, :] = features[j]
            to_seq_fea = torch.FloatTensor(to_seq_fea)
            ts_to_seq = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ts)
            to_seq_fea = time_information(to_seq_fea, ts_to_seq)
            to_seq_feature = time_encode(to_seq_fea, torch.tensor(to_ts)).to(device)
            to_node_seq_mask = torch.LongTensor(np.array(to_node_seq_mask)).to(device)

            src_embed = model.getEmbed(context_feature, temporal_feature, batch_ngh_mask, batch_node_seq_mask)
            tgt_embed = model.getEmbed(to_con_feature, to_seq_feature, to_ngh_mask, to_node_seq_mask)
            pos_embed = model.norm(model.linear(torch.cat([src_embed, tgt_embed], dim=1)))
            pos_embed_np = pos_embed.cpu().numpy()

            u_np = batch_data['idx'][:, 0].cpu().numpy()
            i_np = batch_data['idx'][:, 1].cpu().numpy()
            ts_np = batch_data['idx'][:, 2].cpu().numpy().astype(np.float64)
            lab = batch_data['labels'].cpu().numpy()
            for j in range(node_sum):
                edge_list.append((int(u_np[j]), int(i_np[j]), float(ts_np[j]), pos_embed_np[j].copy(), int(lab[j] == 1)))
    return edge_list


def aggregate_ui_time_weighted(edge_list):
    """按 (u,i) 时间加权平均 embedding，得到课程级向量与 label。与 main_passfail 的 aggregate_course_level_time_weighted 一致。"""
    groups = defaultdict(list)
    for (u, i, ts, vec, label) in edge_list:
        groups[(u, i)].append((float(ts), vec, label))
    scores_vec = []
    labels = []
    for key in sorted(groups.keys()):
        arr = groups[key]
        w_sum = sum(ts for ts, _, _ in arr)
        vecs = np.array([v for _, v, _ in arr])
        ts_arr = np.array([ts for ts, _, _ in arr])
        if w_sum > 0:
            weights = ts_arr / w_sum
            s_ui = np.average(vecs, axis=0, weights=weights)
        else:
            s_ui = np.mean(vecs, axis=0)
        scores_vec.append(s_ui)
        labels.append(arr[0][2])
    return np.array(scores_vec, dtype=np.float32), np.array(labels, dtype=np.int64)


def aggregate_ui_sequences(edge_list, seq_max_len, embed_dim):
    """按 (u,i) 收集边 embedding 序列（按 ts 排序），padding 到 seq_max_len。"""
    groups = defaultdict(list)
    for (u, i, ts, vec, label) in edge_list:
        groups[(u, i)].append((float(ts), vec, label))
    X_seqs = []
    lengths = []
    for key in sorted(groups.keys()):
        arr = sorted(groups[key], key=lambda x: x[0])
        arr = arr[:seq_max_len]
        vecs = np.array([v for _, v, _ in arr], dtype=np.float32)
        lengths.append(len(vecs))
        if len(vecs) < seq_max_len:
            pad = np.zeros((seq_max_len - len(vecs), embed_dim), dtype=np.float32)
            vecs = np.vstack([vecs, pad]) if len(vecs) > 0 else pad
        else:
            vecs = vecs[:seq_max_len]
        X_seqs.append(vecs)
    return X_seqs, lengths


def run_extract(edges_file, feature_file, pretrain_path, train_data, val_data, test_data,
                edge, train_ui, val_ui, test_ui, ctx_sample=40, tmp_sample=31, batch_size=256,
                device=None, seed=60, seq_max_len=64):
    """
    对 train/val/test 边分别提取 embedding，再按 (u,i) 聚合成单向量与序列，与 train_ui/val_ui/test_ui 顺序一致。
    返回 dict: X_train, X_val, X_test (单向量), X_train_seq, X_val_seq, X_test_seq, seq_lengths_*, embed_dim
    """
    init_seeds(seed)
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features = pd.read_csv(feature_file, header=None)
    features = normalize_features(features)
    if hasattr(features, 'values'):
        features = features.values
    features = np.array(features, dtype=np.float32)
    fea_dim = features.shape[1]
    indim, outdim = fea_dim, 128
    nheads, dropout, N = 4, 0.2, 2
    from utils import TimeEncode
    time_encode = TimeEncode(time_dim=fea_dim, time_encoding='concat')
    time_information = TimeEncode(time_dim=fea_dim, time_encoding='sum')
    adj_list = get_adj_list(edge)
    node_l, ts_l, idx_l, offset_l = init_offset(adj_list)
    interaction_list, idx_list_sorted = get_interaction_list(edge)

    model = SpatialTemporal(in_dim=indim, out_dim=outdim, n_heads=nheads, dropout=dropout, N=N)
    model.load_state_dict(torch.load(pretrain_path, map_location=device))
    model = model.to(device)
    model.eval()
    embed_dim = outdim

    from utils import RandEdgeSampler
    train_rand_sampler = RandEdgeSampler(edge['idx'][:, 1].numpy())

    def run_data(data):
        return collect_edge_embeddings(
            data, batch_size, model, features, time_encode, time_information,
            node_l, ts_l, idx_l, offset_l, edge, interaction_list, idx_list_sorted,
            ctx_sample, tmp_sample, train_rand_sampler, device, indim)

    el_train = run_data(train_data)
    el_val = run_data(val_data)
    el_test = run_data(test_data)

    def to_ui_ordered(edge_list, ui_list):
        key_to_idx = {k: i for i, k in enumerate(sorted(set((u, i) for u, i, _, _, _ in edge_list)))}
        key_to_vec = {}
        key_to_seq = defaultdict(list)
        for (u, i, ts, vec, lab) in edge_list:
            key = (u, i)
            if key not in key_to_vec:
                key_to_vec[key] = []
            key_to_vec[key].append((ts, vec, lab))
        single = []
        for (u, i) in ui_list:
            arr = key_to_vec.get((u, i), [])
            if not arr:
                single.append(np.zeros(embed_dim, dtype=np.float32))
                continue
            w_sum = sum(ts for ts, _, _ in arr)
            vecs = np.array([v for _, v, _ in arr])
            ts_arr = np.array([ts for ts, _, _ in arr])
            if w_sum > 0:
                weights = ts_arr / w_sum
                single.append(np.average(vecs, axis=0, weights=weights).astype(np.float32))
            else:
                single.append(np.mean(vecs, axis=0).astype(np.float32))
        return np.array(single)

    X_train = to_ui_ordered(el_train, train_ui)
    X_val = to_ui_ordered(el_val, val_ui)
    X_test = to_ui_ordered(el_test, test_ui)

    def to_seq_ordered(edge_list, ui_list, seq_max_len):
        groups = defaultdict(list)
        for (u, i, ts, vec, _) in edge_list:
            groups[(u, i)].append((float(ts), vec))
        X_seqs = []
        lengths = []
        for (u, i) in ui_list:
            arr = sorted(groups.get((u, i), []), key=lambda x: x[0])[:seq_max_len]
            vecs = np.array([v for _, v in arr], dtype=np.float32)
            lengths.append(len(vecs))
            if len(vecs) < seq_max_len:
                pad = np.zeros((seq_max_len - len(vecs), embed_dim), dtype=np.float32)
                vecs = np.vstack([vecs, pad]) if len(vecs) > 0 else pad
            else:
                vecs = vecs[:seq_max_len]
            X_seqs.append(vecs)
        return X_seqs, lengths

    X_train_seq, sl_train = to_seq_ordered(el_train, train_ui, seq_max_len)
    X_val_seq, sl_val = to_seq_ordered(el_val, val_ui, seq_max_len)
    X_test_seq, sl_test = to_seq_ordered(el_test, test_ui, seq_max_len)

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'X_train_seq': X_train_seq, 'X_val_seq': X_val_seq, 'X_test_seq': X_test_seq,
        'seq_lengths_train': sl_train, 'seq_lengths_val': sl_val, 'seq_lengths_test': sl_test,
        'seq_max_len': seq_max_len, 'embed_dim': embed_dim,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_abc_0.01')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ablation_suffix', type=str, default='', help='消融时与 pretrain 命名一致，如 baseline；配合 --seed 加载 pretrain_model/<data>_<suffix>_seed<seed>.pth')
    parser.add_argument('--seq_max_len', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    class A:
        data_dir = args.data_dir
        gpu = args.gpu
    edges_file, feature_file, data_name = get_data_paths(A)
    _ablation_sfx = getattr(args, 'ablation_suffix', '') or ''
    _seed_sfx = ('_seed' + str(args.seed)) if _ablation_sfx else ''
    pretrain_path = os.path.join(CODE_ROOT, 'script', 'pretrain_model', f'{data_name}{"_" + _ablation_sfx if _ablation_sfx else ""}{_seed_sfx}.pth')
    train_ui, val_ui, test_ui, y_train, y_val, y_test, edge = get_ui_splits_and_labels(
        edges_file, random_state=args.seed)
    edge_ds, _, _, _, train_data, test_data, val_data, _, _ = Dataset(
        file=edges_file, split_by_ui=False, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8, random_state=args.seed)
    out = run_extract(edges_file, feature_file, pretrain_path, train_data, val_data, test_data,
                      edge_ds, train_ui, val_ui, test_ui, seq_max_len=args.seq_max_len, seed=args.seed,
                      device=torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'))
    print('X_train', out['X_train'].shape, 'embed_dim', out['embed_dim'])
    out_dir = os.path.join(CODE_ROOT, 'result', 'passfail', data_name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'edutgt_emb_{data_name}_s{args.seed}.npz'),
             X_train=out['X_train'], X_val=out['X_val'], X_test=out['X_test'],
             X_train_seq=np.array(out['X_train_seq']), X_val_seq=np.array(out['X_val_seq']), X_test_seq=np.array(out['X_test_seq']),
             seq_lengths_train=out['seq_lengths_train'], seq_lengths_val=out['seq_lengths_val'], seq_lengths_test=out['seq_lengths_test'])
