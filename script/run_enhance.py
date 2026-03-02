"""
一键：ContraTGT 微调 + 生成 embedding + 拼表 + 输出增强数据到 process_data/<data_dir>_enhanced/。
主表为 ml_oulad.csv，新增列 [h_u || h_i]；其余三文件原样复制。严格按边时间顺序生成 embedding，避免时间泄漏。
"""
from __future__ import print_function

import os
import sys
import argparse
import subprocess
import shutil
import math
import copy
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# script dir and code dir
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils import (
    get_code_dir,
    get_data_paths,
    Dataset,
    get_edges,
    normalize_features,
    init_seeds,
    TimeEncode,
)
from sampling import (
    get_adj_list,
    init_offset,
    get_interaction_list,
    get_neighbor_list,
    get_unique_node_sequence,
)
from model import SpatialTemporal


def _get_data_paths_for_enhance(data_dir, all_data_root=None):
    class Args(object):
        pass
    args = Args()
    args.data_dir = data_dir
    args.all_data_root = all_data_root
    args.data = 'oulad'
    return get_data_paths(args)


def run_pretrain_and_main(data_dir, all_data_root, gpu, code_dir):
    env = os.environ.copy()
    cmd_base = [sys.executable, os.path.join(code_dir, 'script', 'pretrain.py'), '--data_dir', data_dir, '--gpu', str(gpu)]
    if all_data_root:
        cmd_base.extend(['--all_data_root', all_data_root])
    print('Running pretrain: ', ' '.join(cmd_base))
    ret = subprocess.run(cmd_base, cwd=code_dir, env=env)
    if ret.returncode != 0:
        raise RuntimeError('pretrain.py exited with code {}'.format(ret.returncode))

    cmd_main = [sys.executable, os.path.join(code_dir, 'script', 'main.py'), '--data_dir', data_dir, '--gpu', str(gpu)]
    if all_data_root:
        cmd_main.extend(['--all_data_root', all_data_root])
    print('Running main (finetune): ', ' '.join(cmd_main))
    ret = subprocess.run(cmd_main, cwd=code_dir, env=env)
    if ret.returncode != 0:
        raise RuntimeError('main.py exited with code {}'.format(ret.returncode))


def export_embeddings_and_save(
    data_dir,
    out_dir,
    csv_path,
    content_path,
    checkpoint_path,
    all_data_root,
    process_data_root,
    gpu,
    batch_size=800,
    ctx_sample=40,
    tmp_sample=31,
):
    code_dir = get_code_dir()
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

    # Load CSV (keep original column order)
    df = pd.read_csv(csv_path)
    n_rows = len(df)

    # Graph and features (same as main)
    edges, num_nodes, nodes_list, node_time, train_data, test_data, val_data, nn_test_data, nn_val_data = Dataset(file=csv_path)
    adj_list = get_adj_list(edges)
    node_l, ts_l, idx_l, offset_l = init_offset(adj_list)
    interaction_list = get_interaction_list(edges)

    features = pd.read_csv(content_path, header=None)
    features = normalize_features(features)
    # normalize_features 可能返回 ndarray 或 sparse matrix，统一转为 tensor
    if hasattr(features, 'toarray'):
        features = features.toarray()
    elif hasattr(features, 'values'):
        features = features.values
    features = torch.tensor(np.asarray(features), dtype=torch.float32)
    fea_dim = features.shape[1]
    time_encode = TimeEncode(time_dim=fea_dim, time_encoding='concat')
    time_information = TimeEncode(time_dim=fea_dim, time_encoding='sum')

    indim = fea_dim
    outdim = 128
    nheads = 4
    dropout = 0.2
    N = 2

    # Model and checkpoint
    model = SpatialTemporal(in_dim=indim, out_dim=outdim, n_heads=nheads, dropout=dropout, N=N)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # Iterate over ALL edges in CSV order (no time leakage: same logic as main)
    edge_data = edges
    num_instance = len(edge_data['idx'])
    num_batch = math.ceil(num_instance / batch_size)
    init_seeds(60)

    emb_list = []
    with torch.no_grad():
        for k in tqdm(range(num_batch), desc='Export embeddings', unit='batch'):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)
            batch_data = get_edges(s_idx, e_idx, edge_data)
            node_sum = len(batch_data['idx'])

            # src (u) spatial
            batch_ngh_node, batch_ngh_ts, batch_ngh_idx, batch_ngh_mask = get_neighbor_list(
                node_l, ts_l, idx_l, offset_l,
                batch_data['idx'][:, 0], batch_data['idx'][:, 2], num_sample=ctx_sample
            )
            con_seq_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(batch_ngh_node):
                for idj, j in enumerate(i):
                    con_seq_fea[idx, idj, :] = features[j].numpy()
            con_seq_fea = torch.FloatTensor(con_seq_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ngh_ts)
            con_seq_fea = time_information(con_seq_fea, ts)
            context_feature = time_encode(con_seq_fea, torch.tensor(batch_ngh_ts)).to(device)
            batch_ngh_mask = torch.LongTensor(np.array(batch_ngh_mask)).to(device)

            # src temporal
            batch_node_seq, batch_node_seq_mask, batch_ts = get_unique_node_sequence(
                batch_data, edges, tmp_sample, interaction_list, flag=True
            )
            temp_seq_fea = np.empty((node_sum, tmp_sample, indim))
            for idx, i in enumerate(batch_node_seq):
                for idj, j in enumerate(i):
                    temp_seq_fea[idx, idj, :] = features[j].numpy()
            temp_seq_fea = torch.FloatTensor(temp_seq_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ts)
            temp_seq_fea = time_information(temp_seq_fea, ts)
            temporal_feature = time_encode(temp_seq_fea, torch.tensor(batch_ts)).to(device)
            batch_node_seq_mask = torch.LongTensor(np.array(batch_node_seq_mask)).to(device)

            h_u = model.getEmbed(context_feature, temporal_feature, batch_ngh_mask, batch_node_seq_mask)

            # tgt (i) spatial
            to_ngh_node, to_ngh_ts, to_ngh_idx, to_ngh_mask = get_neighbor_list(
                node_l, ts_l, idx_l, offset_l,
                batch_data['idx'][:, 1], batch_data['idx'][:, 2], num_sample=ctx_sample
            )
            con_to_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(to_ngh_node):
                for idj, j in enumerate(i):
                    con_to_fea[idx, idj, :] = features[j].numpy()
            con_to_fea = torch.FloatTensor(con_to_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ngh_ts)
            con_to_fea = time_information(con_to_fea, ts)
            to_con_feature = time_encode(con_to_fea, torch.tensor(to_ngh_ts)).to(device)
            to_ngh_mask = torch.LongTensor(np.array(to_ngh_mask)).to(device)

            # tgt temporal
            to_node_seq, to_node_seq_mask, to_ts = get_unique_node_sequence(
                batch_data, edges, tmp_sample, interaction_list, flag=False
            )
            to_seq_fea = np.empty((node_sum, tmp_sample, indim))
            for idx, i in enumerate(to_node_seq):
                for idj, j in enumerate(i):
                    to_seq_fea[idx, idj, :] = features[j].numpy()
            to_seq_fea = torch.FloatTensor(to_seq_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ts)
            to_seq_fea = time_information(to_seq_fea, ts)
            to_seq_feature = time_encode(to_seq_fea, torch.tensor(to_ts)).to(device)
            to_node_seq_mask = torch.LongTensor(np.array(to_node_seq_mask)).to(device)

            h_i = model.getEmbed(to_con_feature, to_seq_feature, to_ngh_mask, to_node_seq_mask)

            # [h_u || h_i]
            h_u_np = h_u.cpu().numpy()
            h_i_np = h_i.cpu().numpy()
            batch_emb = np.concatenate([h_u_np, h_i_np], axis=1)
            emb_list.append(batch_emb)

    emb_all = np.concatenate(emb_list, axis=0)
    d = indim
    col_u = ['emb_u_{}'.format(i) for i in range(d)]
    col_i = ['emb_i_{}'.format(i) for i in range(d)]
    for i, c in enumerate(col_u):
        df[c] = emb_all[:, i]
    for i, c in enumerate(col_i):
        df[c] = emb_all[:, d + i]

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'ml_oulad.csv')
    df.to_csv(out_csv, index=False)

    # Copy other three files (option 3-C)
    data_dir_path = os.path.dirname(csv_path)
    for name in ['oulad.content', 'ml_oulad_edge_feature.csv', 'ml_oulad_pairs.csv']:
        src = os.path.join(data_dir_path, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))
        else:
            print('Warning: not found, skip copy: {}'.format(src))

    return out_csv, n_rows, d, df.columns.tolist()


def self_check(original_csv_path, out_csv_path, n_rows_expected, d, expected_col_order):
    errors = []
    df_out = pd.read_csv(out_csv_path)
    if len(df_out) != n_rows_expected:
        errors.append('Row count mismatch: got {} expected {}'.format(len(df_out), n_rows_expected))
    orig_cols = ['id', 'u', 'i', 'ts', 'label', 'idx']
    actual = df_out.columns.tolist()
    if not actual[:len(orig_cols)] == orig_cols:
        errors.append('First columns order mismatch: expected {} got {}'.format(orig_cols, actual[:len(orig_cols)]))
    emb_cols = ['emb_u_{}'.format(i) for i in range(d)] + ['emb_i_{}'.format(i) for i in range(d)]
    if not actual[len(orig_cols):len(orig_cols) + 2 * d] == emb_cols:
        errors.append('Embedding column names/order mismatch')
    nan_count = df_out[emb_cols].isna().sum().sum()
    if nan_count > 0:
        errors.append('Found {} NaN in embedding columns'.format(nan_count))
    if errors:
        for e in errors:
            print('Self-check FAILED:', e)
        return False
    print('Self-check passed: rows={}, cols order ok, no NaN in embeddings'.format(len(df_out)))
    return True


def main():
    parser = argparse.ArgumentParser(description='ContraTGT: one-click finetune + export enhanced table to process_data/')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory name under all_data/ (e.g. data_0.1)')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory; default process_data/<data_dir>_enhanced')
    parser.add_argument('--all_data_root', type=str, default=None, help='Root of all_data; default code/all_data')
    parser.add_argument('--process_data_root', type=str, default=None, help='Root of process_data; default code/process_data')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip_finetune', action='store_true', help='Only export (assume checkpoint already exists)')
    args = parser.parse_args()

    code_dir = get_code_dir()
    all_data_root = args.all_data_root or os.path.join(code_dir, 'all_data')
    process_data_root = args.process_data_root or os.path.join(code_dir, 'process_data')
    out_dir = args.out_dir or os.path.join(process_data_root, '{}_enhanced'.format(args.data_dir))

    csv_path, content_path, data_name = _get_data_paths_for_enhance(args.data_dir, all_data_root)
    if not os.path.isfile(csv_path):
        print('Error: CSV not found:', csv_path)
        sys.exit(1)
    if not os.path.isfile(content_path):
        print('Error: content not found:', content_path)
        sys.exit(1)

    if not args.skip_finetune:
        run_pretrain_and_main(args.data_dir, args.all_data_root, args.gpu, code_dir)

    checkpoint_path = os.path.join(code_dir, 'saved_checkpoints', '{}.pth'.format(data_name))
    if not os.path.isfile(checkpoint_path):
        print('Error: checkpoint not found:', checkpoint_path)
        sys.exit(1)

    print('Exporting embeddings and writing to', out_dir)
    out_csv, n_rows, d, col_order = export_embeddings_and_save(
        data_dir=args.data_dir,
        out_dir=out_dir,
        csv_path=csv_path,
        content_path=content_path,
        checkpoint_path=checkpoint_path,
        all_data_root=args.all_data_root,
        process_data_root=process_data_root,
        gpu=args.gpu,
    )
    ok = self_check(csv_path, out_csv, n_rows, d, col_order)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
