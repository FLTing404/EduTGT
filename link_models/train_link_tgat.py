"""
TGAT (Temporal Graph Attention) 链路预测基线。
论文: Xu et al., Inductive Representation Learning on Temporal Graphs, ICLR 2020.
简化实现：时序编码 + 邻居注意力聚合 + 链接解码器。
从项目根运行: python link_models/train_link_tgat.py --data_dir data_abc_0.01
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from collections import defaultdict

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from link_models.data_utils import (
    get_link_data_paths,
    load_edges_and_features,
    split_edges_by_time,
    split_edges_by_ui,
    split_edges_by_ratio,
    get_inductive_mask,
    list_available_data,
)


class TimeEncode(nn.Module):
    """Bochner 风格时间编码: cos(omega * t + b)。"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.omega = nn.Parameter(torch.randn(dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        # t: (N,), (N, 1) 或 (B, K)；输出 (..., time_dim)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.unsqueeze(-1)  # (..., 1)
        omega = self.omega.view(1, 1, -1)
        b = self.b.view(1, 1, -1)
        return torch.cos(t * omega + b)


class TGATLayer(nn.Module):
    """单层时序图注意力：对每个节点聚合其时序邻居（带时间编码）的加权特征。"""
    def __init__(self, in_dim, time_dim, out_dim, n_heads=2, dropout=0.1, time_enc=None):
        super().__init__()
        self.in_dim = in_dim
        self.time_dim = time_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = max(1, out_dim // n_heads)
        self.time_enc = time_enc or TimeEncode(time_dim)
        # query 来自中心节点 x (in_dim)，key/value 来自邻居 [feat, time_enc] (in_dim+time_dim)
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim + time_dim, out_dim)
        self.W_v = nn.Linear(in_dim + time_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ngh_feat, ngh_dt, mask):
        """
        x: (B, in_dim) 目标节点特征
        ngh_feat: (B, K, in_dim) 邻居特征
        ngh_dt: (B, K) 邻居相对时间
        mask: (B, K) 1=有效 0=pad
        """
        B, K, _ = ngh_feat.shape
        time_enc = self.time_enc(ngh_dt)  # (B, K, time_dim)
        ngh = torch.cat([ngh_feat, time_enc], dim=-1)  # (B, K, in_dim+time_dim)
        q = self.W_q(x).unsqueeze(1)  # (B, 1, out_dim)
        k = self.W_k(ngh)  # (B, K, out_dim)
        v = self.W_v(ngh)
        scores = (q * k).sum(-1) / (self.head_dim ** 0.5)  # (B, K)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=1)
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * v).sum(1)  # (B, out_dim)
        return out


class TGATBlock(nn.Module):
    def __init__(self, in_dim, time_dim, out_dim, n_heads=2, dropout=0.1):
        super().__init__()
        self.time_enc = TimeEncode(time_dim)
        self.layer = TGATLayer(in_dim, time_dim, out_dim, n_heads, dropout, time_enc=self.time_enc)

    def forward(self, x, ngh_feat, ngh_dt, mask):
        return self.layer(x, ngh_feat, ngh_dt, mask)


class TGATLinkPredictor(nn.Module):
    """TGAT 风格：节点特征 + 时序邻居聚合（单层）-> 节点嵌入 -> MLP 链接预测。"""
    def __init__(self, in_dim, time_dim=32, hidden_dim=128, out_dim=64, dropout=0.2, k_ngh=10):
        super().__init__()
        self.k_ngh = k_ngh
        self.tgat = TGATBlock(in_dim, time_dim, hidden_dim, n_heads=2, dropout=dropout)
        self.self_proj = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, src_self, dst_self, ngh_src, ngh_src_t, mask_src, ngh_dst, ngh_dst_t, mask_dst):
        # src_self, dst_self: (B, in_dim); ngh_*: (B, K, in_dim), (B, K), (B, K)
        h_src = self.tgat(src_self, ngh_src, ngh_src_t, mask_src) + self.self_proj(src_self)
        h_dst = self.tgat(dst_self, ngh_dst, ngh_dst_t, mask_dst) + self.self_proj(dst_self)
        z = torch.cat([h_src, h_dst], dim=-1)
        return self.decoder(z).squeeze(-1)


def build_temporal_ngh(u, i, ts, num_nodes, k=10):
    """对每条边 (u,i,ts)，取 u 和 i 各自在 ts 之前的 k 条最近邻（按时间）。"""
    # 按时间排序的边列表
    order = np.argsort(ts)
    u, i, ts = u[order], i[order], ts[order]
    # 每个节点的时间序邻居: node_id -> [(ngh, t), ...]
    ngh_u = defaultdict(list)
    ngh_i = defaultdict(list)
    for idx in range(len(u)):
        uu, ii, t = u[idx], i[idx], ts[idx]
        ngh_i[uu].append((ii, t))
        if len(ngh_i[uu]) > k:
            ngh_i[uu].sort(key=lambda x: -x[1])
            ngh_i[uu] = ngh_i[uu][:k]
        ngh_u[ii].append((uu, t))
        if len(ngh_u[ii]) > k:
            ngh_u[ii].sort(key=lambda x: -x[1])
            ngh_u[ii] = ngh_u[ii][:k]
    return ngh_u, ngh_i, u, i, ts


def get_ngh_tensors(ngh_dict, nodes, t_batch, x, k, device):
    """nodes: (B,), t_batch: (B,) -> ngh_feat (B,k,feat_dim), ngh_dt (B,k), mask (B,k)"""
    B = len(nodes)
    feat_dim = x.shape[1]
    ngh_feat = np.zeros((B, k, feat_dim), dtype=np.float32)
    ngh_dt = np.zeros((B, k), dtype=np.float32)
    mask = np.zeros((B, k), dtype=np.float32)
    for b in range(B):
        n = nodes[b]
        t = t_batch[b]
        lst = ngh_dict.get(n, [])[:k]
        for j, (ngh_id, ngh_t) in enumerate(lst):
            if ngh_t <= t:
                ngh_feat[b, j] = x[ngh_id]
                ngh_dt[b, j] = t - ngh_t
                mask[b, j] = 1
        if len(lst) == 0:
            ngh_feat[b, 0] = x[n]
            mask[b, 0] = 1
    return (
        torch.from_numpy(ngh_feat).to(device),
        torch.from_numpy(ngh_dt).float().to(device),
        torch.from_numpy(mask).float().to(device),
    )


def run_tgat(edges_file, feature_file, data_name, code_root, split_by_ui=False,
             paper_eval=False, train_ratio=None, val_ratio=0.15, test_ratio=0.2, seed=42, device=None,
             hidden_dim=128, out_dim=64, time_dim=32, k_ngh=10, dropout=0.2,
             epochs=80, lr=1e-3, batch_size=512):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    u, i, ts, num_nodes, x_np, _ = load_edges_and_features(edges_file, feature_file)
    if paper_eval or (train_ratio is not None):
        tr, val, te = (0.1, 0.1, 0.8) if paper_eval else (train_ratio, val_ratio, test_ratio)
        (u_tr, i_tr, t_tr), (u_val, i_val, t_val), (u_te, i_te, t_te) = split_edges_by_ratio(
            u, i, ts, train_ratio=tr, val_ratio=val, test_ratio=te, seed=seed
        )
    elif split_by_ui:
        (u_tr, i_tr, t_tr), (u_val, i_val, t_val), (u_te, i_te, t_te) = split_edges_by_ui(
            u, i, ts, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
    else:
        (u_tr, i_tr, t_tr), (u_val, i_val, t_val), (u_te, i_te, t_te) = split_edges_by_time(
            u, i, ts, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )

    # 用时序训练边构建邻居表（只取 t 之前的边）
    ngh_u, ngh_i, u_all, i_all, ts_all = build_temporal_ngh(u_tr, i_tr, t_tr, num_nodes, k=k_ngh)
    x = torch.from_numpy(x_np).float().to(device)
    in_dim = x_np.shape[1]

    model = TGATLinkPredictor(
        in_dim=in_dim, time_dim=time_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        dropout=dropout, k_ngh=k_ngh,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    def make_batch(uu, ii, tt, neg_ratio=1.0):
        n = len(uu)
        neg_n = int(n * neg_ratio)
        neg_u = np.random.randint(0, num_nodes, neg_n)
        neg_i = np.random.randint(0, num_nodes, neg_n)
        all_u = np.concatenate([uu, neg_u])
        all_i = np.concatenate([ii, neg_i])
        all_t = np.concatenate([tt, np.repeat(tt.max(), neg_n)])
        y = np.concatenate([np.ones(n), np.zeros(neg_n)]).astype(np.float32)
        return all_u, all_i, all_t, y

    def train_step(u_b, i_b, t_b, y_b):
        model.train()
        optimizer.zero_grad()
        ngh_src_f, ngh_src_dt, mask_src = get_ngh_tensors(ngh_u, u_b, t_b, x_np, k_ngh, device)
        ngh_dst_f, ngh_dst_dt, mask_dst = get_ngh_tensors(ngh_i, i_b, t_b, x_np, k_ngh, device)
        src_self = x[torch.from_numpy(u_b).long().to(device)]
        dst_self = x[torch.from_numpy(i_b).long().to(device)]
        logits = model(src_self, dst_self, ngh_src_f, ngh_src_dt, mask_src, ngh_dst_f, ngh_dst_dt, mask_dst)
        loss = criterion(logits, torch.from_numpy(y_b).float().to(device))
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(uu, ii, tt, yy, batch_size=512):
        model.eval()
        preds, labels = [], []
        for start in range(0, len(yy), batch_size):
            end = min(start + batch_size, len(yy))
            u_b, i_b = uu[start:end], ii[start:end]
            t_b, y_b = tt[start:end], yy[start:end]
            ngh_src_f, ngh_src_dt, mask_src = get_ngh_tensors(ngh_u, u_b, t_b, x_np, k_ngh, device)
            ngh_dst_f, ngh_dst_dt, mask_dst = get_ngh_tensors(ngh_i, i_b, t_b, x_np, k_ngh, device)
            src_self = x[torch.from_numpy(u_b).long().to(device)]
            dst_self = x[torch.from_numpy(i_b).long().to(device)]
            logits = model(src_self, dst_self, ngh_src_f, ngh_src_dt, mask_src, ngh_dst_f, ngh_dst_dt, mask_dst)
            preds.append(logits.cpu().numpy())
            labels.append(y_b)
        pred = np.concatenate(preds)
        y_true = np.concatenate(labels)
        prob = 1.0 / (1.0 + np.exp(-np.clip(pred, -20, 20)))
        return {
            'AUC': roc_auc_score(y_true, prob),
            'AP': average_precision_score(y_true, prob),
            'Acc': accuracy_score(y_true, (prob >= 0.5).astype(int)),
        }

    # 训练集带负样本
    train_u, train_i, train_t, train_y = make_batch(u_tr, i_tr, t_tr, neg_ratio=1.0)
    n_train = len(train_y)
    # Val/Test: 正 + 等量负
    val_u = np.concatenate([u_val, np.random.randint(0, num_nodes, len(u_val))])
    val_i = np.concatenate([i_val, np.random.randint(0, num_nodes, len(i_val))])
    val_t = np.concatenate([t_val, np.repeat(t_val.max(), len(t_val))])
    val_y = np.concatenate([np.ones(len(u_val)), np.zeros(len(u_val))]).astype(np.float32)
    te_u = np.concatenate([u_te, np.random.randint(0, num_nodes, len(u_te))])
    te_i = np.concatenate([i_te, np.random.randint(0, num_nodes, len(i_te))])
    te_t = np.concatenate([t_te, np.repeat(t_te.max(), len(t_te))])
    te_y = np.concatenate([np.ones(len(u_te)), np.zeros(len(u_te))]).astype(np.float32)

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n_train)
        total_loss = 0
        n_b = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            loss = train_step(
                train_u[idx], train_i[idx], train_t[idx], train_y[idx]
            )
            total_loss += loss
            n_b += 1
        if (epoch % 20 == 0) or epoch == 1:
            v_m = evaluate(val_u, val_i, val_t, val_y, batch_size)
            print(f"Epoch {epoch}, Loss: {total_loss/max(n_b,1):.4f}, Val AUC: {v_m['AUC']:.4f}, Val AP: {v_m['AP']:.4f}")

    train_m = evaluate(train_u, train_i, train_t, train_y, batch_size)
    val_m = evaluate(val_u, val_i, val_t, val_y, batch_size)
    test_m = evaluate(te_u, te_i, te_t, te_y, batch_size)

    # Inductive（Section V-B）：仅测试集中至少含一个“新节点”的边
    ind_mask = get_inductive_mask(u_tr, i_tr, u_te, i_te)
    if ind_mask.sum() > 0:
        nn_u = np.concatenate([u_te[ind_mask], np.random.randint(0, num_nodes, ind_mask.sum())])
        nn_i = np.concatenate([i_te[ind_mask], np.random.randint(0, num_nodes, ind_mask.sum())])
        nn_t = np.concatenate([t_te[ind_mask], np.repeat(t_te.max(), ind_mask.sum())])
        nn_y = np.concatenate([np.ones(ind_mask.sum()), np.zeros(ind_mask.sum())]).astype(np.float32)
        nn_test_m = evaluate(nn_u, nn_i, nn_t, nn_y, batch_size)
        results = {
            'Train_AUC': train_m['AUC'], 'Train_AP': train_m['AP'], 'Train_Acc': train_m['Acc'],
            'Val_AUC': val_m['AUC'], 'Val_AP': val_m['AP'], 'Val_Acc': val_m['Acc'],
            'Test_AUC': test_m['AUC'], 'Test_AP': test_m['AP'], 'Test_Acc': test_m['Acc'],
            'NN_Test_AUC': nn_test_m['AUC'], 'NN_Test_AP': nn_test_m['AP'], 'NN_Test_Acc': nn_test_m['Acc'],
        }
    else:
        results = {
            'Train_AUC': train_m['AUC'], 'Train_AP': train_m['AP'], 'Train_Acc': train_m['Acc'],
            'Val_AUC': val_m['AUC'], 'Val_AP': val_m['AP'], 'Val_Acc': val_m['Acc'],
            'Test_AUC': test_m['AUC'], 'Test_AP': test_m['AP'], 'Test_Acc': test_m['Acc'],
            'NN_Test_AUC': np.nan, 'NN_Test_AP': np.nan, 'NN_Test_Acc': np.nan,
        }
    out_dir = os.path.join(code_root, 'result', 'link', data_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'link_tgat_{data_name}.csv')
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"TGAT 结果已保存: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='TGAT 链路预测')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--list_data', action='store_true')
    parser.add_argument('--split_by_ui', action='store_true')
    parser.add_argument('--paper_eval', action='store_true', help='与论文一致：按边 10%%:10%%:80%% 划分')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--k_ngh', type=int, default=10)
    args = parser.parse_args()

    if args.list_data:
        for d in list_available_data(CODE_ROOT):
            print(d)
        return

    edges_file, feature_file, data_name = get_link_data_paths(args.data_dir, CODE_ROOT)
    if not edges_file:
        print("错误: 未指定或找不到 data_dir")
        return
    run_tgat(
        edges_file, feature_file, data_name, CODE_ROOT,
        split_by_ui=args.split_by_ui, paper_eval=args.paper_eval, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_ngh=args.k_ngh,
    )


if __name__ == '__main__':
    main()