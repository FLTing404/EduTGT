"""
GraphSAGE 链路预测基线。
论文: Hamilton et al., Inductive Representation Learning on Large Graphs, NeurIPS 2017.
表中方法之一，PyTorch Geometric 实现。
从项目根运行: python link_models/train_link_graphsage.py --data_dir data_abc_0.01
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

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
    get_dst_nodes,
    sample_neg_dst_same_u,
    list_available_data,
)

try:
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def build_edge_index(u, i, num_nodes):
    """无向图 edge_index (2, E)，0-indexed。"""
    u = np.clip(u, 0, num_nodes - 1)
    i = np.clip(i, 0, num_nodes - 1)
    edge_u = np.concatenate([u, i])
    edge_v = np.concatenate([i, u])
    return torch.from_numpy(np.stack([edge_u, edge_v])).long()


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h

    def link_logits(self, z, u_idx, i_idx):
        zu = z[u_idx]
        zi = z[i_idx]
        return self.decoder(torch.cat([zu, zi], dim=-1)).squeeze(-1)


def train_epoch(model, x, edge_index, train_ui, y_train, optimizer, criterion, device, batch_size=1024):
    model.train()
    perm = np.random.permutation(len(y_train))
    total_loss = 0
    n_batches = 0
    for start in range(0, len(perm), batch_size):
        idx = perm[start:start + batch_size]
        u_idx = torch.from_numpy(train_ui[idx, 0]).long().to(device)
        i_idx = torch.from_numpy(train_ui[idx, 1]).long().to(device)
        y = torch.from_numpy(y_train[idx]).float().to(device)
        optimizer.zero_grad()
        z = model(x, edge_index)
        logits = model.link_logits(z, u_idx, i_idx)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, x, edge_index, ui, y, device, batch_size=1024):
    model.eval()
    preds = []
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        u_idx = torch.from_numpy(ui[start:end, 0]).long().to(device)
        i_idx = torch.from_numpy(ui[start:end, 1]).long().to(device)
        z = model(x, edge_index)
        logits = model.link_logits(z, u_idx, i_idx)
        preds.append(logits.cpu().numpy())
    pred = np.concatenate(preds)
    prob = 1.0 / (1.0 + np.exp(-np.clip(pred, -20, 20)))
    return {
        'AUC': roc_auc_score(y, prob),
        'AP': average_precision_score(y, prob),
        'Acc': accuracy_score(y, (prob >= 0.5).astype(int)),
    }


def run_graphsage(edges_file, feature_file, data_name, code_root, split_by_ui=False,
                  paper_eval=False, train_ratio=None, val_ratio=0.1, test_ratio=0.8, seed=42, device=None,
                  hidden_dim=128, out_dim=64, dropout=0.2, epochs=100, lr=0.01, batch_size=1024):
    if not HAS_PYG:
        raise RuntimeError("需要安装 PyTorch Geometric: pip install torch_geometric")

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
        # 与 ContraTGT 统一：按时间 quantile 0.1, 0.2 => 10% train, 10% val, 80% test
        (u_tr, i_tr, t_tr), (u_val, i_val, t_val), (u_te, i_te, t_te) = split_edges_by_time(
            u, i, ts, val_ratio=0.1, test_ratio=0.8, seed=seed
        )

    # 仅用训练集边建图（防泄露）
    edge_index = build_edge_index(u_tr, i_tr, num_nodes).to(device)
    x = torch.from_numpy(x_np).float().to(device)
    in_dim = x_np.shape[1]

    # 与 ContraTGT 公平一致：训练负采样用训练集目标节点；Val/Test 用全量边目标节点
    rng = np.random.default_rng(seed)
    dst_nodes_train = get_dst_nodes(i_tr)
    all_dst = np.unique(np.concatenate([i_tr, i_val, i_te]))
    pos_ui = np.stack([u_tr, i_tr], axis=1)
    n_pos = len(pos_ui)
    neg_i = sample_neg_dst_same_u(i_tr, dst_nodes_train, rng)
    neg_ui = np.stack([u_tr, neg_i], axis=1)
    train_ui = np.vstack([pos_ui, neg_ui])
    train_y = np.concatenate([np.ones(n_pos), np.zeros(n_pos)]).astype(np.float32)
    neg_vi = sample_neg_dst_same_u(i_val, all_dst, rng)
    val_ui = np.vstack([np.stack([u_val, i_val], axis=1), np.stack([u_val, neg_vi], axis=1)])
    val_y = np.concatenate([np.ones(len(u_val)), np.zeros(len(u_val))]).astype(np.float32)
    neg_ti = sample_neg_dst_same_u(i_te, all_dst, rng)
    te_ui = np.vstack([np.stack([u_te, i_te], axis=1), np.stack([u_te, neg_ti], axis=1)])
    te_y = np.concatenate([np.ones(len(u_te)), np.zeros(len(u_te))]).astype(np.float32)

    model = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, x, edge_index, train_ui, train_y, optimizer, criterion, device, batch_size)
        if (epoch % 20 == 0) or epoch == 1:
            v_m = evaluate(model, x, edge_index, val_ui, val_y, device, batch_size)
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Val AUC: {v_m['AUC']:.4f}, Val AP: {v_m['AP']:.4f}")

    train_m = evaluate(model, x, edge_index, train_ui, train_y, device, batch_size)
    val_m = evaluate(model, x, edge_index, val_ui, val_y, device, batch_size)
    test_m = evaluate(model, x, edge_index, te_ui, te_y, device, batch_size)

    ind_mask = get_inductive_mask(u_tr, i_tr, u_te, i_te)
    if ind_mask.sum() > 0:
        nn_pos = np.stack([u_te[ind_mask], i_te[ind_mask]], axis=1)
        nn_neg_i = sample_neg_dst_same_u(i_te[ind_mask], all_dst, rng)
        nn_ui = np.vstack([nn_pos, np.stack([u_te[ind_mask], nn_neg_i], axis=1)])
        nn_y = np.concatenate([np.ones(ind_mask.sum()), np.zeros(ind_mask.sum())]).astype(np.float32)
        nn_test_m = evaluate(model, x, edge_index, nn_ui, nn_y, device, batch_size)
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
    out_path = os.path.join(out_dir, f'link_graphsage_{data_name}.csv')
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"GraphSAGE 结果已保存: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='GraphSAGE 链路预测')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--list_data', action='store_true')
    parser.add_argument('--split_by_ui', action='store_true')
    parser.add_argument('--paper_eval', action='store_true', help='与论文一致：按边 10%%:10%%:80%% 划分')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    if args.list_data:
        for d in list_available_data(CODE_ROOT):
            print(d)
        return

    edges_file, feature_file, data_name = get_link_data_paths(args.data_dir, CODE_ROOT)
    if not edges_file:
        print("错误: 未指定或找不到 data_dir")
        return
    run_graphsage(
        edges_file, feature_file, data_name, CODE_ROOT,
        split_by_ui=args.split_by_ui, paper_eval=args.paper_eval, seed=args.seed,
        epochs=args.epochs, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
        lr=args.lr, batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
