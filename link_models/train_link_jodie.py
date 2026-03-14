"""
JODIE (Predicting Dynamic Embedding Trajectory) 链路预测基线。
论文: Kumar et al., Predicting Dynamic Embedding Trajectory for Temporal Link Prediction, KDD 2019.
简化实现：节点嵌入 + 时序 GRU 更新 + 链接 MLP。与 TGAT/GraphSAGE 公平对比：同一数据划分、
同一训练负采样池（dst_train）、同一 Val/Test 负采样池（all_dst）、同一评估指标（AUC/AP/Acc）。
从项目根运行: python link_models/train_link_jodie.py --data_dir data_abc_0.01
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


class TimeEncode(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(1, dim)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return torch.cos(self.lin(t))


class JODIE(nn.Module):
    """JODIE 风格：节点嵌入 + GRU 按时间流更新 + 链接 MLP。"""
    def __init__(self, num_nodes, in_dim, emb_dim=128, time_dim=32, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.time_enc = TimeEncode(time_dim)
        self.feat_proj = nn.Linear(in_dim, emb_dim)
        self.emb = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.01)
        self.gru = nn.GRUCell(emb_dim * 2 + time_dim, emb_dim)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim * 2 + time_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, u, i, t, x_np, device, update_emb=True):
        """
        u, i, t: (B,) 当前 batch 的 src, dst, time（可为 numpy 或 tensor）
        x_np: (num_nodes, in_dim) 节点特征
        update_emb: 是否用 GRU 更新嵌入（训练时 True，评估可 False 或按时间顺序 True）
        """
        if not torch.is_tensor(u):
            u = torch.from_numpy(u).long().to(device)
            i = torch.from_numpy(i).long().to(device)
            t = torch.from_numpy(t).float().to(device)
        else:
            u, i = u.to(device), i.to(device)
            t = t.float().to(device)
        B = u.shape[0]
        t_enc = self.time_enc(t.unsqueeze(1))  # (B, time_dim)
        u_np = u.cpu().numpy()
        i_np = i.cpu().numpy()
        emb_u = self.emb[u] + self.feat_proj(torch.from_numpy(x_np[u_np]).float().to(device))
        emb_i = self.emb[i] + self.feat_proj(torch.from_numpy(x_np[i_np]).float().to(device))
        if update_emb:
            msg_u = torch.cat([emb_i, emb_u, t_enc], dim=-1)
            msg_i = torch.cat([emb_u, emb_i, t_enc], dim=-1)
            new_u = self.gru(msg_u, emb_u)
            new_i = self.gru(msg_i, emb_i)
            self.emb.data[u] = new_u.detach()
            self.emb.data[i] = new_i.detach()
            emb_u, emb_i = new_u, new_i
        z = torch.cat([emb_u, emb_i, t_enc], dim=-1)
        return self.decoder(z).squeeze(-1)


def run_jodie(edges_file, feature_file, data_name, code_root, split_by_ui=False,
              paper_eval=False, train_ratio=None, val_ratio=0.1, test_ratio=0.8, seed=42,
              emb_dim=128, time_dim=32, batch_size=256, epochs=80, lr=1e-3, dropout=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    u, i, ts, num_nodes, x_np, _ = load_edges_and_features(edges_file, feature_file)
    in_dim = x_np.shape[1]
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
            u, i, ts, val_ratio=0.1, test_ratio=0.8, seed=seed
        )

    all_dst = np.unique(np.concatenate([i_tr, i_val, i_te]))
    dst_train = get_dst_nodes(i_tr)
    rng = np.random.default_rng(seed)

    n_train = len(u_tr)
    train_u = np.concatenate([u_tr, u_tr])
    neg_i_tr = sample_neg_dst_same_u(i_tr, dst_train, rng)
    train_i = np.concatenate([i_tr, neg_i_tr])
    train_t = np.concatenate([t_tr, t_tr])
    train_y = np.concatenate([np.ones(len(u_tr)), np.zeros(len(u_tr))]).astype(np.float32)
    neg_i_val = sample_neg_dst_same_u(i_val, all_dst, rng)
    val_u = np.concatenate([u_val, u_val])
    val_i = np.concatenate([i_val, neg_i_val])
    val_t = np.concatenate([t_val, t_val])
    val_y = np.concatenate([np.ones(len(u_val)), np.zeros(len(u_val))]).astype(np.float32)
    neg_i_te = sample_neg_dst_same_u(i_te, all_dst, rng)
    te_u = np.concatenate([u_te, u_te])
    te_i = np.concatenate([i_te, neg_i_te])
    te_t = np.concatenate([t_te, t_te])
    te_y = np.concatenate([np.ones(len(u_te)), np.zeros(len(u_te))]).astype(np.float32)

    model = JODIE(num_nodes, in_dim, emb_dim=emb_dim, time_dim=time_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    def train_step(u_b, i_b, t_b, y_b):
        model.train()
        optimizer.zero_grad()
        out = model(u_b, i_b, t_b, x_np, device, update_emb=True)
        loss = criterion(out, torch.from_numpy(y_b).float().to(device))
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(u_arr, i_arr, t_arr, y_arr, bs, update_emb=False):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for start in range(0, len(u_arr), bs):
                end = min(start + bs, len(u_arr))
                u_b = torch.from_numpy(u_arr[start:end]).long()
                i_b = torch.from_numpy(i_arr[start:end]).long()
                t_b = torch.from_numpy(t_arr[start:end]).float()
                out = model(u_b, i_b, t_b, x_np, device, update_emb=update_emb)
                preds.append(out.sigmoid().cpu().numpy())
                labels.append(y_arr[start:end])
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        return {
            'AUC': roc_auc_score(labels, preds),
            'AP': average_precision_score(labels, preds),
            'Acc': accuracy_score(labels, (preds >= 0.5).astype(float)),
        }

    # 训练：正+负样本一起 shuffle 后 batch 训练
    n_train_pairs = len(train_u)
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n_train_pairs)
        total_loss = 0
        n_b = 0
        for start in range(0, n_train_pairs, batch_size):
            idx = perm[start:start + batch_size]
            u_b = train_u[idx]
            i_b = train_i[idx]
            t_b = train_t[idx]
            y_b = train_y[idx]
            loss = train_step(u_b, i_b, t_b, y_b)
            total_loss += loss
            n_b += 1
        if (epoch % 20 == 0) or epoch == 1:
            v_m = evaluate(val_u, val_i, val_t, val_y, batch_size, update_emb=False)
            print(f"Epoch {epoch}, Loss: {total_loss/max(n_b,1):.4f}, Val AUC: {v_m['AUC']:.4f}, Val AP: {v_m['AP']:.4f}")

    train_m = evaluate(train_u, train_i, train_t, train_y, batch_size)
    val_m = evaluate(val_u, val_i, val_t, val_y, batch_size)
    test_m = evaluate(te_u, te_i, te_t, te_y, batch_size)
    ind_mask = get_inductive_mask(u_tr, i_tr, u_te, i_te)
    if ind_mask.sum() > 0:
        nn_neg_i = sample_neg_dst_same_u(i_te[ind_mask], all_dst, rng)
        nn_u = np.concatenate([u_te[ind_mask], u_te[ind_mask]])
        nn_i = np.concatenate([i_te[ind_mask], nn_neg_i])
        nn_t = np.concatenate([t_te[ind_mask], t_te[ind_mask]])
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
    out_path = os.path.join(out_dir, f'link_jodie_{data_name}.csv')
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"JODIE 结果已保存: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='JODIE 链路预测')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--list_data', action='store_true')
    parser.add_argument('--split_by_ui', action='store_true')
    parser.add_argument('--paper_eval', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--emb_dim', type=int, default=128)
    args = parser.parse_args()

    if args.list_data:
        for d in list_available_data(CODE_ROOT):
            print(d)
        return

    edges_file, feature_file, data_name = get_link_data_paths(args.data_dir, CODE_ROOT)
    if not edges_file:
        print("错误: 未指定或找不到 data_dir")
        return
    run_jodie(
        edges_file, feature_file, data_name, CODE_ROOT,
        split_by_ui=args.split_by_ui, paper_eval=args.paper_eval, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, emb_dim=args.emb_dim,
    )


if __name__ == '__main__':
    main()
