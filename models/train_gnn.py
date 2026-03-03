"""
GNN 基线 - 2 层 GCN 链接预测
使用 ml_oulad.csv 与 oulad.content，按 (u,i) 划分防泄露，与 XGBoost/LSTM 一致。
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, log_loss
import os
import argparse
from tqdm import tqdm


def load_node_features(feature_path):
    """加载节点特征文件"""
    print(f"正在加载节点特征: {feature_path}")
    features = []
    with open(feature_path, 'r') as f:
        for line in tqdm(f, desc="加载特征", unit="行"):
            line = line.strip()
            if line:
                feature_vec = [float(x) for x in line.split(',')]
                features.append(feature_vec)
    return np.array(features, dtype=np.float32)


def build_edge_index(edges_df, num_nodes):
    """构建无向图 edge_index (2, E)，节点 0-indexed。"""
    u = (edges_df['u'].values - 1).astype(np.int64)
    i = (edges_df['i'].values - 1).astype(np.int64)
    u, i = np.clip(u, 0, num_nodes - 1), np.clip(i, 0, num_nodes - 1)
    edge_u = np.concatenate([u, i])
    edge_v = np.concatenate([i, u])
    return torch.from_numpy(np.stack([edge_u, edge_v]))


def gcn_norm(edge_index, num_nodes):
    """加自环后计算 D^{-1/2}(A+I)D^{-1/2} 的 edge_weight。"""
    self_loop = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    edge_index_loop = torch.cat([edge_index, self_loop], dim=1)
    row, col = edge_index_loop[0], edge_index_loop[1]
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32, device=edge_index.device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index_loop, weight


class GCN(nn.Module):
    """2 层 GCN + 链接预测 MLP"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x, edge_index, edge_weight):
        # 第 1 层：H = A_norm @ X @ W1, 用 scatter 实现 A_norm @ (X @ W1)
        h = self.conv1(x)
        h = self._propagate(edge_index, edge_weight, h)
        h = torch.relu(h)
        h = self.dropout(h)
        # 第 2 层
        h = self.conv2(h)
        h = self._propagate(edge_index, edge_weight, h)
        return h

    def _propagate(self, edge_index, edge_weight, x):
        row, col = edge_index[0], edge_index[1]
        out = torch.zeros_like(x)
        weight_x = edge_weight.unsqueeze(-1) * x[col]
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(weight_x), weight_x)
        return out

    def link_logits(self, z, u_idx, i_idx):
        zu = z[u_idx]
        zi = z[i_idx]
        return self.decoder(torch.cat([zu, zi], dim=-1)).squeeze(-1)


def train_gnn(model, x, edge_index, edge_weight, train_ui, y_train, optimizer, criterion, device, batch_size=1024):
    model.train()
    indices = np.random.permutation(len(y_train))
    total_loss = 0.0
    n_batch = (len(indices) + batch_size - 1) // batch_size
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        idx = indices[start:end]
        u = train_ui[idx, 0]
        i = train_ui[idx, 1]
        u_t = torch.from_numpy(u).long().to(device)
        i_t = torch.from_numpy(i).long().to(device)
        y_t = torch.from_numpy(y_train[idx]).float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        z = model(x, edge_index, edge_weight)
        logits = model.link_logits(z, u_t, i_t).unsqueeze(1)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / n_batch


@torch.no_grad()
def eval_gnn(model, x, edge_index, edge_weight, ui, y, device, batch_size=1024):
    model.eval()
    preds = []
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        u_t = torch.from_numpy(ui[start:end, 0]).long().to(device)
        i_t = torch.from_numpy(ui[start:end, 1]).long().to(device)
        z = model(x, edge_index, edge_weight)
        logits = model.link_logits(z, u_t, i_t)
        preds.append(logits.cpu().numpy())
    pred = np.concatenate(preds)
    return pred


def get_data_dir_path(code_root, data_dir):
    all_data_dir = os.path.join(code_root, 'data', 'all_data')
    cand = os.path.join(all_data_dir, data_dir)
    if os.path.isdir(cand) and os.path.exists(os.path.join(cand, 'ml_oulad.csv')):
        return cand
    return None


def list_available_data(script_dir):
    code_root = os.path.dirname(script_dir)
    all_data_dir = os.path.join(code_root, 'data', 'all_data')
    available_dirs = []
    if os.path.exists(all_data_dir):
        for item in os.listdir(all_data_dir):
            item_path = os.path.join(all_data_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'ml_oulad.csv')):
                available_dirs.append(item)
    return sorted(available_dirs)


def main():
    parser = argparse.ArgumentParser(description='GNN (GCN) 链接预测基线')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录名，如 data_abc_0.01')
    parser.add_argument('--list_data', action='store_true', help='列出可用数据目录')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--feature_path', type=str, default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_root = os.path.dirname(script_dir)

    if args.list_data:
        print("可用的数据文件夹 (code/data/all_data/):")
        for d in list_available_data(script_dir):
            path = get_data_dir_path(code_root, d) or os.path.join(code_root, 'data', 'all_data', d)
            fp = os.path.join(path, 'ml_oulad.csv')
            n = len(pd.read_csv(fp)) if os.path.exists(fp) else 0
            print(f"  {d} - {n:,} 条边")
        print("\n用法: python models/train_gnn.py --data_dir <目录名>")
        return

    if args.data_dir:
        data_dir_path = get_data_dir_path(code_root, args.data_dir) or os.path.join(code_root, 'data', 'all_data', args.data_dir)
        data_path = os.path.join(data_dir_path, 'ml_oulad.csv')
        feature_path = os.path.join(data_dir_path, 'oulad.content')
        results_filename = f'gnn_results_{args.data_dir}.csv'
        output_dir = args.output_dir or os.path.join(code_root, 'result', args.data_dir)
    else:
        data_path = args.data_path or os.path.join(code_root, 'data', 'ml_oulad.csv')
        feature_path = args.feature_path or os.path.join(code_root, 'ContraTGT', 'node_feature', 'oulad.content')
        results_filename = 'gnn_results.csv'
        output_dir = args.output_dir or './'

    if not os.path.exists(data_path) or not os.path.exists(feature_path):
        print(f"错误: 找不到数据或特征文件")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    edges_df = pd.read_csv(data_path)
    if args.sample_ratio < 1.0:
        edges_df = edges_df.sample(frac=args.sample_ratio, random_state=args.random_state).reset_index(drop=True)
    node_features = load_node_features(feature_path)
    num_nodes = len(node_features)
    x = torch.from_numpy(node_features).float().to(device)

    edge_index = build_edge_index(edges_df, num_nodes).to(device)
    edge_index, edge_weight = gcn_norm(edge_index, num_nodes)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    unique_ui = edges_df[['u', 'i', 'label']].drop_duplicates(subset=['u', 'i']).reset_index(drop=True)
    y_ui = (unique_ui['label'] == 1).astype(np.float32)
    ui = unique_ui[['u', 'i']].values.astype(np.int64)

    ui_train, ui_temp, y_train, y_temp = train_test_split(
        ui, y_ui, test_size=args.test_size + args.val_size, random_state=args.random_state, stratify=y_ui
    )
    val_ratio = args.val_size / (args.test_size + args.val_size)
    ui_val, ui_test, y_val, y_test = train_test_split(
        ui_temp, y_temp, test_size=1 - val_ratio, random_state=args.random_state, stratify=y_temp
    )

    # 转为 0-indexed 用于索引
    ui_train_idx = ui_train - 1
    ui_val_idx = ui_val - 1
    ui_test_idx = ui_test - 1
    ui_train_idx = np.clip(ui_train_idx, 0, num_nodes - 1)
    ui_val_idx = np.clip(ui_val_idx, 0, num_nodes - 1)
    ui_test_idx = np.clip(ui_test_idx, 0, num_nodes - 1)

    in_dim = node_features.shape[1]
    model = GCN(in_dim, args.hidden_dim, args.out_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print("训练 GCN 链接预测...")
    for epoch in range(args.num_epochs):
        loss = train_gnn(model, x, edge_index, edge_weight, ui_train_idx, y_train.values, optimizer, criterion, device)
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1} loss {loss:.4f}")

    def metrics(y_true, y_pred):
        y_p = 1.0 / (1.0 + np.exp(-y_pred))
        return {
            'AUC': roc_auc_score(y_true, y_p),
            'AP': average_precision_score(y_true, y_p),
            'Acc': accuracy_score(y_true, (y_p >= 0.5).astype(int)),
            'Loss': log_loss(y_true, np.clip(y_p, 1e-7, 1 - 1e-7)),
        }

    train_pred = eval_gnn(model, x, edge_index, edge_weight, ui_train_idx, y_train.values, device)
    val_pred = eval_gnn(model, x, edge_index, edge_weight, ui_val_idx, y_val.values, device)
    test_pred = eval_gnn(model, x, edge_index, edge_weight, ui_test_idx, y_test.values, device)

    train_m = metrics(y_train.values, train_pred)
    val_m = metrics(y_val.values, val_pred)
    test_m = metrics(y_test.values, test_pred)

    def prefixed(d, prefix):
        return {f'{prefix}{k}': v for k, v in d.items()}

    results = {**prefixed(train_m, 'Train_'), **prefixed(val_m, 'Val_'), **prefixed(test_m, 'Test_')}
    results['hidden_dim'] = args.hidden_dim
    results['out_dim'] = args.out_dim
    results['num_epochs'] = args.num_epochs

    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame([results])
    results_path = os.path.join(output_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"结果已保存: {results_path}")


if __name__ == '__main__':
    main()
