"""
Node2Vec + MLP 基线 - 图嵌入链接预测
在图上做随机游走，用 Skip-Gram 负采样得到节点嵌入，再对 (u,i) 用 MLP 做二分类。数据与划分与 XGBoost/LSTM 一致。
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
from collections import defaultdict


def load_node_features(feature_path):
    """加载节点特征（本脚本主要用图结构，特征可选用于初始化）"""
    features = []
    with open(feature_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                features.append([float(x) for x in line.split(',')])
    return np.array(features, dtype=np.float32)


def build_adj(edges_df, num_nodes):
    """无向图邻接表，节点 0-indexed。"""
    adj = defaultdict(list)
    u = (edges_df['u'].values - 1).astype(int)
    i = (edges_df['i'].values - 1).astype(int)
    u, i = np.clip(u, 0, num_nodes - 1), np.clip(i, 0, num_nodes - 1)
    for a, b in zip(u, i):
        adj[a].append(b)
        adj[b].append(a)
    return dict(adj)


def random_walks(adj, num_nodes, num_walks=10, walk_length=80, seed=42):
    """每个节点出发 num_walks 次，每次走 walk_length 步。"""
    np.random.seed(seed)
    walks = []
    nodes = list(range(num_nodes))
    for _ in range(num_walks):
        for start in nodes:
            walk = [start]
            cur = start
            for _ in range(walk_length - 1):
                ne = adj.get(cur, [cur])
                cur = np.random.choice(ne)
                walk.append(cur)
            walks.append(walk)
    return walks


def train_node2vec_embedding(adj, num_nodes, embed_dim=64, num_epochs=5, window=5, neg_samples=5, lr=0.025, device='cpu',
                             num_walks=10, walk_length=80, batch_size=2048):
    """Skip-Gram 负采样学习节点嵌入（小批量训练以加速）。"""
    walks = random_walks(adj, num_nodes, num_walks=num_walks, walk_length=walk_length)
    # 构建 (center, context) 正样本
    pairs = []
    for walk in walks:
        for i in range(len(walk)):
            for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                if i != j:
                    pairs.append((walk[i], walk[j]))
    if not pairs:
        # 退化为随机初始化
        emb = nn.Embedding(num_nodes, embed_dim)
        return emb.weight.data.numpy()

    pairs = np.array(pairs, dtype=np.int64)  # (N, 2)
    emb = nn.Embedding(num_nodes, embed_dim)
    nn.init.xavier_uniform_(emb.weight)
    emb = emb.to(device)
    optimizer = torch.optim.Adam(emb.parameters(), lr=lr)

    for epoch in range(num_epochs):
        np.random.shuffle(pairs)
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(range(0, len(pairs), batch_size), desc=f"Node2Vec epoch {epoch+1}", leave=False)
        for start in pbar:
            end = min(start + batch_size, len(pairs))
            c_batch = torch.tensor(pairs[start:end, 0], device=device).long()
            ctx_batch = torch.tensor(pairs[start:end, 1], device=device).long()
            B = c_batch.size(0)
            pos_score = (emb(c_batch) * emb(ctx_batch)).sum(dim=1)
            neg_nodes = np.random.randint(0, num_nodes, size=(B, neg_samples))
            neg_t = torch.tensor(neg_nodes, device=device).long()
            neg_score = (emb(c_batch).unsqueeze(1) * emb(neg_t)).sum(dim=2)
            loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean() - torch.log(torch.sigmoid(-neg_score) + 1e-8).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return emb.weight.data.cpu().numpy()


class LinkMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb_u, emb_i):
        return self.mlp(torch.cat([emb_u, emb_i], dim=-1)).squeeze(-1)


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
    parser = argparse.ArgumentParser(description='Node2Vec + MLP 链接预测基线')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--list_data', action='store_true')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--feature_path', type=str, default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_walks', type=int, default=10, help='每个节点随机游走次数，小数据可减至 5')
    parser.add_argument('--walk_length', type=int, default=80, help='每条游走长度，小数据可减至 40')
    parser.add_argument('--n2v_batch_size', type=int, default=2048, help='Node2Vec Skip-Gram 小批量大小，越大越快')
    parser.add_argument('--n2v_epochs', type=int, default=5)
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
        print("\n用法: python models/train_node2vec.py --data_dir <目录名>")
        return

    if args.data_dir:
        data_dir_path = get_data_dir_path(code_root, args.data_dir) or os.path.join(code_root, 'data', 'all_data', args.data_dir)
        data_path = os.path.join(data_dir_path, 'ml_oulad.csv')
        feature_path = os.path.join(data_dir_path, 'oulad.content')
        results_filename = f'node2vec_results_{args.data_dir}.csv'
        output_dir = args.output_dir or os.path.join(code_root, 'result', args.data_dir)
    else:
        data_path = args.data_path or os.path.join(code_root, 'data', 'ml_oulad.csv')
        feature_path = args.feature_path or os.path.join(code_root, 'ContraTGT', 'node_feature', 'oulad.content')
        results_filename = 'node2vec_results.csv'
        output_dir = args.output_dir or './'

    if not os.path.exists(data_path):
        print("错误: 找不到边数据文件")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    edges_df = pd.read_csv(data_path)
    if args.sample_ratio < 1.0:
        edges_df = edges_df.sample(frac=args.sample_ratio, random_state=args.random_state).reset_index(drop=True)

    if os.path.exists(feature_path):
        node_features = load_node_features(feature_path)
        num_nodes = len(node_features)
    else:
        num_nodes = int(edges_df[['u', 'i']].max().max())
        node_features = np.zeros((num_nodes, args.embed_dim), dtype=np.float32)

    adj = build_adj(edges_df, num_nodes)
    print("学习 Node2Vec 嵌入...")
    embeddings = train_node2vec_embedding(
        adj, num_nodes, embed_dim=args.embed_dim,
        num_epochs=args.n2v_epochs, device=device,
        num_walks=args.num_walks, walk_length=args.walk_length,
        batch_size=args.n2v_batch_size
    )
    emb_tensor = torch.from_numpy(embeddings).float().to(device)

    unique_ui = edges_df[['u', 'i', 'label']].drop_duplicates(subset=['u', 'i']).reset_index(drop=True)
    y_ui = (unique_ui['label'] == 1).astype(np.float32)
    ui = unique_ui[['u', 'i']].values.astype(np.int64) - 1
    ui = np.clip(ui, 0, num_nodes - 1)

    ui_train, ui_temp, y_train, y_temp = train_test_split(
        ui, y_ui, test_size=args.test_size + args.val_size, random_state=args.random_state, stratify=y_ui
    )
    val_ratio = args.val_size / (args.test_size + args.val_size)
    ui_val, ui_test, y_val, y_test = train_test_split(
        ui_temp, y_temp, test_size=1 - val_ratio, random_state=args.random_state, stratify=y_temp
    )

    model = LinkMLP(args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    def train_epoch(ui_batch, y_batch):
        model.train()
        idx = np.random.permutation(len(y_batch))
        ui_batch = ui_batch[idx]
        y_batch = y_batch.values[idx]
        for start in range(0, len(y_batch), 256):
            end = min(start + 256, len(y_batch))
            u_t = torch.from_numpy(ui_batch[start:end, 0]).long().to(device)
            i_t = torch.from_numpy(ui_batch[start:end, 1]).long().to(device)
            y_t = torch.from_numpy(y_batch[start:end]).float().to(device)
            optimizer.zero_grad()
            eu = emb_tensor[u_t]
            ei = emb_tensor[i_t]
            logits = model(eu, ei)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict(ui_batch, y_batch):
        model.eval()
        preds = []
        for start in range(0, len(y_batch), 256):
            end = min(start + 256, len(y_batch))
            u_t = torch.from_numpy(ui_batch[start:end, 0]).long().to(device)
            i_t = torch.from_numpy(ui_batch[start:end, 1]).long().to(device)
            eu = emb_tensor[u_t]
            ei = emb_tensor[i_t]
            preds.append(model(eu, ei).cpu().numpy())
        return np.concatenate(preds)

    print("训练 MLP 链接预测...")
    for epoch in range(args.num_epochs):
        train_epoch(ui_train, y_train)
        if (epoch + 1) % 20 == 0:
            p = predict(ui_train, y_train)
            loss = log_loss(y_train.values, 1.0 / (1.0 + np.exp(-np.clip(p, -500, 500))))
            print(f"  epoch {epoch+1} train_loss {loss:.4f}")

    def metrics(y_true, y_pred):
        y_p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -500, 500)))
        return {
            'AUC': roc_auc_score(y_true, y_p),
            'AP': average_precision_score(y_true, y_p),
            'Acc': accuracy_score(y_true, (y_p >= 0.5).astype(int)),
            'Loss': log_loss(y_true, np.clip(y_p, 1e-7, 1 - 1e-7)),
        }

    train_pred = predict(ui_train, y_train)
    val_pred = predict(ui_val, y_val)
    test_pred = predict(ui_test, y_test)

    train_m = metrics(y_train.values, train_pred)
    val_m = metrics(y_val.values, val_pred)
    test_m = metrics(y_test.values, test_pred)

    def prefixed(d, prefix):
        return {f'{prefix}{k}': v for k, v in d.items()}

    results = {**prefixed(train_m, 'Train_'), **prefixed(val_m, 'Val_'), **prefixed(test_m, 'Test_')}
    results['embed_dim'] = args.embed_dim
    results['num_epochs'] = args.num_epochs

    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame([results])
    results_path = os.path.join(output_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"结果已保存: {results_path}")


if __name__ == '__main__':
    main()
