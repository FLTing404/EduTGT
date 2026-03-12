"""
TGN (Temporal Graph Networks) 链路预测基线。
论文: Rossi et al., Temporal Graph Networks for Deep Learning on Dynamic Graphs, 2020.
代码参考: PyTorch Geometric examples/tgn.py, twitter-research/tgn.
从项目根运行: python link_models/train_link_tgn.py --data_dir data_abc_0.01
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from link_models.data_utils import (
    get_link_data_paths,
    load_edges_and_features,
    split_edges_by_time,
    split_edges_by_ui,
    list_available_data,
)

try:
    from torch_geometric.data import TemporalData
    from torch_geometric.loader import TemporalDataLoader
    from torch_geometric.nn import TGNMemory, TransformerConv
    from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader
    from torch.nn import Linear
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).squeeze(-1)


def build_temporal_data(u, i, ts, msg_dim=1):
    """构建 PyG TemporalData，按时间排序。"""
    order = np.argsort(ts)
    u, i, ts = u[order], i[order], ts[order]
    src = torch.from_numpy(u).long()
    dst = torch.from_numpy(i).long()
    t = torch.from_numpy(ts).float().unsqueeze(1)
    msg = torch.zeros(len(u), msg_dim, dtype=torch.float32)
    return TemporalData(src=src, dst=dst, t=t, msg=msg)


def run_tgn(edges_file, feature_file, data_name, code_root, split_by_ui=False,
            val_ratio=0.15, test_ratio=0.15, seed=42, device=None,
            memory_dim=100, time_dim=100, embedding_dim=100,
            batch_size=200, epochs=50, lr=1e-4):
    if not HAS_PYG:
        raise RuntimeError("需要安装 PyTorch Geometric: pip install torch_geometric")

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    u, i, ts, num_nodes, x_np, _ = load_edges_and_features(edges_file, feature_file)
    if split_by_ui:
        (u_tr, i_tr, t_tr), (u_val, i_val, t_val), (u_te, i_te, t_te) = split_edges_by_ui(
            u, i, ts, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
    else:
        (u_tr, i_tr, t_tr), (u_val, i_val, t_val), (u_te, i_te, t_te) = split_edges_by_time(
            u, i, ts, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )

    # 全量时序图用于 TGN（按时间排序）
    data_full = build_temporal_data(u, i, ts)
    data_full = data_full.to(device)
    train_data, val_data, test_data = data_full.train_val_test_split(
        val_ratio=val_ratio, test_ratio=test_ratio
    )

    train_loader = TemporalDataLoader(train_data, batch_size=batch_size, neg_sampling_ratio=1.0)
    val_loader = TemporalDataLoader(val_data, batch_size=batch_size, neg_sampling_ratio=1.0)
    test_loader = TemporalDataLoader(test_data, batch_size=batch_size, neg_sampling_ratio=1.0)
    neighbor_loader = LastNeighborLoader(data_full.num_nodes, size=10, device=device)

    msg_dim = data_full.msg.size(-1)
    memory = TGNMemory(
        data_full.num_nodes,
        msg_dim,
        memory_dim,
        time_dim,
        message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)
    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=msg_dim,
        time_enc=memory.time_enc,
    ).to(device)
    link_pred = LinkPredictor(in_channels=embedding_dim).to(device)
    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()), lr=lr
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    assoc = torch.empty(data_full.num_nodes, dtype=torch.long, device=device)

    def train_epoch():
        memory.train()
        gnn.train()
        link_pred.train()
        memory.reset_state()
        neighbor_loader.reset_state()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            n_id, edge_index, e_id = neighbor_loader(batch.n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, data_full.t[e_id].to(device), data_full.msg[e_id].to(device))
            pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
            neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])
            loss = criterion(pos_out, torch.ones_like(pos_out)) + criterion(neg_out, torch.zeros_like(neg_out))
            loss.backward()
            optimizer.step()
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)
            memory.detach()
            total_loss += float(loss) * batch.num_events
        return total_loss / train_data.num_events

    @torch.no_grad()
    def evaluate(loader):
        memory.eval()
        gnn.eval()
        link_pred.eval()
        torch.manual_seed(12345)
        aps, aucs, accs = [], [], []
        for batch in loader:
            batch = batch.to(device)
            n_id, edge_index, e_id = neighbor_loader(batch.n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, data_full.t[e_id].to(device), data_full.msg[e_id].to(device))
            pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
            neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])
            y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu().numpy()
            y_true = np.concatenate([np.ones(pos_out.size(0)), np.zeros(neg_out.size(0))])
            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))
            accs.append(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)
        return np.mean(aps), np.mean(aucs), np.mean(accs)

    for epoch in range(1, epochs + 1):
        loss = train_epoch()
        val_ap, val_auc, val_acc = evaluate(val_loader)
        if (epoch % 10 == 0) or epoch == 1:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")

    train_ap, train_auc, train_acc = evaluate(train_loader)
    val_ap, val_auc, val_acc = evaluate(val_loader)
    test_ap, test_auc, test_acc = evaluate(test_loader)

    results = {
        'Train_AUC': train_auc, 'Train_AP': train_ap, 'Train_Acc': train_acc,
        'Val_AUC': val_auc, 'Val_AP': val_ap, 'Val_Acc': val_acc,
        'Test_AUC': test_auc, 'Test_AP': test_ap, 'Test_Acc': test_acc,
    }
    out_dir = os.path.join(code_root, 'result', 'link', data_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'link_tgn_{data_name}.csv')
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"TGN 结果已保存: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='TGN 链路预测')
    parser.add_argument('--data_dir', type=str, default=None, help='如 data_abc_0.01')
    parser.add_argument('--list_data', action='store_true', help='列出可用数据目录')
    parser.add_argument('--split_by_ui', action='store_true', help='按 (u,i) 划分')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    if args.list_data:
        for d in list_available_data(CODE_ROOT):
            print(d)
        return

    edges_file, feature_file, data_name = get_link_data_paths(args.data_dir, CODE_ROOT)
    if not edges_file:
        print("错误: 未指定或找不到 data_dir，请使用 --data_dir 或 --list_data")
        return
    run_tgn(
        edges_file, feature_file, data_name, CODE_ROOT,
        split_by_ui=args.split_by_ui, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )


if __name__ == '__main__':
    main()
