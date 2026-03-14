"""
课程通过/不通过：纯 MLP 基线（原始特征 → MLP → 通过/不通过）。
与 main_passfail（ContraTGT + MLP 头）对照。
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, log_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(SCRIPT_DIR)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
if os.path.join(CODE_ROOT, 'script') not in sys.path:
    sys.path.insert(0, os.path.join(CODE_ROOT, 'script'))

from utils import get_data_paths
from passfail_data import get_ui_splits_and_labels, load_features_and_build_raw
import pandas as pd


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LeakyReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def eval_metrics(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        auc, ap = 0.5, 0.5
    else:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    prob_clip = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    loss = log_loss(y_true, prob_clip)
    return auc, ap, acc, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden_dims', type=str, default='64,32')
    parser.add_argument('--n_epoch', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))

    class A:
        data_dir = args.data_dir
        gpu = args.gpu
    edges_file, feature_file, data_name = get_data_paths(A)
    train_ui, val_ui, test_ui, y_train, y_val, y_test, edge = get_ui_splits_and_labels(
        edges_file, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8, random_state=args.seed)
    raw = load_features_and_build_raw(feature_file, edge, train_ui, val_ui, test_ui)
    X_train = torch.FloatTensor(raw['X_train']).to(device)
    X_val = torch.FloatTensor(raw['X_val']).to(device)
    X_test = torch.FloatTensor(raw['X_test']).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    model = MLPHead(raw['feat_dim'], hidden_dims=hidden_dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 正类权重：缓解类别不平衡，避免模型全预测多数类导致 Acc 极低
    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_ap = 0
    for epoch in range(args.n_epoch):
        model.train()
        logits = model(X_train)
        loss = criterion(logits, y_train_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(X_val)).cpu().numpy()
        val_auc, val_ap, _, _ = eval_metrics(y_val, p_val)
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob_train = torch.sigmoid(model(X_train)).cpu().numpy()
        prob_val = torch.sigmoid(model(X_val)).cpu().numpy()
        prob_test = torch.sigmoid(model(X_test)).cpu().numpy()

    train_auc, train_ap, train_acc, train_loss = eval_metrics(y_train, prob_train)
    val_auc, val_ap, val_acc, val_loss = eval_metrics(y_val, prob_val)
    test_auc, test_ap, test_acc, test_loss = eval_metrics(y_test, prob_test)

    results = {
        'Train_AUC': train_auc, 'Train_AP': train_ap, 'Train_Acc': train_acc, 'Train_Loss': train_loss,
        'Val_AUC': val_auc, 'Val_AP': val_ap, 'Val_Acc': val_acc, 'Val_Loss': val_loss,
        'Test_AUC': test_auc, 'Test_AP': test_ap, 'Test_Acc': test_acc, 'Test_Loss': test_loss,
    }
    out_dir = args.output_dir or os.path.join(CODE_ROOT, 'result', 'passfail', data_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'mlp_pure_{data_name}.csv')
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f'结果已保存: {out_path}')


if __name__ == '__main__':
    main()
