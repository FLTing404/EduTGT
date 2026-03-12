"""
课程通过/不通过：LSTM 头。
--mode pure: 原始特征序列 + LSTM
--mode contratgt: 预训练表示序列 + LSTM
与 main_passfail 同一 (u,i) 划分、课程级 AUC/AP/Acc。
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
from extract_embeddings import run_extract
import pandas as pd


class LSTMHead(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        return self.fc(out).squeeze(-1)


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
    parser.add_argument('--mode', type=str, choices=['pure', 'contratgt'], required=True)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    class A:
        data_dir = args.data_dir
        gpu = args.gpu
    edges_file, feature_file, data_name = get_data_paths(A)
    train_ui, val_ui, test_ui, y_train, y_val, y_test, edge = get_ui_splits_and_labels(
        edges_file, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8, random_state=args.seed)

    if args.mode == 'pure':
        raw = load_features_and_build_raw(feature_file, edge, train_ui, val_ui, test_ui, seq_max_len=args.max_len)
        X_train_seq = raw['X_train_seq']
        X_val_seq = raw['X_val_seq']
        X_test_seq = raw['X_test_seq']
        sl_train, sl_val, sl_test = raw['seq_lengths_train'], raw['seq_lengths_val'], raw['seq_lengths_test']
        input_dim = raw['feat_dim_seq']
    else:
        from utils import Dataset
        _, _, _, _, train_data, test_data, val_data, _, _ = Dataset(
            file=edges_file, split_by_ui=False, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8, random_state=args.seed)
        out = run_extract(edges_file, feature_file,
                         os.path.join(CODE_ROOT, 'script', 'pretrain_model', f'{data_name}.pth'),
                         train_data, val_data, test_data, edge, train_ui, val_ui, test_ui,
                         seed=args.seed, seq_max_len=args.max_len, device=device)
        X_train_seq, X_val_seq, X_test_seq = out['X_train_seq'], out['X_val_seq'], out['X_test_seq']
        sl_train, sl_val, sl_test = out['seq_lengths_train'], out['seq_lengths_val'], out['seq_lengths_test']
        input_dim = out['embed_dim']

    X_train_t = torch.FloatTensor(np.array(X_train_seq)).to(device)
    X_val_t = torch.FloatTensor(np.array(X_val_seq)).to(device)
    X_test_t = torch.FloatTensor(np.array(X_test_seq)).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    model = LSTMHead(input_dim, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_ap = 0
    for epoch in range(args.n_epoch):
        model.train()
        logits = model(X_train_t, sl_train)
        loss = criterion(logits, y_train_t.squeeze(1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(X_val_t, sl_val)).cpu().numpy()
        val_auc, val_ap, _, _ = eval_metrics(y_val, p_val)
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob_train = torch.sigmoid(model(X_train_t, sl_train)).cpu().numpy()
        prob_val = torch.sigmoid(model(X_val_t, sl_val)).cpu().numpy()
        prob_test = torch.sigmoid(model(X_test_t, sl_test)).cpu().numpy()

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
    out_path = os.path.join(out_dir, f'lstm_{args.mode}_{data_name}.csv')
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f'结果已保存: {out_path}')


if __name__ == '__main__':
    main()
