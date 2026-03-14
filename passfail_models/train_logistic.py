"""
课程通过/不通过：Logistic 头。
--mode pure: 原始特征 + 逻辑回归
--mode contratgt: 预训练表示 + 逻辑回归
与 main_passfail 同一 (u,i) 划分、课程级 AUC/AP/Acc。
"""
import os
import sys
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
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
    parser.add_argument('--ablation_suffix', type=str, default='', help='消融时与 pretrain 一致，如 baseline；配合 --seed 加载对应预训练')
    parser.add_argument('--mode', type=str, choices=['pure', 'contratgt'], required=True)
    parser.add_argument('--C', type=float, default=1.0, help='LogisticRegression 正则化')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    class A:
        data_dir = args.data_dir
        gpu = args.gpu
    edges_file, feature_file, data_name = get_data_paths(A)
    train_ui, val_ui, test_ui, y_train, y_val, y_test, edge = get_ui_splits_and_labels(
        edges_file, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8, random_state=args.seed)

    if args.mode == 'pure':
        raw = load_features_and_build_raw(feature_file, edge, train_ui, val_ui, test_ui)
        X_train, X_val, X_test = raw['X_train'], raw['X_val'], raw['X_test']
    else:
        import torch
        _ablation_sfx = getattr(args, 'ablation_suffix', '') or ''
        _seed_sfx = ('_seed' + str(args.seed)) if _ablation_sfx else ''
        pretrain_path = os.path.join(CODE_ROOT, 'script', 'pretrain_model', f'{data_name}{"_" + _ablation_sfx if _ablation_sfx else ""}{_seed_sfx}.pth')
        from utils import Dataset
        _, _, _, _, train_data, test_data, val_data, _, _ = Dataset(
            file=edges_file, split_by_ui=False, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8, random_state=args.seed)
        out = run_extract(edges_file, feature_file, pretrain_path, train_data, val_data, test_data,
                          edge, train_ui, val_ui, test_ui, seed=args.seed,
                          device=torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'))
        X_train, X_val, X_test = out['X_train'], out['X_val'], out['X_test']

    clf = LogisticRegression(C=args.C, max_iter=1000, random_state=args.seed, class_weight='balanced')
    clf.fit(X_train, y_train)

    for name, X, y in [('Train', X_train, y_train), ('Val', X_val, y_val), ('Test', X_test, y_test)]:
        prob = clf.predict_proba(X)[:, 1]
        auc, ap, acc, loss = eval_metrics(y, prob)
        print(f'{name} AUC={auc:.4f} AP={ap:.4f} Acc={acc:.4f} Loss={loss:.4f}')

    prob_train = clf.predict_proba(X_train)[:, 1]
    prob_val = clf.predict_proba(X_val)[:, 1]
    prob_test = clf.predict_proba(X_test)[:, 1]
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
    out_path = os.path.join(out_dir, f'logistic_{args.mode}_{data_name}.csv')
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f'结果已保存: {out_path}')


if __name__ == '__main__':
    main()
