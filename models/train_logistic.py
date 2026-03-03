"""
逻辑回归训练脚本 - OULAD 数据集
与 train_xgboost 相同的数据与划分：边表 + 节点特征（可选增强表 emb），按 (u,i) 划分防泄露。
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
        total_lines = sum(1 for _ in f)
    with open(feature_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="加载特征", unit="行"):
            line = line.strip()
            if line:
                feature_vec = [float(x) for x in line.split(',')]
                features.append(feature_vec)
    features = np.array(features)
    print(f"节点特征形状: {features.shape}")
    return features


def _get_embedding_columns(edges_df):
    """若边表为增强版（含 emb_u_* / emb_i_*），返回 (emb_u 列列表, emb_i 列列表)，否则 (None, None)。"""
    emb_u_cols = sorted([c for c in edges_df.columns if c.startswith('emb_u_')],
                        key=lambda x: int(x.replace('emb_u_', '')))
    emb_i_cols = sorted([c for c in edges_df.columns if c.startswith('emb_i_')],
                        key=lambda x: int(x.replace('emb_i_', '')))
    if emb_u_cols and emb_i_cols:
        return emb_u_cols, emb_i_cols
    return None, None


def prepare_features(edges_df, node_features):
    """与 train_xgboost 一致：每条边一个样本，特征为节点特征 + 可选 emb + 时间戳与统计量。"""
    print("正在准备特征...")
    emb_u_cols, emb_i_cols = _get_embedding_columns(edges_df)
    if emb_u_cols is not None:
        print(f"检测到增强表：使用 emb_u ({len(emb_u_cols)} 维) + emb_i ({len(emb_i_cols)} 维)")

    X_list = []
    y_list = []
    pair_list = []

    student_course_counts = edges_df.groupby(['u', 'i']).size().reset_index(name='interaction_count')
    edges_df = edges_df.merge(student_course_counts, on=['u', 'i'], how='left')
    student_ts_stats = edges_df.groupby('u')['ts'].agg(['min', 'max', 'count']).reset_index()
    student_ts_stats.columns = ['u', 'ts_min', 'ts_max', 'ts_count']
    edges_df = edges_df.merge(student_ts_stats, on='u', how='left')
    course_ts_stats = edges_df.groupby('i')['ts'].agg(['min', 'max', 'count']).reset_index()
    course_ts_stats.columns = ['i', 'course_ts_min', 'course_ts_max', 'course_ts_count']
    edges_df = edges_df.merge(course_ts_stats, on='i', how='left')

    for idx, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="准备特征", unit="边"):
        u = int(row['u']) - 1
        i = int(row['i']) - 1
        ts = row['ts']
        label = row['label']
        if u < 0 or u >= len(node_features) or i < 0 or i >= len(node_features):
            continue
        student_feat = node_features[u]
        course_feat = node_features[i]
        base_parts = [
            student_feat,
            course_feat,
            [ts],
            [row.get('interaction_count', 0)],
            [row.get('ts_count', 0)],
            [row.get('course_ts_count', 0)],
        ]
        if emb_u_cols is not None:
            emb_u = np.array([row[c] for c in emb_u_cols], dtype=np.float32)
            emb_i = np.array([row[c] for c in emb_i_cols], dtype=np.float32)
            base_parts = [student_feat, course_feat, emb_u, emb_i] + base_parts[2:]
        feature_vec = np.concatenate(base_parts)
        X_list.append(feature_vec)
        y_list.append(1 if label == 1 else 0)
        pair_list.append((int(row['u']), int(row['i'])))

    X = np.array(X_list)
    y = np.array(y_list)
    pairs = np.array(pair_list)
    print(f"特征矩阵形状: {X.shape}, 标签分布: 正={np.sum(y==1)}, 负={np.sum(y==0)}")
    return X, y, pairs


def get_data_dir_path(code_root, data_dir):
    """仅从 code/data/all_data 解析数据目录。"""
    all_data_dir = os.path.join(code_root, 'data', 'all_data')
    cand = os.path.join(all_data_dir, data_dir)
    if os.path.isdir(cand) and os.path.exists(os.path.join(cand, 'ml_oulad.csv')):
        return cand
    return None


def list_available_data(script_dir):
    """列出 code/data/all_data 下所有可用的数据文件夹。"""
    code_root = os.path.dirname(script_dir)
    all_data_dir = os.path.join(code_root, 'data', 'all_data')
    available_dirs = []
    if os.path.exists(all_data_dir):
        for item in os.listdir(all_data_dir):
            item_path = os.path.join(all_data_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'ml_oulad.csv')):
                available_dirs.append(item)
    return sorted(available_dirs)


def train_logistic(X_train, y_train, X_val, y_val, X_test, y_test,
                   output_dir='./', results_filename='logistic_results.csv',
                   C=1.0, max_iter=1000, random_state=42):
    """训练逻辑回归并评估，保存与 XGBoost 同格式的 CSV。"""
    print("\n开始训练逻辑回归...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # n_jobs=1 避免 Windows 下 joblib/loky 多进程报错（_winapi.SYNCHRONIZE 等）
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=1,
        solver='lbfgs',
    )
    model.fit(X_train_s, y_train)

    def compute_metrics(y_true, y_pred_proba, prefix=""):
        y_pred = (y_pred_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)
        acc = accuracy_score(y_true, y_pred)
        loss = log_loss(y_true, y_pred_proba)
        return {f'{prefix}AUC': auc, f'{prefix}AP': ap, f'{prefix}Acc': acc, f'{prefix}Loss': loss}

    y_train_p = model.predict_proba(X_train_s)[:, 1]
    y_val_p = model.predict_proba(X_val_s)[:, 1]
    y_test_p = model.predict_proba(X_test_s)[:, 1]

    train_metrics = compute_metrics(y_train, y_train_p, "Train_")
    val_metrics = compute_metrics(y_val, y_val_p, "Val_")
    test_metrics = compute_metrics(y_test, y_test_p, "Test_")

    print("\n" + "="*50)
    print("训练集:", train_metrics)
    print("验证集:", val_metrics)
    print("测试集:", test_metrics)
    print("="*50)

    os.makedirs(output_dir, exist_ok=True)
    results = {**train_metrics, **val_metrics, **test_metrics, 'C': C, 'max_iter': max_iter}
    results_df = pd.DataFrame([results])
    results_path = os.path.join(output_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"结果已保存: {results_path}")
    return model, scaler, results


def main():
    parser = argparse.ArgumentParser(description='逻辑回归训练脚本 - OULAD')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录名，如 data_abc_0.01 或 data_abc_0.01_enhanced')
    parser.add_argument('--list_data', action='store_true', help='列出可用数据目录后退出')
    parser.add_argument('--data_path', type=str, default=None, help='边表路径（未指定 data_dir 时用）')
    parser.add_argument('--feature_path', type=str, default=None, help='节点特征路径（未指定 data_dir 时用）')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--output_dir', type=str, default=None, help='结果目录，默认 code/result/')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='边采样比例')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--C', type=float, default=1.0, help='正则化强度')
    parser.add_argument('--max_iter', type=int, default=1000, help='最大迭代次数')
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
        print("\n用法: python models/train_logistic.py --data_dir <目录名>")
        return

    if args.data_dir:
        data_dir_path = get_data_dir_path(code_root, args.data_dir) or os.path.join(code_root, 'data', 'all_data', args.data_dir)
        data_path = os.path.join(data_dir_path, 'ml_oulad.csv')
        feature_path = os.path.join(data_dir_path, 'oulad.content')
        results_filename = f'logistic_results_{args.data_dir}.csv'
        # 结果按 data_dir 分子目录：result/data_0.1/logistic_results_data_0.1.csv
        output_dir = args.output_dir or os.path.join(code_root, 'result', args.data_dir)
    else:
        data_path = args.data_path or os.path.join(code_root, 'data', 'ml_oulad.csv')
        feature_path = args.feature_path or os.path.join(code_root, 'ContraTGT', 'node_feature', 'oulad.content')
        results_filename = 'logistic_results.csv'
        output_dir = args.output_dir or './'

    if not os.path.exists(data_path):
        print(f"错误: 找不到边数据 {data_path}")
        return
    if not os.path.exists(feature_path):
        print(f"错误: 找不到节点特征 {feature_path}")
        return

    print("逻辑回归 - OULAD")
    print(f"边数据: {data_path}")
    print(f"特征: {feature_path}")

    edges_df = pd.read_csv(data_path)
    if args.sample_ratio < 1.0:
        edges_df = edges_df.sample(frac=args.sample_ratio, random_state=args.random_state).reset_index(drop=True)
    node_features = load_node_features(feature_path)
    X, y, pairs = prepare_features(edges_df, node_features)

    unique_ui = edges_df[['u', 'i', 'label']].drop_duplicates(subset=['u', 'i']).reset_index(drop=True)
    y_ui = (unique_ui['label'] == 1).astype(int).values
    ui_train, ui_temp, y_ui_train, y_ui_temp = train_test_split(
        unique_ui[['u', 'i']], y_ui,
        test_size=args.test_size + args.val_size,
        random_state=args.random_state, stratify=y_ui
    )
    val_size_adj = args.val_size / (args.test_size + args.val_size)
    ui_val, ui_test, y_ui_val, y_ui_test = train_test_split(
        ui_temp, y_ui_temp, test_size=1 - val_size_adj,
        random_state=args.random_state, stratify=y_ui_temp
    )
    train_ui_set = set(map(tuple, ui_train.values))
    val_ui_set = set(map(tuple, ui_val.values))
    test_ui_set = set(map(tuple, ui_test.values))
    train_mask = np.array([(p[0], p[1]) in train_ui_set for p in pairs])
    val_mask = np.array([(p[0], p[1]) in val_ui_set for p in pairs])
    test_mask = np.array([(p[0], p[1]) in test_ui_set for p in pairs])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"训练 {X_train.shape[0]}, 验证 {X_val.shape[0]}, 测试 {X_test.shape[0]} 样本")

    train_logistic(
        X_train, y_train, X_val, y_val, X_test, y_test,
        output_dir=output_dir,
        results_filename=results_filename,
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    print("训练完成。")


if __name__ == '__main__':
    main()
