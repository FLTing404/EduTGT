"""
XGBoost 训练脚本 - OULAD 数据集
使用 data/ml_oulad.csv 和节点特征文件进行训练
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, log_loss
import os
import argparse
from datetime import datetime
from tqdm import tqdm

def load_node_features(feature_path):
    """加载节点特征文件"""
    print(f"正在加载节点特征: {feature_path}")
    features = []
    # 先统计总行数
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
    """
    准备XGBoost特征
    每条边对应一个样本，特征包括：
    - 学生节点特征（16维）+ 课程节点特征（16维）（来自 oulad.content）
    - 若边表为增强版：emb_u_* + emb_i_*（图模型 embedding，每边一条）
    - 时间戳 + 统计特征（交互次数等）
    """
    print("正在准备特征...")
    emb_u_cols, emb_i_cols = _get_embedding_columns(edges_df)
    if emb_u_cols is not None:
        print(f"检测到增强表：使用 emb_u ({len(emb_u_cols)} 维) + emb_i ({len(emb_i_cols)} 维)")
    
    # 注意：节点ID从1开始，但数组索引从0开始，需要减1
    X_list = []
    y_list = []
    pair_list = []  # 每条边对应的 (u, i)，用于按 (u,i) 划分避免泄露
    
    # 统计每个学生-课程对的交互次数（用于特征工程）
    student_course_counts = edges_df.groupby(['u', 'i']).size().reset_index(name='interaction_count')
    edges_df = edges_df.merge(student_course_counts, on=['u', 'i'], how='left')
    
    # 统计每个学生的时间戳范围
    student_ts_stats = edges_df.groupby('u')['ts'].agg(['min', 'max', 'count']).reset_index()
    student_ts_stats.columns = ['u', 'ts_min', 'ts_max', 'ts_count']
    edges_df = edges_df.merge(student_ts_stats, on='u', how='left')
    
    # 统计每个课程的时间戳范围
    course_ts_stats = edges_df.groupby('i')['ts'].agg(['min', 'max', 'count']).reset_index()
    course_ts_stats.columns = ['i', 'course_ts_min', 'course_ts_max', 'course_ts_count']
    edges_df = edges_df.merge(course_ts_stats, on='i', how='left')
    
    for idx, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="准备特征", unit="边"):
        u = int(row['u']) - 1  # 转换为0-based索引
        i = int(row['i']) - 1  # 转换为0-based索引
        ts = row['ts']
        label = row['label']
        
        # 检查节点ID是否在有效范围内（仅在使用 oulad.content 节点特征时需要）
        if u < 0 or u >= len(node_features) or i < 0 or i >= len(node_features):
            continue
        
        # 基础特征：节点特征（来自 oulad.content）
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
        # 若为增强表，追加该边的图 embedding（emb_u + emb_i）
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
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签分布: 正样本={np.sum(y==1)}, 负样本={np.sum(y==0)}")
    print(f"唯一 (u,i) 数: {len(edges_df.drop_duplicates(subset=['u','i']))}")
    
    return X, y, pairs

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, 
                  pos_weight=None, output_dir='./results', results_filename='xgboost_results.csv'):
    """训练XGBoost模型"""
    print("\n开始训练XGBoost模型...")
    
    # 计算正负样本权重
    if pos_weight is None:
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        if n_pos > 0 and n_neg > 0:
            pos_weight = n_neg / n_pos
        else:
            pos_weight = 1.0
    
    print(f"正负样本权重: {pos_weight:.4f}")
    
    # XGBoost参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': pos_weight,  # 处理类别不平衡
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"模型参数: {params}")
    
    # 创建DMatrix（XGBoost的数据格式）
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 训练模型（兼容 XGBoost 1.6+：已移除旧版 callback，改用 verbose_eval）
    print("\n开始训练XGBoost模型...")
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10,
    )
    
    # 预测
    print("\n正在评估模型...")
    y_train_pred = model.predict(dtrain)
    y_val_pred = model.predict(dval)
    y_test_pred = model.predict(dtest)
    
    # 计算指标
    def compute_metrics(y_true, y_pred, prefix=""):
        y_pred_binary = (y_pred > 0.5).astype(int)
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred_binary)
        loss = log_loss(y_true, y_pred)
        return {
            f'{prefix}AUC': auc,
            f'{prefix}AP': ap,
            f'{prefix}Acc': acc,
            f'{prefix}Loss': loss
        }
    
    train_metrics = compute_metrics(y_train, y_train_pred, "Train_")
    val_metrics = compute_metrics(y_val, y_val_pred, "Val_")
    test_metrics = compute_metrics(y_test, y_test_pred, "Test_")
    
    # 打印结果
    print("\n" + "="*50)
    print("训练集结果:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n验证集结果:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n测试集结果:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("="*50)
    
    # 保存结果（只保存CSV，不保存JSON模型）
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        **train_metrics,
        **val_metrics,
        **test_metrics,
        'pos_weight': pos_weight,
        'n_estimators': model.best_iteration if hasattr(model, 'best_iteration') else params['n_estimators']
    }
    
    results_df = pd.DataFrame([results])
    results_path = os.path.join(output_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"\n结果已保存: {results_path}")
    return model, results

def get_data_dir_path(code_root, data_dir):
    """将 data_dir 解析为实际路径：支持 all_data/<name>、process_data/<name>、process_data/<base>/<name>（如 data_0.1_train）。"""
    process_data_dir = os.path.join(code_root, 'process_data')
    if os.path.exists(process_data_dir):
        for base in os.listdir(process_data_dir):
            candidate = os.path.join(process_data_dir, base, data_dir)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, 'ml_oulad.csv')):
                return candidate
    cand = os.path.join(process_data_dir, data_dir)
    if os.path.exists(cand) and os.path.exists(os.path.join(cand, 'ml_oulad.csv')):
        return cand
    cand = os.path.join(code_root, 'all_data', data_dir)
    if os.path.exists(cand):
        return cand
    return None

def list_available_data(script_dir):
    """列出所有可用的数据文件夹（all_data 一级 + process_data 一级 + process_data/<base>/ 下子目录）。"""
    code_root = os.path.dirname(script_dir)
    all_data_dir = os.path.join(code_root, 'all_data')
    process_data_dir = os.path.join(code_root, 'process_data')
    available_dirs = []
    if os.path.exists(all_data_dir):
        for item in os.listdir(all_data_dir):
            item_path = os.path.join(all_data_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'ml_oulad.csv')):
                available_dirs.append(item)
    if os.path.exists(process_data_dir):
        for base in os.listdir(process_data_dir):
            base_path = os.path.join(process_data_dir, base)
            if not os.path.isdir(base_path):
                continue
            for sub in os.listdir(base_path):
                sub_path = os.path.join(base_path, sub)
                if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, 'ml_oulad.csv')):
                    available_dirs.append(sub)
            if os.path.exists(os.path.join(base_path, 'ml_oulad.csv')):
                if base not in available_dirs:
                    available_dirs.append(base)
    return sorted(set(available_dirs))

def main():
    parser = argparse.ArgumentParser(description='XGBoost训练脚本')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录名称，如 data_0_1 或 data_1，如果指定则使用 all_data/{data_dir}/ 下的数据')
    parser.add_argument('--list_data', action='store_true',
                       help='列出所有可用的数据文件夹')
    parser.add_argument('--data_path', type=str, default=None,
                       help='边数据文件路径（如果指定data_dir则忽略）')
    parser.add_argument('--feature_path', type=str, default=None,
                       help='节点特征文件路径（如果指定data_dir则忽略）')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='验证集比例（从训练集中划分）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录。未指定且指定了 --data_dir 时，保存到 code/result/，结果文件名为 xgboost_results_<data_dir>.csv')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='数据采样比例（用于快速测试，如果已使用采样数据则设为1.0）')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 根据 data_dir 自动设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 如果请求列出可用数据，则列出并退出
    if args.list_data:
        print("="*50)
        print("可用的数据文件夹 (all_data/ 与 process_data/):")
        print("="*50)
        available_dirs = list_available_data(script_dir)
        code_root = os.path.dirname(script_dir)
        if available_dirs:
            for i, data_dir in enumerate(available_dirs, 1):
                data_path = get_data_dir_path(code_root, data_dir)
                if data_path is None:
                    data_path = os.path.join(code_root, 'all_data', data_dir)
                edges_file = os.path.join(data_path, 'ml_oulad.csv')
                if os.path.exists(edges_file):
                    edges_df = pd.read_csv(edges_file)
                    print(f"{i}. {data_dir} - {len(edges_df):,} 条边")
                else:
                    print(f"{i}. {data_dir}")
            print("="*50)
            print(f"\n使用方法: python train_xgboost.py --data_dir <文件夹名>")
            print(f"例如: python train_xgboost.py --data_dir data_0.1")
            print(f"      python train_xgboost.py --data_dir enhance_data_abc_0.01")
        else:
            print("没有找到可用的数据文件夹")
        return
    code_root = os.path.dirname(script_dir)
    if args.data_dir:
        data_dir_path = get_data_dir_path(code_root, args.data_dir)
        if data_dir_path is None:
            data_dir_path = os.path.join(code_root, 'all_data', args.data_dir)
        data_path = os.path.join(data_dir_path, 'ml_oulad.csv')
        feature_path = os.path.join(data_dir_path, 'oulad.content')
    else:
        # 使用默认路径或用户指定的路径
        if args.data_path:
            data_path = args.data_path
        else:
            data_path = os.path.join(code_root, 'data', 'ml_oulad.csv')
        
        if args.feature_path:
            feature_path = args.feature_path
        else:
            feature_path = os.path.join(code_root, 'ContraTGT', 'node_feature', 'oulad.content')
    
    print("="*50)
    print("XGBoost 训练脚本 - OULAD 数据集")
    # 输出目录与结果文件名：指定了 data_dir 时结果文件名带 data_dir，默认输出到 code/result/
    if args.data_dir:
        args.results_filename = f'xgboost_results_{args.data_dir}.csv'
        if args.output_dir is None:
            args.output_dir = os.path.join(code_root, 'result')
    else:
        args.results_filename = 'xgboost_results.csv'
    if args.output_dir is None:
        args.output_dir = './'

    print("="*50)
    if args.data_dir:
        data_dir_path = get_data_dir_path(code_root, args.data_dir)
        print(f"数据目录: {data_dir_path or args.data_dir}")
    print(f"边数据路径: {data_path}")
    print(f"节点特征路径: {feature_path}")
    print(f"测试集比例: {args.test_size}")
    print(f"验证集比例: {args.val_size}")
    print(f"采样比例: {args.sample_ratio}")
    print("="*50)
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 找不到边数据文件 {data_path}")
        return
    if not os.path.exists(feature_path):
        print(f"错误: 找不到节点特征文件 {feature_path}")
        return
    
    # 加载数据
    print("\n1. 加载数据...")
    edges_df = pd.read_csv(data_path)
    print(f"边数据形状: {edges_df.shape}")
    
    # 数据采样（如果指定，且不是使用已采样的数据）
    if args.sample_ratio < 1.0:
        sample_size = int(len(edges_df) * args.sample_ratio)
        edges_df = edges_df.sample(n=sample_size, random_state=args.random_state).reset_index(drop=True)
        print(f"采样后边数据形状: {edges_df.shape}")
    
    # 加载节点特征
    node_features = load_node_features(feature_path)
    
    # 准备特征
    print("\n2. 准备特征...")
    X, y, pairs = prepare_features(edges_df, node_features)
    
    # 划分数据集：按 (u,i) 划分，避免同一 (u,i) 既在 train 又在 test 导致的数据泄露
    print("\n3. 划分数据集（按学生-课程对划分，防泄露）...")
    unique_ui = edges_df[['u', 'i', 'label']].drop_duplicates(subset=['u', 'i'])
    unique_ui = unique_ui.reset_index(drop=True)
    y_ui = (unique_ui['label'] == 1).astype(int).values
    ui_train, ui_temp, y_ui_train, y_ui_temp = train_test_split(
        unique_ui[['u', 'i']], y_ui,
        test_size=args.test_size + args.val_size,
        random_state=args.random_state, stratify=y_ui
    )
    val_size_adjusted = args.val_size / (args.test_size + args.val_size)
    ui_val, ui_test, y_ui_val, y_ui_test = train_test_split(
        ui_temp, y_ui_temp, test_size=1 - val_size_adjusted,
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
    
    print(f"唯一 (u,i): 训练 {len(train_ui_set)}, 验证 {len(val_ui_set)}, 测试 {len(test_ui_set)}")
    print(f"训练集: {X_train.shape[0]} 样本（边）")
    print(f"验证集: {X_val.shape[0]} 样本（边）")
    print(f"测试集: {X_test.shape[0]} 样本（边）")
    
    # 训练模型
    print("\n4. 训练模型...")
    model, results = train_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test,
        output_dir=args.output_dir,
        results_filename=args.results_filename
    )
    
    print("\n训练完成！")

if __name__ == '__main__':
    main()

