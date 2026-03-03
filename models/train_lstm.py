"""
LSTM 训练脚本 - OULAD 数据集
使用 data/ml_oulad.csv 和节点特征文件进行训练
将图数据转换为时间序列格式
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, log_loss
import os
import argparse
from tqdm import tqdm

class LSTMDataset(Dataset):
    """LSTM数据集类"""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.labels[idx]])

class LSTMModel(nn.Module):
    """LSTM模型"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 全连接层
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # 全连接层
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

def load_node_features(feature_path):
    """加载节点特征文件"""
    print(f"正在加载节点特征: {feature_path}")
    features = []
    with open(feature_path, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="加载特征"):
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


def build_sequences(edges_df, node_features, max_seq_len=50, min_seq_len=1):
    """
    构建时间序列数据
    为每个学生-课程对构建历史交互序列。
    若边表为增强版（含 emb_u_* / emb_i_*），每个时间步会拼接该边的 embedding。
    """
    print("正在构建时间序列...")
    emb_u_cols, emb_i_cols = _get_embedding_columns(edges_df)
    if emb_u_cols is not None:
        print(f"检测到增强表：每个时间步拼接 emb_u ({len(emb_u_cols)} 维) + emb_i ({len(emb_i_cols)} 维)")

    sequences = []
    labels = []
    skipped_short = 0
    skipped_invalid = 0

    max_ts = edges_df['ts'].max()
    min_ts = edges_df['ts'].min()

    groups = list(edges_df.groupby(['u', 'i']))
    for (u, i), group in tqdm(groups, desc="构建序列"):
        group = group.sort_values('ts').reset_index(drop=True)
        label = group['label'].iloc[0]
        u_idx = int(u) - 1
        i_idx = int(i) - 1

        if u_idx < 0 or u_idx >= len(node_features) or i_idx < 0 or i_idx >= len(node_features):
            skipped_invalid += 1
            continue

        seq = []
        for _, row in group.iterrows():
            ts = row['ts']
            student_feat = node_features[u_idx]
            course_feat = node_features[i_idx]
            ts_norm = (ts - min_ts) / (max_ts - min_ts + 1e-8)

            # 基础特征：节点特征 + 时间戳
            parts = [student_feat, course_feat, [ts_norm]]
            # 增强表：该时间步（该条边）的图 embedding
            if emb_u_cols is not None:
                emb_u = np.array([row[c] for c in emb_u_cols], dtype=np.float32)
                emb_i = np.array([row[c] for c in emb_i_cols], dtype=np.float32)
                parts = [student_feat, course_feat, emb_u, emb_i, [ts_norm]]

            feature_vec = np.concatenate(parts)
            seq.append(feature_vec)

        if len(seq) < min_seq_len:
            skipped_short += 1
            continue

        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]
        else:
            padding = np.zeros((max_seq_len - len(seq), seq[0].shape[0]))
            seq = padding.tolist() + seq

        sequences.append(np.array(seq))
        labels.append(1 if label == 1 else 0)

    print(f"构建了 {len(sequences)} 个序列")
    print(f"跳过的序列: 太短={skipped_short}, 无效节点={skipped_invalid}")
    print(f"序列长度: {max_seq_len}")
    print(f"特征维度: {sequences[0].shape[1] if len(sequences) > 0 else 0}")
    print(f"标签分布: 正样本={np.sum(labels)}, 负样本={len(labels)-np.sum(labels)}")

    return sequences, labels

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for sequences, labels in tqdm(dataloader, desc="Training"):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # 计算指标
    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    preds_binary = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_binary)
    
    return avg_loss, auc, ap, acc

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Evaluating"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # 计算指标
    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    preds_binary = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_binary)
    
    return avg_loss, auc, ap, acc

def train_lstm(X_train, y_train, X_val, y_val, X_test, y_test,
               input_dim, hidden_dim=128, num_layers=2, dropout=0.2,
               batch_size=32, num_epochs=50, learning_rate=0.001,
               pos_weight=None, output_dir='./results', results_filename='lstm_results.csv',
               model_filename='lstm_model.pth', device='cuda'):
    """训练LSTM模型"""
    print("\n开始训练LSTM模型...")
    
    # 创建数据集
    train_dataset = LSTMDataset(X_train, y_train)
    val_dataset = LSTMDataset(X_val, y_val)
    test_dataset = LSTMDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数（处理类别不平衡）
    if pos_weight is None:
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos]).to(device)
        else:
            pos_weight = torch.tensor([1.0]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    print(f"正负样本权重: {pos_weight.item():.4f}")
    print(f"学习率: {learning_rate}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {num_epochs}")
    
    # 训练循环
    best_val_auc = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in tqdm(range(num_epochs), desc="训练进度"):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_auc, train_ap, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_auc, val_ap, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, AP: {train_ap:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, AP: {val_ap:.4f}, Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, model_filename))
            print(f"✓ 保存最佳模型 (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发 (patience={patience})")
                break
    
    # 加载最佳模型并测试
    print(f"\n加载最佳模型 (Epoch {best_epoch})...")
    model.load_state_dict(torch.load(os.path.join(output_dir, model_filename)))
    
    test_loss, test_auc, test_ap, test_acc = evaluate(
        model, test_loader, criterion, device
    )
    
    # 训练集最终结果
    train_loss_final, train_auc_final, train_ap_final, train_acc_final = evaluate(
        model, train_loader, criterion, device
    )
    
    # 验证集最终结果
    val_loss_final, val_auc_final, val_ap_final, val_acc_final = evaluate(
        model, val_loader, criterion, device
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("最终结果:")
    print("="*50)
    print(f"训练集 - Loss: {train_loss_final:.4f}, AUC: {train_auc_final:.4f}, AP: {train_ap_final:.4f}, Acc: {train_acc_final:.4f}")
    print(f"验证集 - Loss: {val_loss_final:.4f}, AUC: {val_auc_final:.4f}, AP: {val_ap_final:.4f}, Acc: {val_acc_final:.4f}")
    print(f"测试集 - Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, AP: {test_ap:.4f}, Acc: {test_acc:.4f}")
    print("="*50)
    
    # 保存结果
    results = {
        'Train_Loss': train_loss_final,
        'Train_AUC': train_auc_final,
        'Train_AP': train_ap_final,
        'Train_Acc': train_acc_final,
        'Val_Loss': val_loss_final,
        'Val_AUC': val_auc_final,
        'Val_AP': val_ap_final,
        'Val_Acc': val_acc_final,
        'Test_Loss': test_loss,
        'Test_AUC': test_auc,
        'Test_AP': test_ap,
        'Test_Acc': test_acc,
        'pos_weight': pos_weight.item(),
        'best_epoch': best_epoch,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout
    }
    
    results_df = pd.DataFrame([results])
    results_path = os.path.join(output_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"\n结果已保存: {results_path}")
    
    return model, results

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

def main():
    parser = argparse.ArgumentParser(description='LSTM训练脚本')
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
                       help='输出目录。未指定且指定了 --data_dir 时，保存到 code/result/，结果文件名为 lstm_results_<data_dir>.csv')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='数据采样比例（用于快速测试，如果已使用采样数据则设为1.0）')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--max_seq_len', type=int, default=50,
                       help='最大序列长度')
    parser.add_argument('--min_seq_len', type=int, default=1,
                       help='最小序列长度')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout率')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_root = os.path.dirname(script_dir)

    # 如果请求列出可用数据，则列出并退出
    if args.list_data:
        print("="*50)
        print("可用的数据文件夹 (code/data/all_data/):")
        print("="*50)
        available_dirs = list_available_data(script_dir)
        if available_dirs:
            for i, data_dir in enumerate(available_dirs, 1):
                data_path = get_data_dir_path(code_root, data_dir)
                if data_path is None:
                    data_path = os.path.join(code_root, 'data', 'all_data', data_dir)
                edges_file = os.path.join(data_path, 'ml_oulad.csv')
                if os.path.exists(edges_file):
                    edges_df = pd.read_csv(edges_file)
                    print(f"{i}. {data_dir} - {len(edges_df):,} 条边")
                else:
                    print(f"{i}. {data_dir}")
            print("="*50)
            print(f"\n使用方法: python train_lstm.py --data_dir <文件夹名>")
            print(f"例如: python train_lstm.py --data_dir data_0.1")
            print(f"      python train_lstm.py --data_dir enhance_data_abc_0.01")
        else:
            print("没有找到可用的数据文件夹")
        return
    if args.data_dir:
        data_dir_path = get_data_dir_path(code_root, args.data_dir)
        if data_dir_path is None:
            data_dir_path = os.path.join(code_root, 'data', 'all_data', args.data_dir)
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
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 输出目录与结果文件名：指定了 data_dir 时输出到 result/data_dir/，如 result/data_0.1/lstm_results_data_0.1.csv
    if args.data_dir:
        args.results_filename = f'lstm_results_{args.data_dir}.csv'
        args.model_filename = f'lstm_model_{args.data_dir}.pth'
        if args.output_dir is None:
            args.output_dir = os.path.join(code_root, 'result', args.data_dir)
    else:
        args.results_filename = 'lstm_results.csv'
        args.model_filename = 'lstm_model.pth'
    if args.output_dir is None:
        args.output_dir = './'

    print("="*50)
    print("LSTM 训练脚本 - OULAD 数据集")
    print("="*50)
    if args.data_dir:
        data_dir_path = get_data_dir_path(code_root, args.data_dir)
        print(f"数据目录: {data_dir_path or args.data_dir}")
    print(f"边数据路径: {data_path}")
    print(f"节点特征路径: {feature_path}")
    print(f"测试集比例: {args.test_size}")
    print(f"验证集比例: {args.val_size}")
    print(f"采样比例: {args.sample_ratio}")
    print(f"最大序列长度: {args.max_seq_len}")
    print(f"设备: {args.device}")
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
        # 按学生-课程对采样
        unique_pairs = edges_df[['u', 'i']].drop_duplicates()
        sample_pairs = unique_pairs.sample(frac=args.sample_ratio, random_state=args.random_state)
        edges_df = edges_df.merge(sample_pairs, on=['u', 'i'], how='inner')
        print(f"采样后边数据形状: {edges_df.shape}")
    
    # 加载节点特征
    node_features = load_node_features(feature_path)
    
    # 构建序列
    print("\n2. 构建时间序列...")
    sequences, labels = build_sequences(
        edges_df, node_features,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len
    )
    
    if len(sequences) == 0:
        print("错误: 没有构建出有效的序列！")
        return
    
    # 获取特征维度
    input_dim = sequences[0].shape[1]
    print(f"输入特征维度: {input_dim}")
    
    # 划分数据集
    print("\n3. 划分数据集...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=args.test_size + args.val_size,
        random_state=args.random_state, stratify=labels
    )
    
    val_size_adjusted = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_size_adjusted,
        random_state=args.random_state, stratify=y_temp
    )
    
    print(f"训练集: {len(X_train)} 序列")
    print(f"验证集: {len(X_val)} 序列")
    print(f"测试集: {len(X_test)} 序列")
    
    # 训练模型
    print("\n4. 训练模型...")
    model, results = train_lstm(
        X_train, y_train, X_val, y_val, X_test, y_test,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        results_filename=args.results_filename,
        model_filename=args.model_filename,
        device=args.device
    )
    
    print("\n训练完成！")

if __name__ == '__main__':
    main()

