import numpy as np
import torch
import torch.nn as nn
import os
import random
import argparse
import sys
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser('Interface for Inductive Dynamic Representation Learning for Link Prediction on Temporal Graphs')

    # select dataset and training mode
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit',
                        choices=['socialevolve_1m', 'wiki', 'slashdot', 'bitcoinotc','ubuntu','oulad'],
                        default='slashdot')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='指定 data/all_data 下子目录名（项目根下），如 data_abc_0.01，将使用该目录下 ml_oulad.csv 与 oulad.content；指定后忽略 -d/--data')

    # general training hyper-parameters
    parser.add_argument('--ctx_sample', type=int, default=30, help='spatial neighbor 长度 ls，论文建议 20~40')
    parser.add_argument('--tmp_sample', type=int, default=21, help='temporal 序列长度 lt，论文建议 20~40')
    parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs (优化：从50增加到100)')
    parser.add_argument('--bs', type=int, default=800, help='batch_size')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate (优化：从1e-3增加到2e-3，配合分层学习率)')
    parser.add_argument('--drop_out', type=float, default=0.2, help='dropout probability for all dropout layers')
    parser.add_argument('--tolerance', type=float, default=0,
                        help='tolerated marginal improvement for early stopper')

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=60, help='random seed for all randomized algorithms')
    parser.add_argument('--ngh_cache', action='store_true',
                        help='(currently not suggested due to overwhelming memory consumption) cache temporal neighbors previously calculated to speed up repeated lookup')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--aug_len', type=float, default=1.5, help='augmentation seq length')
    parser.add_argument('--p_related', type=float, default=0.7, help='课程关系负采样用；主流程用 neg_sampler')
    parser.add_argument('--alpha', type=float, default=0.35, help='预训练 consistency/diversity 中 diversity 权重，论文建议 small α（0.25~0.5）')
    parser.add_argument('--top_k_steps', type=int, default=6, help='预训练每 batch 内 top_k 阶段内循环次数，减小可加速')
    parser.add_argument('--model_steps', type=int, default=3, help='预训练每 batch 内 model 阶段内循环次数，减小可加速')
    parser.add_argument('--no_topk', action='store_true', help='消融：关闭 Top_k 与双阶段，仅单阶段 BCE（ContraTGT baseline）')
    parser.add_argument('--no_student_consistency', action='store_true', help='消融：关闭同学生时序一致性损失')
    parser.add_argument('--neg_sampler', type=str, default='random', choices=['random', 'two_stage'],
                        help='random=仅 RandEdgeSampler；two_stage=70%% random + 30%% hard（推荐稳定）')
    parser.add_argument('--neg_hard_ratio', type=float, default=0.3, help='two_stage 时 hard 负样本比例，1-此为 random；0.2 更稳、更接近 test 分布')
    parser.add_argument('--consistency_weight', type=float, default=0.1, help='同学生时序一致性损失权重，扩展配置可适当调低避免压过主损失')
    parser.add_argument('--ablation_suffix', type=str, default='', help='消融保存后缀，如 baseline/neg/topk/consistency/all，模型存为 <data_name>_<suffix>.pth')
    parser.add_argument('--split_by_ui', action='store_true',
                        help='按 (u,i) 划分 train/val/test，与基线一致；默认按时间划分')
    parser.add_argument('--paper_eval', action='store_true',
                        help='与论文 Section V-A 一致：按边随机 1:1:8 划分（10%% train, 10%% val, 80%% test）')
    parser.add_argument('--train_ratio', type=float, default=None, help='与 paper_eval 二选一；若设则按边随机划分')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例（paper_eval 时为 0.1）')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集比例（paper_eval 时为 0.8）')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


def get_data_paths(args):
    """
    根据 args.data 或 args.data_dir 解析边数据文件、节点特征文件与 data_name。
    - 若指定 --data_dir（如 data_abc_0.01）：使用 <项目根>/data/all_data/<data_dir>/ml_oulad.csv 与 oulad.content，
      假定在项目根（EduTGT）下运行，data_name 为目录名。
    - 否则使用 data/ml_{data}.csv 与 node_feature/{data}.content。
    返回 (edges_file, feature_file, data_name)。
    """
    if getattr(args, 'data_dir', None) and args.data_dir.strip():
        cwd = os.getcwd()
        d = args.data_dir.strip()
        if os.path.isabs(d):
            data_dir = d
        elif os.path.dirname(d):
            data_dir = os.path.join(cwd, d)
        else:
            data_dir = os.path.join(cwd, 'data', 'all_data', d)
        edges_file = os.path.join(data_dir, 'ml_oulad.csv')
        feature_file = os.path.join(data_dir, 'oulad.content')
        data_name = os.path.basename(os.path.normpath(data_dir))
        return edges_file, feature_file, data_name
    data_name = args.data
    edges_file = f'data/ml_{data_name}.csv'
    feature_file = f'node_feature/{data_name}.content'
    return edges_file, feature_file, data_name


class EarlyStopping:
    def __init__(self, dn, max_round=3, higher_better=True, tolerance=1e-10, checkpoint_dir=None):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        if checkpoint_dir:
            self.path = os.path.join(checkpoint_dir, f'{dn}.pth')
        else:
            self.path = f'saved_checkpoints/{dn}.pth'

    def __call__(self, curr_val, model):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.save_checkpoint(curr_val, model)
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.save_checkpoint(curr_val, model)
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round

    def save_checkpoint(self, val_ap, model):
        print('Validation ap increased (', self.last_best, ' --> ', val_ap, ').  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.last_best = val_ap


class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def Dataset(file='data/ml_slashdot.csv', starting_line=1, split_by_ui=False, train_ratio=None, val_ratio=0.1, test_ratio=0.2, random_state=42):
    ecols = Namespace({'id': 0,
                       'FromNodeId': 1,
                       'ToNodeId': 2,
                       'TimeStep': 3,
                       'label': 4,
                       'idx': 5
                       })
    
    with open(file) as f:
        lines = f.read().splitlines()
    edges = [[float(r) for r in row.split(',')] for row in lines[starting_line:]]
    edges = torch.tensor(edges, dtype=torch.long)

    new_edges = edges[:, [ecols.FromNodeId, ecols.ToNodeId]]  
    _, new_edges = new_edges.unique(return_inverse=True)  
    edges[:, [ecols.FromNodeId, ecols.ToNodeId]] = new_edges  
    timestamp = edges[:, [ecols.TimeStep]]

    
    num_nodes = edges[:, [ecols.FromNodeId, ecols.ToNodeId]].unique().size(0)
    nodes_list = edges[:, [ecols.FromNodeId, ecols.ToNodeId]].unique()  

    
    node_time = [-1] * num_nodes  
    for i, edg in enumerate(edges):
        st = int(edg[ecols.FromNodeId])  
        en = int(edg[ecols.ToNodeId])  
        if node_time[st] == -1:  
            node_time[st] = int(timestamp[i])
        if node_time[en] == -1:
            node_time[en] = int(timestamp[i])

    idx = edges[:, [ecols.FromNodeId,
                    ecols.ToNodeId,
                    ecols.TimeStep,
                    ecols.idx]]
    labels = edges[:, ecols.label]

    edge = {'idx': idx, 'labels': labels}

    if train_ratio is not None:
        # 按边随机划分，与论文 Section V-A 一致：1:1:8 => train_ratio=0.1, val_ratio=0.1, test_ratio=0.8
        n = edge['idx'].size(0)
        np.random.seed(random_state)
        perm = np.random.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        valid_train_flag = torch.zeros(n, dtype=torch.bool)
        valid_val_flag = torch.zeros(n, dtype=torch.bool)
        valid_test_flag = torch.zeros(n, dtype=torch.bool)
        valid_train_flag[perm[:n_train]] = True
        valid_val_flag[perm[n_train:n_train + n_val]] = True
        valid_test_flag[perm[n_train + n_val:]] = True
    elif split_by_ui:
        # 按 (u,i) 划分：与基线一致，训练集上可看到每个 (u,i) 的完整时间边
        ui = np.array(edge['idx'][:, [0, 1]])
        lab = np.array(edge['labels'])
        ui_to_label = {}
        for i in range(len(ui)):
            key = (int(ui[i, 0]), int(ui[i, 1]))
            if key not in ui_to_label:
                ui_to_label[key] = int(lab[i])
        unique_ui = np.array(list(ui_to_label.keys()))
        y_ui = np.array([ui_to_label[tuple(k)] for k in unique_ui])
        ui_train, ui_temp, y_train, y_temp = train_test_split(
            unique_ui, y_ui, test_size=val_ratio + test_ratio, random_state=random_state, stratify=y_ui)
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        ui_val, ui_test, y_val, y_test = train_test_split(
            ui_temp, y_temp, test_size=1 - val_size_adjusted, random_state=random_state, stratify=y_temp)
        train_ui_set = set(map(tuple, ui_train))
        val_ui_set = set(map(tuple, ui_val))
        test_ui_set = set(map(tuple, ui_test))
        valid_train_flag = np.array([(int(edge['idx'][i, 0]), int(edge['idx'][i, 1])) in train_ui_set for i in range(edge['idx'].size(0))])
        valid_val_flag = np.array([(int(edge['idx'][i, 0]), int(edge['idx'][i, 1])) in val_ui_set for i in range(edge['idx'].size(0))])
        valid_test_flag = np.array([(int(edge['idx'][i, 0]), int(edge['idx'][i, 1])) in test_ui_set for i in range(edge['idx'].size(0))])
        valid_train_flag = torch.from_numpy(valid_train_flag)
        valid_val_flag = torch.from_numpy(valid_val_flag)
        valid_test_flag = torch.from_numpy(valid_test_flag)
    else:
        val_time, test_time = list(np.quantile(edge['idx'][:, 2].numpy(), [0.10, 0.20]))
        valid_train_flag = (edge['idx'][:, 2] <= val_time)
        valid_val_flag = ((edge['idx'][:, 2] > val_time) & (edge['idx'][:, 2] <= test_time))
        valid_test_flag = (edge['idx'][:, 2] > test_time)

    train_edge = edge['idx'][valid_train_flag]
    train_label = edge['labels'][valid_train_flag]
    train_data = {'idx': train_edge, 'labels': train_label}
    test_edge = edge['idx'][valid_test_flag]
    test_label = edge['labels'][valid_test_flag]
    test_data = {'idx': test_edge, 'labels': test_label}
    val_edge = edge['idx'][valid_val_flag]
    val_label = edge['labels'][valid_val_flag]
    val_data = {'idx': val_edge, 'labels': val_label}

    
    total_node_set = set(np.array(edge['idx'][:, 0])).union(np.array(edge['idx'][:, 1]))
    train_node_set = set(np.array(train_data['idx'][:, 0])).union(np.array(train_data['idx'][:, 1]))
    new_node_set = total_node_set - train_node_set
    
    is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in
                                 zip(np.array(edge['idx'][:, 0]), np.array(edge['idx'][:, 1]))])
    #     print(is_new_node_edge)

    _mask = torch.from_numpy(is_new_node_edge).to(device=edge['idx'].device)
    nn_val_flag = ((valid_val_flag.bool() if valid_val_flag.dtype != torch.bool else valid_val_flag) & _mask)
    nn_test_flag = ((valid_test_flag.bool() if valid_test_flag.dtype != torch.bool else valid_test_flag) & _mask)

    nn_test_edge = edge['idx'][nn_test_flag]
    nn_test_label = edge['labels'][nn_test_flag]
    nn_test_data = {'idx': nn_test_edge, 'labels': nn_test_label}
    nn_val_edge = edge['idx'][nn_val_flag]
    nn_val_label = edge['labels'][nn_val_flag]
    nn_val_data = {'idx': nn_val_edge, 'labels': nn_val_label}

    return edge, num_nodes, nodes_list, node_time, train_data, test_data, val_data, nn_test_data, nn_val_data


class TimeEncode(torch.nn.Module):
    def __init__(self, time_dim, factor=5, time_encoding='concat'):
        super(TimeEncode, self).__init__()
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float(),
                                             requires_grad=False)
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float(), requires_grad=False)
        self.fc1 = nn.Linear(time_dim * 2, time_dim)
        self.time_encoding = time_encoding
        self.act = nn.LeakyReLU()
        self.norm = nn.LayerNorm(time_dim, eps=1e-6)

    def forward(self, u, ts):
        # ts: [N, L]
        batch_size = ts.size(0)  # batchsize
        seq_len = ts.size(1)  # seq

        ts = ts.view(batch_size, seq_len, 1)
        map_ts = ts * self.basis_freq.view(1, 1, -1)
        map_ts += self.phase.view(1, 1, -1)
        harmonic = self.norm(torch.cos(map_ts))
        #         print(map_ts)
        if self.time_encoding == "concat":
            x = self.fc1(torch.cat([u, harmonic], dim=-1))
        elif self.time_encoding == "sum":
            x = u * 0.8 + harmonic * 0.2  
        return x

def get_edges(s_idx,e_idx,edge):
    batch_edge = edge['idx'][s_idx:e_idx]
    batch_label = edge['labels'][s_idx:e_idx]
    batch_data = {'idx':batch_edge,'labels':batch_label}
    return batch_data

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RandEdgeSampler(object):
    def __init__(self, dst_list):
        self.dst_list = np.unique(dst_list)
    def sample(self, size):
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.dst_list[dst_index]

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
