import os
import sys

# 保证从项目根运行 python passfail_models/main_passfail.py 时能导入 script 下的模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(SCRIPT_DIR)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, os.path.join(CODE_ROOT, 'script'))

from utils import *
import pandas as pd
from sampling import *
import scipy.sparse as sp
import math
import copy
from model import  *
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import random
from collections import defaultdict
import numpy as np

args, sys_argv = get_args()
# 与 pretrain/main_link 一致：同一 seed 保证数据划分与训练/评估可复现
init_seeds(getattr(args, 'seed', 60))

GPU = args.gpu
DATA = args.data
LEARNING_RATE = args.lr

device = torch.device('cuda:{}'.format(GPU))

edges_file, feature_file, data_name = get_data_paths(args)
# 下游默认与预训练对齐：按边随机 1:1:8、相同 seed，不按时间划分，以体现预训练对下游的增益
if getattr(args, 'split_by_ui', False):
    train_ratio, val_ratio, test_ratio = None, getattr(args, 'val_ratio', 0.1), getattr(args, 'test_ratio', 0.2)
else:
    train_ratio = getattr(args, 'train_ratio', 0.1)
    val_ratio = getattr(args, 'val_ratio', 0.1)
    test_ratio = getattr(args, 'test_ratio', 0.8)
edges, num_nodes, nodes_list, node_time, train_data, test_data, val_data, nn_test_data, nn_val_data = Dataset(
    file=edges_file, split_by_ui=getattr(args, 'split_by_ui', False),
    train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, random_state=args.seed)
adj_list = get_adj_list(edges)
node_l, ts_l, idx_l, offset_l = init_offset(adj_list)
interaction_list, idx_list_sorted = get_interaction_list(edges)


features = pd.read_csv(feature_file, header=None)
features = normalize_features(features)
features = torch.tensor(features)
fea_dim = features.shape[1]
time_encode = TimeEncode(time_dim = fea_dim,time_encoding = 'concat')
time_information = TimeEncode(time_dim = fea_dim,time_encoding = 'sum')

indim = fea_dim
outdim = 128
nheads = 4
dropout = args.drop_out
N = 2
n_epoch = args.n_epoch
BATCH_SIZE = args.bs
num_instance = len(train_data['idx'])
num_batch = math.ceil(num_instance / BATCH_SIZE)
ctx_sample = args.ctx_sample
tmp_sample = args.tmp_sample
# 消融多 seed 时与 pretrain/main_link 命名一致，可用 --ablation_suffix + --seed 指定加载哪份预训练
_ablation_sfx = getattr(args, 'ablation_suffix', '') or ''
_seed = getattr(args, 'seed', 60)
_seed_sfx = ('_seed' + str(_seed)) if _ablation_sfx else ''
pretrain_path = os.path.join(CODE_ROOT, 'script', 'pretrain_model', f'{data_name}{"_" + _ablation_sfx if _ablation_sfx else ""}{_seed_sfx}.pth')

# 修复不公平：为了公平对比，所有 rand_sampler 应该只使用训练集的节点
train_rand_sampler = RandEdgeSampler(train_data['idx'][:,1])
val_rand_sampler = RandEdgeSampler(train_data['idx'][:,1])
test_rand_sampler = RandEdgeSampler(train_data['idx'][:,1])

st_model = SpatialTemporal(in_dim=indim,out_dim=outdim,n_heads=nheads,dropout=dropout,N=N)
st_model.load_state_dict(torch.load(pretrain_path))
st_model = st_model.to(device)
st_optimizer = optim.Adam(st_model.parameters(),lr=LEARNING_RATE, weight_decay=1e-5)
# 正类权重：平衡通过(1)/不通过(0)，避免模型全预测通过导致 AUC 低
train_labels_01 = (train_data['labels'] == 1).float()
n_pos = train_labels_01.sum().item()
n_neg = len(train_labels_01) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
st_criterion = torch.nn.BCELoss(reduction='mean')
st_criterion_eval = torch.nn.BCELoss()
_passfail_dn = f'{data_name}{"_" + _ablation_sfx if _ablation_sfx else ""}{_seed_sfx}'  # 消融多 seed 时 passfail checkpoint 也按 seed 区分
early_stopping = EarlyStopping(dn=_passfail_dn, max_round=8, checkpoint_dir=os.path.join(CODE_ROOT, 'script', 'saved_checkpoints'))
MODEL_SAVE_PATH = os.path.join(CODE_ROOT, 'script', 'saved_models', f'{_passfail_dn}.pth')
# 预训练/checkpoint/saved_models 使用 script 目录
os.makedirs(os.path.join(CODE_ROOT, 'script', 'saved_models'), exist_ok=True)
os.makedirs(os.path.join(CODE_ROOT, 'script', 'saved_checkpoints'), exist_ok=True)

def eval_epoch(data, batch_size, model, ctx_sample, tmp_sample, rand_sampler):
    init_seeds(getattr(args, 'seed', 60))
    num_instance = len(data['idx'])
    loss, acc, ap, auc = [], [], [], []
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
        model.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance - 1, s_idx + batch_size)
            batch_data = get_edges(s_idx, e_idx, data)  
            #             batch_rand_sampler = RandEdgeSampler(batch_data['idx'][:,1])
            node_sum = len(batch_data['idx'])

            # spatial layer
            batch_ngh_node, batch_ngh_ts, batch_ngh_idx, batch_ngh_mask = get_neighbor_list(node_l, ts_l, idx_l,
                                                                                            offset_l,
                                                                                            batch_data['idx'][:, 0],
                                                                                            batch_data['idx'][:, 2],
                                                                                            num_sample=ctx_sample)
            to_ngh_node, to_ngh_ts, to_ngh_idx, to_ngh_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                batch_data['idx'][:, 1],
                                                                                batch_data['idx'][:, 2],
                                                                                num_sample=ctx_sample)

            con_seq_fea = np.empty((node_sum, ctx_sample, indim))  
            con_to_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(batch_ngh_node):
                for idj, j in enumerate(i):
                    con_seq_fea[idx, idj, :] = features[j]
            for idx, i in enumerate(to_ngh_node):
                for idj, j in enumerate(i):
                    con_to_fea[idx, idj, :] = features[j]

            # target node spatial sequence metric
            con_seq_fea = torch.FloatTensor(np.array(con_seq_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ngh_ts)
            con_seq_fea = time_information(con_seq_fea, ts)  
            context_feature = time_encode(con_seq_fea, torch.tensor(batch_ngh_ts)).to(device)
            batch_ngh_mask = torch.LongTensor(np.array(batch_ngh_mask)).to(device)

            # dest node spatial sequence metric
            con_to_fea = torch.FloatTensor(np.array(con_to_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ngh_ts)
            con_to_fea = time_information(con_to_fea, ts)  
            to_con_feature = time_encode(con_to_fea, torch.tensor(to_ngh_ts)).to(device)
            to_ngh_mask = torch.LongTensor(np.array(to_ngh_mask)).to(device)

            # ---------------------temporal layer----------------------
            batch_node_seq, batch_node_seq_mask, batch_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                                     interaction_list, flag=True, idx_list_sorted=idx_list_sorted)
            temp_seq_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(batch_node_seq):
                for idj, j in enumerate(i):
                    temp_seq_fea[idx, idj, :] = features[j]
            # target node temporal sequence metric
            temp_seq_fea = torch.FloatTensor(np.array(temp_seq_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ts)
            temp_seq_fea = time_information(temp_seq_fea, ts)  #########
            temporal_feature = time_encode(temp_seq_fea, torch.tensor(batch_ts)).to(device)
            batch_node_seq_mask = torch.LongTensor(np.array(batch_node_seq_mask)).to(device)

            # dest node temporal sequence metric
            to_node_seq, to_node_seq_mask, to_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                            interaction_list, flag=False, idx_list_sorted=idx_list_sorted)
            to_seq_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(to_node_seq):
                for idj, j in enumerate(i):
                    to_seq_fea[idx, idj, :] = features[j]
            to_seq_fea = torch.FloatTensor(to_seq_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ts)
            to_seq_fea = time_information(to_seq_fea, ts)  
            to_seq_feature = time_encode(to_seq_fea, torch.tensor(to_ts)).to(device)
            to_node_seq_mask = torch.LongTensor(np.array(to_node_seq_mask)).to(device)

            # fake node spatial and temporal sequence metric
            fake_node = rand_sampler.sample(node_sum)
            fake_con_node, fake_con_ts, fake_con_idx, fake_con_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                        fake_node,
                                                                                        batch_data['idx'][:, 2],
                                                                                        num_sample=ctx_sample)
            fake_con_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(fake_con_node):
                for idj, j in enumerate(i):
                    fake_con_fea[idx, idj, :] = features[j]
            fake_con_fea = torch.FloatTensor(np.array(fake_con_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_con_ts)
            fake_con_fea = time_information(fake_con_fea, ts)
            fake_con_fea = time_encode(fake_con_fea, torch.tensor(fake_con_ts)).to(device)
            fake_con_mask = torch.LongTensor(np.array(fake_con_mask)).to(device)

            fake_batch_data = copy.deepcopy(batch_data)
            fake_batch_data['idx'][:, 1] = torch.tensor(fake_node)
            fake_tmp_seq, fake_tmp_mask, fake_tmp_ts = get_unique_node_sequence(fake_batch_data, edges, tmp_sample,
                                                                                interaction_list, flag=False, idx_list_sorted=idx_list_sorted)
            fake_tmp_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(fake_tmp_seq):
                for idj, j in enumerate(i):
                    fake_tmp_fea[idx, idj, :] = features[j]
            fake_tmp_fea = torch.FloatTensor(fake_tmp_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_tmp_ts)
            fake_tmp_fea = time_information(fake_tmp_fea, ts)
            fake_temp_fea = time_encode(fake_tmp_fea, torch.tensor(fake_tmp_ts)).to(device)
            fake_temp_mask = torch.LongTensor(np.array(fake_tmp_mask)).to(device)

            pos_label = torch.ones(node_sum, dtype=torch.float, device=device)
            neg_label = torch.zeros(node_sum, dtype=torch.float, device=device)

            pos_prob, neg_prob = st_model.linkPredict(context_feature, batch_ngh_mask, temporal_feature,
                                                      batch_node_seq_mask,
                                                      to_con_feature, to_ngh_mask, to_seq_feature, to_node_seq_mask,
                                                      fake_con_fea, fake_con_mask, fake_temp_fea, fake_temp_mask)

            st_loss = st_criterion_eval(pos_prob, pos_label)
            st_loss += st_criterion_eval(neg_prob, neg_label)
            loss.append(st_loss.item())

            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(node_sum), np.zeros(node_sum)])
            auc.append(roc_auc_score(true_label, pred_score))
            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))

    return np.mean(acc), np.mean(ap), np.average(loss), np.mean(auc)


def collect_edge_scores(data, batch_size, model, ctx_sample, tmp_sample, rand_sampler):
    """遍历 data，对每条边得到 (u, i, ts, pos_prob, label_01)，用于课程级聚合。"""
    init_seeds(getattr(args, 'seed', 60))
    model.eval()
    num_instance = len(data['idx'])
    num_batch = math.ceil(num_instance / batch_size)
    edge_list = []
    with torch.no_grad():
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)
            batch_data = get_edges(s_idx, e_idx, data)
            node_sum = len(batch_data['idx'])
            batch_ngh_node, batch_ngh_ts, batch_ngh_idx, batch_ngh_mask = get_neighbor_list(
                node_l, ts_l, idx_l, offset_l,
                batch_data['idx'][:, 0], batch_data['idx'][:, 2], num_sample=ctx_sample)
            to_ngh_node, to_ngh_ts, to_ngh_idx, to_ngh_mask = get_neighbor_list(
                node_l, ts_l, idx_l, offset_l,
                batch_data['idx'][:, 1], batch_data['idx'][:, 2], num_sample=ctx_sample)
            con_seq_fea = np.empty((node_sum, ctx_sample, indim))
            con_to_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(batch_ngh_node):
                for idj, j in enumerate(i):
                    con_seq_fea[idx, idj, :] = features[j]
            for idx, i in enumerate(to_ngh_node):
                for idj, j in enumerate(i):
                    con_to_fea[idx, idj, :] = features[j]
            con_seq_fea = torch.FloatTensor(np.array(con_seq_fea))
            ts_batch = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ngh_ts)
            con_seq_fea = time_information(con_seq_fea, ts_batch)
            context_feature = time_encode(con_seq_fea, torch.tensor(batch_ngh_ts)).to(device)
            batch_ngh_mask = torch.LongTensor(np.array(batch_ngh_mask)).to(device)
            con_to_fea = torch.FloatTensor(np.array(con_to_fea))
            ts_to = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ngh_ts)
            con_to_fea = time_information(con_to_fea, ts_to)
            to_con_feature = time_encode(con_to_fea, torch.tensor(to_ngh_ts)).to(device)
            to_ngh_mask = torch.LongTensor(np.array(to_ngh_mask)).to(device)
            batch_node_seq, batch_node_seq_mask, batch_ts = get_unique_node_sequence(
                batch_data, edges, tmp_sample, interaction_list, flag=True, idx_list_sorted=idx_list_sorted)
            temp_seq_fea = np.empty((node_sum, tmp_sample, indim))
            for idx, i in enumerate(batch_node_seq):
                for idj, j in enumerate(i):
                    temp_seq_fea[idx, idj, :] = features[j]
            temp_seq_fea = torch.FloatTensor(np.array(temp_seq_fea))
            ts_tmp = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ts)
            temp_seq_fea = time_information(temp_seq_fea, ts_tmp)
            temporal_feature = time_encode(temp_seq_fea, torch.tensor(batch_ts)).to(device)
            batch_node_seq_mask = torch.LongTensor(np.array(batch_node_seq_mask)).to(device)
            to_node_seq, to_node_seq_mask, to_ts = get_unique_node_sequence(
                batch_data, edges, tmp_sample, interaction_list, flag=False, idx_list_sorted=idx_list_sorted)
            to_seq_fea = np.empty((node_sum, tmp_sample, indim))
            for idx, i in enumerate(to_node_seq):
                for idj, j in enumerate(i):
                    to_seq_fea[idx, idj, :] = features[j]
            to_seq_fea = torch.FloatTensor(to_seq_fea)
            ts_to_seq = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ts)
            to_seq_fea = time_information(to_seq_fea, ts_to_seq)
            to_seq_feature = time_encode(to_seq_fea, torch.tensor(to_ts)).to(device)
            to_node_seq_mask = torch.LongTensor(np.array(to_node_seq_mask)).to(device)
            fake_node = rand_sampler.sample(node_sum)
            fake_con_node, fake_con_ts, fake_con_idx, fake_con_mask = get_neighbor_list(
                node_l, ts_l, idx_l, offset_l, fake_node, batch_data['idx'][:, 2], num_sample=ctx_sample)
            fake_con_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(fake_con_node):
                for idj, j in enumerate(i):
                    fake_con_fea[idx, idj, :] = features[j]
            fake_con_fea = torch.FloatTensor(np.array(fake_con_fea))
            ts_fake = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_con_ts)
            fake_con_fea = time_information(fake_con_fea, ts_fake)
            fake_con_fea = time_encode(fake_con_fea, torch.tensor(fake_con_ts)).to(device)
            fake_con_mask = torch.LongTensor(np.array(fake_con_mask)).to(device)
            fake_batch_data = copy.deepcopy(batch_data)
            fake_batch_data['idx'][:, 1] = torch.tensor(fake_node)
            fake_tmp_seq, fake_tmp_mask, fake_tmp_ts = get_unique_node_sequence(
                fake_batch_data, edges, tmp_sample, interaction_list, flag=False, idx_list_sorted=idx_list_sorted)
            fake_tmp_fea = np.empty((node_sum, tmp_sample, indim))
            for idx, i in enumerate(fake_tmp_seq):
                for idj, j in enumerate(i):
                    fake_tmp_fea[idx, idj, :] = features[j]
            fake_tmp_fea = torch.FloatTensor(fake_tmp_fea)
            ts_fake_tmp = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_tmp_ts)
            fake_tmp_fea = time_information(fake_tmp_fea, ts_fake_tmp)
            fake_temp_fea = time_encode(fake_tmp_fea, torch.tensor(fake_tmp_ts)).to(device)
            fake_temp_mask = torch.LongTensor(np.array(fake_tmp_mask)).to(device)
            pos_prob, _ = model.linkPredict(
                context_feature, batch_ngh_mask, temporal_feature, batch_node_seq_mask,
                to_con_feature, to_ngh_mask, to_seq_feature, to_node_seq_mask,
                fake_con_fea, fake_con_mask, fake_temp_fea, fake_temp_mask)
            pos_prob_np = pos_prob.cpu().numpy().flatten()
            u_np = batch_data['idx'][:, 0].cpu().numpy()
            i_np = batch_data['idx'][:, 1].cpu().numpy()
            ts_np = batch_data['idx'][:, 2].cpu().numpy().astype(np.float64)
            lab = batch_data['labels'].cpu().numpy()
            label_01 = (lab == 1).astype(np.int64)
            for j in range(node_sum):
                edge_list.append((u_np[j], i_np[j], ts_np[j], float(pos_prob_np[j]), int(label_01[j])))
    return edge_list


def aggregate_course_level_time_weighted(edge_list):
    """方案 E：按 (u,i) 时间加权平均 pos_prob，得到课程级 score 与 label。"""
    groups = defaultdict(list)
    for (u, i, ts, score, label) in edge_list:
        groups[(int(u), int(i))].append((float(ts), float(score), int(label)))
    scores, labels = [], []
    for key in sorted(groups.keys()):
        arr = groups[key]
        w_sum = sum(ts for ts, _, _ in arr)
        if w_sum > 0:
            s_ui = sum(ts * s for ts, s, _ in arr) / w_sum
        else:
            s_ui = sum(s for _, s, _ in arr) / len(arr)
        scores.append(s_ui)
        labels.append(arr[0][2])
    return np.array(scores), np.array(labels)


def eval_course_level(data, batch_size, model, ctx_sample, tmp_sample, rand_sampler):
    """课程级评估：收集边分数 -> 时间加权聚合 -> 算 AUC/AP/Acc/Loss（与 models 一致）。"""
    edge_list = collect_edge_scores(data, batch_size, model, ctx_sample, tmp_sample, rand_sampler)
    if not edge_list:
        return 0.0, 0.0, 0.0, 0.0
    scores, labels = aggregate_course_level_time_weighted(edge_list)
    scores_clip = np.clip(scores, 1e-7, 1.0 - 1e-7)
    if len(np.unique(labels)) < 2:
        auc, ap = 0.5, 0.5
    else:
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
    acc = (labels == (scores >= 0.5).astype(int)).mean()
    loss = log_loss(labels, scores_clip)
    return float(acc), float(ap), float(loss), float(auc)


for m in range(1):
    print('**************************************************')
    print('run {}------ctx:{},tmp:{},dim:{},N:{},dropout:{}'.format(m, ctx_sample, tmp_sample, indim, N, dropout))
    print('**************************************************')

    for epoch in tqdm(range(n_epoch)):
        train_loss, train_acc, train_ap, train_auc = [], [], [], []
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            batch_data = get_edges(s_idx, e_idx, train_data)

            batch_rand_sampler = RandEdgeSampler(batch_data['idx'][:, 1])

            node_sum = len(batch_data['idx'])
            ######################

            # spatial layer
            batch_ngh_node, batch_ngh_ts, batch_ngh_idx, batch_ngh_mask = get_neighbor_list(node_l, ts_l, idx_l,
                                                                                            offset_l,
                                                                                            batch_data['idx'][:, 0],
                                                                                            batch_data['idx'][:, 2],
                                                                                            num_sample=ctx_sample)
            to_ngh_node, to_ngh_ts, to_ngh_idx, to_ngh_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                batch_data['idx'][:, 1],
                                                                                batch_data['idx'][:, 2],
                                                                                num_sample=ctx_sample)

            con_seq_fea = np.empty((node_sum, ctx_sample, indim))  
            con_to_fea = np.empty((node_sum, ctx_sample, indim))

            for idx, i in enumerate(batch_ngh_node):
                for idj, j in enumerate(i):
                    con_seq_fea[idx, idj, :] = features[j]
            for idx, i in enumerate(to_ngh_node):
                for idj, j in enumerate(i):
                    con_to_fea[idx, idj, :] = features[j]

            
            con_seq_fea = torch.FloatTensor(np.array(con_seq_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ngh_ts)
            con_seq_fea = time_information(con_seq_fea, ts)  ########
            context_feature = time_encode(con_seq_fea, torch.tensor(batch_ngh_ts)).to(device)
            batch_ngh_mask = torch.LongTensor(np.array(batch_ngh_mask)).to(device)

            
            con_to_fea = torch.FloatTensor(np.array(con_to_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ngh_ts)
            con_to_fea = time_information(con_to_fea, ts)  #########
            to_con_feature = time_encode(con_to_fea, torch.tensor(to_ngh_ts)).to(device)
            to_ngh_mask = torch.LongTensor(np.array(to_ngh_mask)).to(device)

            #################temporal layer###################
            batch_node_seq, batch_node_seq_mask, batch_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                                     interaction_list, flag=True, idx_list_sorted=idx_list_sorted)
            temp_seq_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(batch_node_seq):
                for idj, j in enumerate(i):
                    temp_seq_fea[idx, idj, :] = features[j]
            
            temp_seq_fea = torch.FloatTensor(np.array(temp_seq_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ts)
            temp_seq_fea = time_information(temp_seq_fea, ts)  
            temporal_feature = time_encode(temp_seq_fea, torch.tensor(batch_ts)).to(device)
            batch_node_seq_mask = torch.LongTensor(np.array(batch_node_seq_mask)).to(device)

            
            to_node_seq, to_node_seq_mask, to_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                            interaction_list, flag=False, idx_list_sorted=idx_list_sorted)
            to_seq_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(to_node_seq):
                for idj, j in enumerate(i):
                    to_seq_fea[idx, idj, :] = features[j]
            to_seq_fea = torch.FloatTensor(to_seq_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ts)
            to_seq_fea = time_information(to_seq_fea, ts)
            to_seq_feature = time_encode(to_seq_fea, torch.tensor(to_ts)).to(device)
            to_node_seq_mask = torch.LongTensor(np.array(to_node_seq_mask)).to(device)

            ############fake node spatial和temporalseq metric##############
            fake_node = train_rand_sampler.sample(node_sum)
            fake_con_node, fake_con_ts, fake_con_idx, fake_con_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                        fake_node,
                                                                                        batch_data['idx'][:, 2],
                                                                                        num_sample=ctx_sample)
            fake_con_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(fake_con_node):
                for idj, j in enumerate(i):
                    fake_con_fea[idx, idj, :] = features[j]
            fake_con_fea = torch.FloatTensor(np.array(fake_con_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_con_ts)
            fake_con_fea = time_information(fake_con_fea, ts)
            fake_con_fea = time_encode(fake_con_fea, torch.tensor(fake_con_ts)).to(device)
            fake_con_mask = torch.LongTensor(np.array(fake_con_mask)).to(device)

            fake_batch_data = copy.deepcopy(batch_data)
            fake_batch_data['idx'][:, 1] = torch.tensor(fake_node)
            fake_tmp_seq, fake_tmp_mask, fake_tmp_ts = get_unique_node_sequence(fake_batch_data, edges, tmp_sample,
                                                                                interaction_list, flag=False, idx_list_sorted=idx_list_sorted)
            fake_tmp_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(fake_tmp_seq):
                for idj, j in enumerate(i):
                    fake_tmp_fea[idx, idj, :] = features[j]
            fake_tmp_fea = torch.FloatTensor(fake_tmp_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_tmp_ts)
            fake_tmp_fea = time_information(fake_tmp_fea, ts)
            fake_temp_fea = time_encode(fake_tmp_fea, torch.tensor(fake_tmp_ts)).to(device)
            fake_temp_mask = torch.LongTensor(np.array(fake_tmp_mask)).to(device)
            ######################train（做法一：pass/fail 监督，无新增边）######################
            # 使用边的通过/不通过标签，不再用「真边 vs 负样本」
            label_01 = (batch_data['labels'] == 1).float().to(device)

            st_optimizer.zero_grad()
            st_model = st_model.train()
            pos_prob, neg_prob = st_model.linkPredict(context_feature, batch_ngh_mask, temporal_feature,
                                                      batch_node_seq_mask,
                                                      to_con_feature, to_ngh_mask, to_seq_feature, to_node_seq_mask,
                                                      fake_con_fea, fake_con_mask, fake_temp_fea, fake_temp_mask)

            # 加权 BCE：正类权重 = n_neg/n_pos，缓解「全预测通过」导致 AUC 低
            w = pos_weight.expand_as(label_01) * label_01 + (1 - label_01)
            st_loss = (w * (-label_01 * torch.log(pos_prob.clamp(1e-7, 1 - 1e-7)) - (1 - label_01) * torch.log((1 - pos_prob).clamp(1e-7, 1 - 1e-7)))).mean()

            st_loss.backward()
            st_optimizer.step()

            with torch.no_grad():
                st_model = st_model.eval()
                train_loss.append(st_loss.item())
                p = pos_prob.cpu().numpy().flatten()
                y = label_01.cpu().numpy().flatten()
                train_acc.append(((p >= 0.5).astype(np.float32) == y).mean())
                if len(np.unique(y)) >= 2:
                    train_ap.append(average_precision_score(y, p))
                    train_auc.append(roc_auc_score(y, p))
                else:
                    train_ap.append(0.5)
                    train_auc.append(0.5)

        train_loss = np.average(train_loss)
        train_acc = np.mean(train_acc)
        train_ap = np.mean(train_ap)
        train_auc = np.mean(train_auc)

        val_acc, val_ap, val_loss, val_auc = eval_course_level(val_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample,
                                                               val_rand_sampler)
        print('epoch ', epoch, 'train_acc:', train_acc, 'train_ap:', train_ap, 'train_loss:', train_loss, 'train_auc:',
              train_auc)
        print('epoch ', epoch, 'val_acc:', val_acc, 'val_ap:', val_ap, 'val_loss:', val_loss, 'val_auc:', val_auc, '(course-level)')

        if early_stopping(val_ap, st_model):
            print("Early stopping")
            best_model_path = os.path.join(CODE_ROOT, 'script', 'saved_checkpoints', f'{_passfail_dn}.pth')
            st_model.load_state_dict(torch.load(best_model_path))
            torch.save(st_model.state_dict(), MODEL_SAVE_PATH)
            print("Loaded the best model at epoch {} for inference".format(early_stopping.best_epoch))
            break
    # 使用最佳模型做最终评估（课程级：与 models 的 (u,i) 级别 AUC/AP/Acc 一致）
    best_model_path = os.path.join(CODE_ROOT, 'script', 'saved_checkpoints', f'{_passfail_dn}.pth')
    if os.path.exists(best_model_path):
        st_model.load_state_dict(torch.load(best_model_path))
    train_acc, train_ap, train_loss, train_auc = eval_course_level(
        train_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample, train_rand_sampler)
    val_acc, val_ap, val_loss, val_auc = eval_course_level(
        val_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample, val_rand_sampler)
    test_acc, test_ap, test_loss, test_auc = eval_course_level(
        test_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample, test_rand_sampler)
    nn_test_acc, nn_test_ap, nn_test_loss, nn_test_auc = eval_course_level(
        nn_test_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample, test_rand_sampler)

    print('test_auc (course-level):', test_auc, 'test_ap:', test_ap, 'test_acc:', test_acc, 'test_loss:', test_loss)
    print('nn_test_auc (course-level):', nn_test_auc, 'nn_test_ap:', nn_test_ap, 'nn_test_acc:', nn_test_acc,
          'nn_test_loss:', nn_test_loss)

    # 通过/不通过结果统一写入 result/passfail/<data_dir>/
    result_dir = os.path.join(CODE_ROOT, 'result', 'passfail', data_name)
    os.makedirs(result_dir, exist_ok=True)
    results = {
        'Train_AUC': train_auc, 'Train_AP': train_ap, 'Train_Acc': train_acc, 'Train_Loss': train_loss,
        'Val_AUC': val_auc, 'Val_AP': val_ap, 'Val_Acc': val_acc, 'Val_Loss': val_loss,
        'Test_AUC': test_auc, 'Test_AP': test_ap, 'Test_Acc': test_acc, 'Test_Loss': test_loss,
        'NN_Test_AUC': nn_test_auc, 'NN_Test_AP': nn_test_ap, 'NN_Test_Acc': nn_test_acc, 'NN_Test_Loss': nn_test_loss,
    }
    results_path = os.path.join(result_dir, f'edutgt_{data_name}.csv')
    pd.DataFrame([results]).to_csv(results_path, index=False)
    print(f"结果已保存: {results_path}")
