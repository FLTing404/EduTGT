"""
ContraTGT 链路预测主训练（边存不存在，边级指标）。
从项目根运行: python link_models/main_link.py --data_dir data_abc_0.01
"""
import os
import sys

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
import numpy as np
from model import  *
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import random

args, sys_argv = get_args()
# 与 pretrain 一致：同一 seed 保证评估阶段负采样等可复现；训练阶段 dropout 等亦与 seed 一致
init_seeds(getattr(args, 'seed', 60))

GPU = args.gpu
LEARNING_RATE = args.lr
device = torch.device('cuda:{}'.format(GPU))

edges_file, feature_file, data_name = get_data_paths(args)
# 与 ContraTGT 及对比组统一：默认按时间划分（quantile 0.1, 0.2）；--paper_eval 时为按边随机 1:1:8
if getattr(args, 'split_by_ui', False):
    train_ratio, val_ratio, test_ratio = None, getattr(args, 'val_ratio', 0.1), getattr(args, 'test_ratio', 0.2)
elif getattr(args, 'paper_eval', False):
    train_ratio, val_ratio, test_ratio = 0.1, 0.1, 0.8
elif getattr(args, 'train_ratio', None) is not None:
    train_ratio = args.train_ratio
    val_ratio = getattr(args, 'val_ratio', 0.1)
    test_ratio = getattr(args, 'test_ratio', 0.8)
else:
    train_ratio = None  # 默认：按时间划分（quantile 0.1, 0.2），与 ContraTGT 及基线统一
    val_ratio, test_ratio = 0.1, 0.8
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
time_encode = TimeEncode(time_dim=fea_dim, time_encoding='concat')
time_information = TimeEncode(time_dim=fea_dim, time_encoding='sum')

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

_ablation_sfx = getattr(args, 'ablation_suffix', '') or ''
_seed = getattr(args, 'seed', 60)
# 消融且多 seed 时预训练权重带 seed 后缀，与 pretrain.py 保存命名一致
_seed_sfx = ('_seed' + str(_seed)) if _ablation_sfx else ''
pretrain_path = os.path.join(CODE_ROOT, 'script', 'pretrain_model', f'{data_name}{"_" + _ablation_sfx if _ablation_sfx else ""}{_seed_sfx}.pth')
# 训练负采样：仅用 RandEdgeSampler（与 ContraTGT 一致）；val/test 用全量边目标节点
train_rand_sampler = RandEdgeSampler(train_data['idx'][:, 1])
val_rand_sampler = RandEdgeSampler(edges['idx'][:, 1])
test_rand_sampler = RandEdgeSampler(edges['idx'][:, 1])

st_model = SpatialTemporal(in_dim=indim, out_dim=outdim, n_heads=nheads, dropout=dropout, N=N)
st_model.load_state_dict(torch.load(pretrain_path))
st_model = st_model.to(device)

# 优化：分层学习率 - 链接头用较大学习率，其他层用较小学习率（微调预训练模型）
link_head_params = []
backbone_params = []
for name, param in st_model.named_parameters():
    if 'affinity_score' in name or 'linear' in name:  # 链接预测头
        link_head_params.append(param)
    else:  # 预训练骨干网络
        backbone_params.append(param)

# 链接头用较大学习率，骨干网络用较小学习率（微调）
st_optimizer = optim.Adam([
    {'params': link_head_params, 'lr': LEARNING_RATE * 5.0, 'weight_decay': 1e-5},  # 链接头：5倍学习率
    {'params': backbone_params, 'lr': LEARNING_RATE * 0.1, 'weight_decay': 1e-5}   # 骨干网络：0.1倍学习率（微调）
])

# 添加学习率调度器（在验证集性能不提升时降低学习率）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    st_optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
)

st_criterion = torch.nn.BCELoss()
st_criterion_eval = torch.nn.BCELoss()
_link_dn = f'{data_name}{"_" + _ablation_sfx if _ablation_sfx else ""}{_seed_sfx}'  # 消融多 seed 时链路 checkpoint 也按 seed 区分
early_stopping = EarlyStopping(dn=_link_dn, max_round=3, checkpoint_dir=os.path.join(CODE_ROOT, 'script', 'saved_checkpoints'))
MODEL_SAVE_PATH = os.path.join(CODE_ROOT, 'script', 'saved_models', f'{_link_dn}.pth')
os.makedirs(os.path.join(CODE_ROOT, 'script', 'saved_models'), exist_ok=True)
os.makedirs(os.path.join(CODE_ROOT, 'script', 'saved_checkpoints'), exist_ok=True)

def eval_epoch(data, batch_size, model, ctx_sample, tmp_sample, rand_sampler, eval_seed=None):
    init_seeds(eval_seed if eval_seed is not None else args.seed)
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

            ############fake node##############
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
            ######################train######################
            with torch.no_grad():
                pos_label = torch.ones(node_sum, dtype=torch.float, device=device)
                neg_label = torch.zeros(node_sum, dtype=torch.float, device=device)

            st_optimizer.zero_grad()
            st_model = st_model.train()  ################
            pos_prob, neg_prob = st_model.linkPredict(context_feature, batch_ngh_mask, temporal_feature,
                                                      batch_node_seq_mask,
                                                      to_con_feature, to_ngh_mask, to_seq_feature, to_node_seq_mask,
                                                      fake_con_fea, fake_con_mask, fake_temp_fea, fake_temp_mask)

            st_loss = st_criterion(pos_prob, pos_label)
            st_loss += st_criterion(neg_prob, neg_label)

            st_loss.backward()
            st_optimizer.step()

            with torch.no_grad():
                st_model = st_model.eval()
                train_loss.append(st_loss.item())
                pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(node_sum), np.zeros(node_sum)])
                train_acc.append((pred_label == true_label).mean())
                train_ap.append(average_precision_score(true_label, pred_score))
                train_auc.append(roc_auc_score(true_label, pred_score))

        train_loss = np.average(train_loss)
        train_acc = np.mean(train_acc)
        train_ap = np.mean(train_ap)
        train_auc = np.mean(train_auc)

        val_acc, val_ap, val_loss, val_auc = eval_epoch(val_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample,
                                                        val_rand_sampler)
        print('epoch ', epoch, 'train_acc:', train_acc, 'train_ap:', train_ap, 'train_loss:', train_loss, 'train_auc:',
              train_auc)
        print('epoch ', epoch, 'val_acc:', val_acc, 'val_ap:', val_ap, 'val_loss:', val_loss, 'val_auc:', val_auc)
        
        # 优化：学习率调度（基于验证集AP）
        scheduler.step(val_ap)

        if early_stopping(val_ap, st_model):
            print("Early stopping")
            best_model_path = os.path.join(CODE_ROOT, 'script', 'saved_checkpoints', f'{_link_dn}.pth')
            st_model.load_state_dict(torch.load(best_model_path))
            torch.save(st_model.state_dict(), MODEL_SAVE_PATH)
            print("Loaded the best model at epoch {} for inference".format(early_stopping.best_epoch))
            break
    best_model_path = os.path.join(CODE_ROOT, 'script', 'saved_checkpoints', f'{_link_dn}.pth')
    if os.path.exists(best_model_path):
        st_model.load_state_dict(torch.load(best_model_path))

    train_acc, train_ap, train_loss, train_auc = eval_epoch(train_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample, train_rand_sampler)
    val_acc, val_ap, val_loss, val_auc = eval_epoch(val_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample, val_rand_sampler)
    test_acc, test_ap, test_loss, test_auc = eval_epoch(test_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample, test_rand_sampler)
    nn_test_acc, nn_test_ap, nn_test_loss, nn_test_auc = eval_epoch(nn_test_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample, test_rand_sampler)

    print('test_auc:', test_auc, 'test_ap:', test_ap, 'test_acc:', test_acc, 'test_loss:', test_loss)
    print('nn_test_auc:', nn_test_auc, 'nn_test_ap:', nn_test_ap, 'nn_test_acc:', nn_test_acc, 'nn_test_loss:', nn_test_loss)

    result_dir = os.path.join(CODE_ROOT, 'result', 'link', data_name)
    os.makedirs(result_dir, exist_ok=True)
    results = {
        'Train_AUC': train_auc, 'Train_AP': train_ap, 'Train_Acc': train_acc, 'Train_Loss': train_loss,
        'Val_AUC': val_auc, 'Val_AP': val_ap, 'Val_Acc': val_acc, 'Val_Loss': val_loss,
        'Test_AUC': test_auc, 'Test_AP': test_ap, 'Test_Acc': test_acc, 'Test_Loss': test_loss,
        'NN_Test_AUC': nn_test_auc, 'NN_Test_AP': nn_test_ap, 'NN_Test_Acc': nn_test_acc, 'NN_Test_Loss': nn_test_loss,
    }
    results_path = os.path.join(result_dir, f'link_edutgt_{data_name}.csv')
    pd.DataFrame([results]).to_csv(results_path, index=False)
    print(f"结果已保存: {results_path}")
