from utils import *
import pandas as pd
from sampling import *
import scipy.sparse as sp
import math
import copy
from model import  *
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import numpy as np
import os

def _sample_fake_node(node_sum, batch_data, train_rand_sampler, enhanced_sampler, neg_sampler_mode, neg_hard_ratio=0.3):
    """训练负采样：random=仅 RandEdgeSampler；two_stage=(1-neg_hard_ratio) random + neg_hard_ratio hard。"""
    if neg_sampler_mode != 'two_stage' or enhanced_sampler is None:
        return train_rand_sampler.sample(node_sum)
    p_random = 1.0 - neg_hard_ratio
    u_np = batch_data['idx'][:, 0].numpy() if hasattr(batch_data['idx'][:, 0], 'numpy') else np.array(batch_data['idx'][:, 0])
    ts_np = batch_data['idx'][:, 2].numpy() if hasattr(batch_data['idx'][:, 2], 'numpy') else np.array(batch_data['idx'][:, 2])
    i_np = batch_data['idx'][:, 1].numpy() if hasattr(batch_data['idx'][:, 1], 'numpy') else np.array(batch_data['idx'][:, 1])
    out = np.empty(node_sum, dtype=np.int64)
    for i in range(node_sum):
        if np.random.rand() < p_random:
            out[i] = train_rand_sampler.sample(1)[0]
        else:
            out[i] = enhanced_sampler.sample(1, u_batch=u_np[i:i+1], ts_batch=ts_np[i:i+1], i_pos=i_np[i:i+1])[0]
    return out

args, sys_argv = get_args()
# 贯穿全流程的 seed：保证数据划分、负采样、模型初始化、dropout 等可复现
init_seeds(getattr(args, 'seed', 60))

# 脚本所在目录：过程性文件统一放在 code/script 下
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GPU = args.gpu
DATA = args.data
LEARNING_RATE = 1e-3

device = torch.device('cuda:{}'.format(GPU))

edges_file, feature_file, data_name = get_data_paths(args)
# 与 ContraTGT 原代码一致：按时间划分（quantile 0.1, 0.2）=> 10% train, 10% val, 80% test
edges, num_nodes, nodes_list, node_time, train_data, _, _, _, _ = Dataset(
    file=edges_file, train_ratio=None, random_state=getattr(args, 'seed', 60))
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
n_epoch = max(80, getattr(args, 'n_epoch', 50))  # 预训练至少 80 epoch 以充分学习
BATCH_SIZE = args.bs
num_instance = len(train_data['idx'])
num_batch = math.ceil(num_instance / BATCH_SIZE)
ctx_sample = getattr(args, 'ctx_sample', 30) or 30
tmp_sample = getattr(args, 'tmp_sample', 21) or 21
pretrain_path = os.path.join(SCRIPT_DIR, 'pretrain_model', f'{data_name}.pth')

# 训练负采样：random=仅 RandEdgeSampler；two_stage=70% random + 30% hard（test 仍 random，避免 distribution shift）
train_rand_sampler = RandEdgeSampler(train_data['idx'][:, 1])
neg_sampler_mode = getattr(args, 'neg_sampler', 'random')
neg_hard_ratio = getattr(args, 'neg_hard_ratio', 0.3)
consistency_weight = getattr(args, 'consistency_weight', 0.1)
enhanced_sampler = None
if neg_sampler_mode == 'two_stage':
    from negative_sampling import EnhancedNegSampler
    enhanced_sampler = EnhancedNegSampler(
        edges['idx'], train_data['idx'][:, 1], seed=getattr(args, 'seed', 60),
        p_module=0.1, p_temporal=0.1, p_degree=0.1
    )

use_topk = not getattr(args, 'no_topk', False)
use_student_consistency = not getattr(args, 'no_student_consistency', False)
model = SpatialTemporal(in_dim=indim,out_dim=outdim,n_heads=nheads,dropout=dropout,N=N)
model = model.to(device)
mid_model = SpatialTemporal(in_dim=indim,out_dim=outdim,n_heads=nheads,dropout=dropout,N=N).to(device) if use_topk else None
optimizer = optim.Adam(model.parameters(),lr=1e-3, weight_decay=1e-5)
criterion = nn.BCELoss()
cos_loss = nn.CosineEmbeddingLoss()
top_k = Top_k(in_dim=indim).to(device) if use_topk else None
optimizer_top = optim.Adam(top_k.parameters(), lr=1e-3, weight_decay=1e-5) if use_topk else None
alpha = getattr(args, 'alpha', 0.35)
ablation_suffix = getattr(args, 'ablation_suffix', '') or ''
sfx = ('_' + ablation_suffix) if ablation_suffix else ''
# 多 seed 消融时按 seed 区分保存，避免互相覆盖；后续 main_link / passfail 可用 --ablation_suffix + --seed 加载
seed_sfx = ('_seed' + str(getattr(args, 'seed', 60))) if ablation_suffix else ''
early_stopping = EarlyStopping(dn=data_name + sfx + seed_sfx, max_round=3, checkpoint_dir=os.path.join(SCRIPT_DIR, 'saved_checkpoints'))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'pretrain_model', f'{data_name}{sfx}{seed_sfx}.pth')
MIDDLE_PATH = os.path.join(SCRIPT_DIR, 'middle_model', f'{data_name}{sfx}{seed_sfx}.pth')
# 过程性文件放在 script 目录下
os.makedirs(os.path.join(SCRIPT_DIR, 'pretrain_model'), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'middle_model'), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'saved_checkpoints'), exist_ok=True)
spasample = round(ctx_sample * args.aug_len)
tmpsample = round(tmp_sample * args.aug_len)
if use_topk:
    torch.save(model.state_dict(), MIDDLE_PATH)

for epoch in tqdm(range(n_epoch)):
    top_loss = []
    train_loss = []
    divloss_1 = []
    divloss_2 = []
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)

        batch_data = get_edges(s_idx, e_idx, train_data)
        batch_rand_sampler = RandEdgeSampler(batch_data['idx'][:, 1])
        node_sum = len(batch_data['idx'])

        ###################################train for top_k#####################################
        if use_topk:
            model = model.eval()
            top_k = top_k.train()
            mid_model.load_state_dict(torch.load(MIDDLE_PATH, map_location=device))
        for se in range(getattr(args, 'top_k_steps', 6) if use_topk else 0):
            # spatial layer (仅 use_topk 时执行)
            
            spa_src_node, spa_src_ts, spa_src_idx, spa_src_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                    batch_data['idx'][:, 0],
                                                                                    batch_data['idx'][:, 2],
                                                                                    num_sample=ctx_sample)
            spa_src_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(spa_src_node):
                for idj, j in enumerate(i):
                    spa_src_fea[idx, idj, :] = features[j]
            spa_src_fea = torch.FloatTensor(spa_src_fea)  
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(spa_src_ts)
            spa_src_feature = time_information(spa_src_fea, ts)
            spa_src_feature = time_encode(spa_src_feature, torch.tensor(spa_src_ts)).to(device)
            spa_src_mask = torch.LongTensor(spa_src_mask).to(device)

            # target node
            spa_dst_node, spa_dst_ts, spa_dst_idx, spa_dst_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                    batch_data['idx'][:, 1],
                                                                                    batch_data['idx'][:, 2],
                                                                                    num_sample=ctx_sample)
            spa_dst_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(spa_dst_node):
                for idj, j in enumerate(i):
                    spa_dst_fea[idx, idj, :] = features[j]
            spa_dst_fea = torch.FloatTensor(np.array(spa_dst_fea))  
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(spa_dst_ts)
            spa_dst_feature = time_information(spa_dst_fea, ts)
            spa_dst_feature = time_encode(spa_dst_feature, torch.tensor(spa_dst_ts)).to(device)
            spa_dst_mask = torch.LongTensor(np.array(spa_dst_mask)).to(device)

            
            spa_node_1, spa_ts_1, spa_idx_1, spa_mask_1 = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                            batch_data['idx'][:, 0],
                                                                            batch_data['idx'][:, 2],
                                                                            num_sample=ctx_sample * 2)

            spa_fea_1 = np.empty((node_sum, ctx_sample * 2, indim))  
            for idx, i in enumerate(spa_node_1):
                for idj, j in enumerate(i):
                    spa_fea_1[idx, idj, :] = features[j]
            spa_fea_1 = torch.FloatTensor(np.array(spa_fea_1))

            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(spa_ts_1)
            spa_fea_1 = time_information(spa_fea_1, ts)  ########
            spa_fea_1 = time_encode(spa_fea_1, torch.tensor(spa_ts_1)).to(device)  
            spa_mask_1 = torch.LongTensor(spa_mask_1).to(device)

            # temporal layer
            
            tmp_src_node, tmp_src_mask, tmp_src_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                              interaction_list, flag=True, idx_list_sorted=idx_list_sorted)
            tmp_src_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(tmp_src_node):
                for idj, j in enumerate(i):
                    tmp_src_fea[idx, idj, :] = features[j]
            tmp_src_fea = torch.FloatTensor(tmp_src_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(tmp_src_ts)
            tmp_src_feature = time_information(tmp_src_fea, ts)
            tmp_src_feature = time_encode(tmp_src_fea, torch.tensor(tmp_src_ts)).to(device)
            tmp_src_mask = torch.LongTensor(tmp_src_mask).to(device)

            
            tmp_dst_node, tmp_dst_mask, tmp_dst_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                              interaction_list, flag=False, idx_list_sorted=idx_list_sorted)
            tmp_dst_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(tmp_dst_node):
                for idj, j in enumerate(i):
                    tmp_dst_fea[idx, idj, :] = features[j]
            tmp_dst_fea = torch.FloatTensor(tmp_dst_fea)  #
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(tmp_dst_ts)
            tmp_dst_feature = time_information(tmp_dst_fea, ts)
            tmp_dst_feature = time_encode(tmp_dst_feature, torch.tensor(tmp_dst_ts)).to(device)
            tmp_dst_mask = torch.LongTensor(np.array(tmp_dst_mask)).to(device)

            
            tmp_node_1, tmp_mask_1, tmp_ts_1 = get_unique_node_sequence(batch_data, edges, (tmp_sample - 1) * 2 + 1,
                                                                        interaction_list, flag=True, idx_list_sorted=idx_list_sorted)

            tmp_fea_1 = np.empty((node_sum, (tmp_sample - 1) * 2 + 1, indim))  
            for idx, i in enumerate(tmp_node_1):
                for idj, j in enumerate(i):
                    tmp_fea_1[idx, idj, :] = features[j]
            tmp_fea_1 = torch.FloatTensor(np.array(tmp_fea_1))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(tmp_ts_1)
            tmp_fea_1 = time_information(tmp_fea_1, ts)  ########
            tmp_fea_1 = time_encode(tmp_fea_1, torch.tensor(tmp_ts_1)).to(device)  
            tmp_mask_1 = torch.tensor(tmp_mask_1).to(device)

            
            fake_node = _sample_fake_node(node_sum, batch_data, train_rand_sampler, enhanced_sampler, neg_sampler_mode, neg_hard_ratio)
            fake_ctx_node, fake_ctx_ts, fake_ctx_idx, fake_ctx_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                        fake_node,
                                                                                        batch_data['idx'][:, 2],
                                                                                        num_sample=ctx_sample)
            fake_ctx_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(fake_ctx_node):
                for idj, j in enumerate(i):
                    fake_ctx_fea[idx, idj, :] = features[j]
            fake_ctx_fea = torch.FloatTensor(np.array(fake_ctx_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_ctx_ts)
            fake_ctx_fea = time_information(fake_ctx_fea, ts)
            fake_ctx_fea = time_encode(fake_ctx_fea, torch.tensor(fake_ctx_ts)).to(device)
            fake_ctx_mask = torch.LongTensor(np.array(fake_ctx_mask)).to(device)

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
            fake_tmp_fea = time_encode(fake_tmp_fea, torch.tensor(fake_tmp_ts)).to(device)
            fake_tmp_mask = torch.LongTensor(np.array(fake_tmp_mask)).to(device)
            # 分别从spatial和temporal中选出k个

            with torch.no_grad():
                spatial_embed = mid_model.get_seq_embed(seq=spa_fea_1, mask=spa_mask_1, view='spatial')
                temporal_embed = mid_model.get_seq_embed(seq=tmp_fea_1, mask=tmp_mask_1, view='temporal')

                target = torch.zeros(node_sum, dtype=torch.float, device=device) - 1
                pos, neg = model.linkPredict(spa_src_feature, spa_src_mask, tmp_src_feature, tmp_src_mask,
                                             spa_dst_feature, spa_dst_mask, tmp_dst_feature, tmp_dst_mask,
                                             fake_ctx_fea, fake_ctx_mask, fake_tmp_fea, fake_tmp_mask)
                pos_label = (pos > 0.5).float()
                neg_label = (neg > 0.5).float()

            
            spa_feature_1, spa_mask_1 = top_k(spa_fea_1, spa_mask_1, spasample, spatial_embed)
            spa_mask_1 = spa_mask_1.to(device)
            spa_feature_1 = spa_feature_1.to(device)

            
            tmp_feature_1, tmp_mask_1 = top_k(tmp_fea_1, tmp_mask_1, tmpsample, temporal_embed)
            tmp_mask_1 = tmp_mask_1.to(device)
            tmp_feature_1 = tmp_feature_1.to(device)

            #######train#######
            optimizer_top.zero_grad()

            # consistency
            pos_prob, neg_prob = mid_model.linkPredict(spa_fea_1, spa_mask_1, tmp_src_feature, tmp_src_mask,
                                                       spa_dst_feature, spa_dst_mask, tmp_dst_feature, tmp_dst_mask,
                                                       fake_ctx_fea, fake_ctx_mask, fake_tmp_fea, fake_tmp_mask)
            con_loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)  
            
            # diversity
            div_1 = mid_model.getEmbed(spa_feature_1, tmp_feature_1, spa_mask_1, tmp_mask_1)
            div_2 = mid_model.getEmbed(spa_src_feature, tmp_src_feature, spa_src_mask, tmp_src_mask)
            div_loss = cos_loss(div_1, div_2, target)

            loss_top = alpha * div_loss + con_loss
            loss_top.backward()
            optimizer_top.step()
            top_loss.append(loss_top.item())
            divloss_1.append(con_loss.item())
            divloss_2.append(div_loss.item())

        ###################################train for model######################################
        model.train()
        if use_topk:
            top_k.eval()
        for se in range(getattr(args, 'model_steps', 3)):
            
            # spatial view
            spa_node, spa_ts, spa_idx, spa_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                    batch_data['idx'][:, 0], batch_data['idx'][:, 2],
                                                                    num_sample=ctx_sample * 2)

            spa_fea = np.empty((node_sum, ctx_sample * 2, indim))  
            for idx, i in enumerate(spa_node):
                for idj, j in enumerate(i):
                    spa_fea[idx, idj, :] = features[j]
            spa_fea = torch.FloatTensor(np.array(spa_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(spa_ts)
            spa_fea = time_information(spa_fea, ts)  ########
            spa_fea = time_encode(spa_fea, torch.tensor(spa_ts)).to(device)

            spa_mask = torch.tensor(spa_mask).to(device)
            spa_embed = model.get_seq_embed(seq=spa_fea, mask=spa_mask, view='spatial')
            if use_topk:
                with torch.no_grad():
                    spa_feature, spa_mask = top_k(spa_fea, spa_mask, spasample, spa_embed)
                spa_mask = spa_mask.to(device)
                spa_feature = spa_feature.to(device)
            else:
                spa_feature, spa_mask = spa_fea.to(device), spa_mask

            # temporal view
            tmp_node, tmp_mask, tmp_ts = get_unique_node_sequence(batch_data, edges, (tmp_sample - 1) * 2 + 1,
                                                                  interaction_list, flag=True, idx_list_sorted=idx_list_sorted)

            tmp_fea = np.empty((node_sum, (tmp_sample - 1) * 2 + 1, indim))  
            for idx, i in enumerate(tmp_node):
                for idj, j in enumerate(i):
                    tmp_fea[idx, idj, :] = features[j]
            tmp_fea = torch.FloatTensor(np.array(tmp_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(tmp_ts)
            tmp_fea = time_information(tmp_fea, ts)  ########
            tmp_fea = time_encode(tmp_fea, torch.tensor(tmp_ts)).to(device)

            tmp_mask = torch.tensor(tmp_mask).to(device)
            tmp_embed = model.get_seq_embed(seq=tmp_fea, mask=tmp_mask, view='temporal')
            if use_topk:
                with torch.no_grad():
                    tmp_feature, tmp_mask = top_k(tmp_fea, tmp_mask, tmpsample, tmp_embed)
                tmp_mask = tmp_mask.to(device)
                tmp_feature = tmp_feature.to(device)
            else:
                tmp_feature, tmp_mask = tmp_fea, tmp_mask

            
            # spatial view
            con_src_node, con_src_ts, con_src_idx, con_src_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                    batch_data['idx'][:, 0],
                                                                                    batch_data['idx'][:, 2],
                                                                                    num_sample=ctx_sample)
            con_src_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(con_src_node):
                for idj, j in enumerate(i):
                    con_src_fea[idx, idj, :] = features[j]
            con_src_fea = torch.FloatTensor(con_src_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(con_src_ts)
            con_src_feature = time_information(con_src_fea, ts)
            con_src_feature = time_encode(con_src_feature, torch.tensor(con_src_ts)).to(device)
            con_src_mask = torch.LongTensor(con_src_mask).to(device)
            # temporal view
            temp_src_node, temp_src_mask, temp_src_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                                 interaction_list, flag=True, idx_list_sorted=idx_list_sorted)
            temp_src_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(temp_src_node):
                for idj, j in enumerate(i):
                    temp_src_fea[idx, idj, :] = features[j]
            temp_src_fea = torch.FloatTensor(temp_src_fea)  # .to(device)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(temp_src_ts)
            temp_src_feature = time_information(temp_src_fea, ts)  ##########
            temp_src_feature = time_encode(temp_src_fea, torch.tensor(temp_src_ts)).to(device)
            temp_src_mask = torch.LongTensor(temp_src_mask).to(device)

            #####fake node#####
            fake_node = _sample_fake_node(node_sum, batch_data, train_rand_sampler, enhanced_sampler, neg_sampler_mode, neg_hard_ratio)
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
            fake_temp_seq, fake_temp_mask, fake_temp_ts = get_unique_node_sequence(fake_batch_data, edges, tmp_sample,
                                                                                   interaction_list, flag=False, idx_list_sorted=idx_list_sorted)
            fake_temp_fea = np.empty((node_sum, tmp_sample, indim)) 
            for idx, i in enumerate(fake_temp_seq):
                for idj, j in enumerate(i):
                    fake_temp_fea[idx, idj, :] = features[j]
            fake_temp_fea = torch.FloatTensor(fake_temp_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_temp_ts)
            fake_temp_fea = time_information(fake_temp_fea, ts)
            fake_temp_fea = time_encode(fake_temp_fea, torch.tensor(fake_temp_ts)).to(device)
            fake_temp_mask = torch.LongTensor(np.array(fake_temp_mask)).to(device)

            #######train#######
            optimizer.zero_grad()

            embed_1 = model.getEmbed(spa_feature, tmp_feature, spa_mask, tmp_mask)
            embed_2 = model.getEmbed(con_src_feature, temp_src_feature, con_src_mask, temp_src_mask)
            embed_fake = model.getEmbed(fake_con_fea, fake_temp_fea, fake_con_mask, fake_temp_mask)

            with torch.no_grad():
                target_1 = torch.ones(node_sum, dtype=torch.float, device=device)
                target_2 = torch.zeros(node_sum, dtype=torch.float, device=device) - 1

            pos_loss = cos_loss(embed_1, embed_2.detach(), target_1)
            neg_loss = cos_loss(embed_1, embed_fake.detach(), target_2)
            loss = pos_loss + neg_loss
            if use_student_consistency:
                src_embed = model.getEmbed(con_src_feature, temp_src_feature, con_src_mask, temp_src_mask)
                batch_u_np = batch_data['idx'][:, 0].numpy() if hasattr(batch_data['idx'][:, 0], 'numpy') else np.array(batch_data['idx'][:, 0])
                consistency_loss = torch.tensor(0.0, device=device)
                num_groups = 0  # 同 batch 内出现 2+ 次的学生数，用于稳定 scale、降低 seed 方差
                for u in np.unique(batch_u_np):
                    idx_u = np.where(batch_u_np == u)[0]
                    if len(idx_u) > 1:
                        emb_u = src_embed[idx_u]
                        mean_u = emb_u.mean(dim=0)
                        consistency_loss = consistency_loss + (emb_u - mean_u).pow(2).sum()
                        num_groups += 1
                if consistency_loss.item() > 0 and num_groups > 0:
                    # 按「学生组」归一化，避免 batch 内组数波动导致梯度尺度不稳定（降低 consistency 单独开时的高方差）
                    loss = loss + consistency_weight * (consistency_loss / num_groups)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        if use_topk:
            torch.save(model.state_dict(), MIDDLE_PATH)

    avg_loss = np.average(train_loss)
    avg_top_loss = np.average(top_loss) if top_loss else 0.0
    avg_con = np.average(divloss_1) if divloss_1 else 0.0
    avg_div = np.average(divloss_2) if divloss_2 else 0.0
    print('epoch ', epoch, 'train_loss:', avg_loss, ',top_loss:', avg_top_loss, 'con_loss:', avg_con, 'div_loss:', avg_div)
    if early_stopping(-avg_loss, model):
        print("Early stopping")
        break
print("Loaded the best model at epoch {} for inference".format(early_stopping.best_epoch))
best_model_path = os.path.join(SCRIPT_DIR, 'saved_checkpoints', f'{data_name}{sfx}{seed_sfx}.pth')
model.load_state_dict(torch.load(best_model_path))
torch.save(model.state_dict(), MODEL_SAVE_PATH)
