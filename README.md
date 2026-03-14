# EduTGT 使用说明

本仓库将 **OULAD** 开放教育数据整理成**时序二部图**（学生–课程边 + 节点特征），在 ContraTGT 框架基础上做了扩展（**双视图** + Top_k 模块 + mid_model 教师 + **同学生时序一致性**），支持两类下游任务：

- **链路预测**（首要）：边是否存在，边级 AUC/AP/Acc；**EduTGT**（基于 ContraTGT 预训练 + 链路头）与 TGAT/GraphSAGE/JODIE 基线公平对比。
- **课程通过/不通过预测**：课程级二分类，EduTGT+MLP 与纯 MLP/Logistic/LSTM 等六种方法对比。

**与 ContraTGT 一致**：训练负采样**仅 RandEdgeSampler**（与 ContraTGT 一致）。**最稳**：baseline 与 EduTGT 使用相同时间划分、相同评估负采样池、相同 early stopping 规则；差异仅来自模型结构。

数据与转换在 `data/`，预训练与模型在 `script/`，链路与通过/不通过分别在 `link_models/`、`passfail_models/`，结果在 `result/link/`、`result/passfail/`。

---

## 一、目录结构

```
EduTGT/
├── README.md                        # 本说明
├── IMPLEMENTATION_AND_DIFFERENCES.md # EduTGT 与 ContraTGT 论文的差异说明
├── requirements.txt
├── data/                            # 数据与转换
│   ├── convert_oulad.py             # OULAD 全量 → 时序图（边表 + 节点特征）
│   ├── convert_oulad_abc.py         # 按模块(AAA/BBB/CCC)筛选 + 采样 → 小规模
│   ├── OULAD_rawdata/               # 原始 7 个 CSV（需自行放置）
│   └── all_data/                    # 转换后的统一数据
│       ├── data_1/, data_0.1/, data_abc_0.01/ 等
│       │   ├── ml_oulad.csv         # 边表：id,u,i,ts,label,idx
│       │   ├── oulad.content        # 节点特征（每行一节点，逗号分隔）
│       │   └── （可选）ml_oulad_pairs.csv、ml_oulad_edge_feature.csv
│       └── ...
├── script/                          # 预训练（ContraTGT 扩展版）
│   ├── pretrain.py                  # 预训练入口（训练用 RandEdgeSampler + Top_k + 同学生一致性）
│   ├── run_ablation.py              # 一键消融 + 三基线：5 种配置 + TGAT/GraphSAGE/JODIE
│   ├── negative_sampling.py         # 增强负采样（消融用）：伪模块、时序共现、度感知
│   ├── utils.py                     # 参数、路径、Dataset、EarlyStopping
│   ├── sampling.py                  # 时空邻居、交互序列（neighbor_ts < current_ts）
│   ├── model.py                     # SpatialTemporal、Top_k、链接头
│   ├── graph_transformer.py         # 多头注意力、Encoder
│   ├── pretrain_model/              # 预训练权重（<data_name>.pth 或 <data_name>_<ablation_suffix>.pth）
│   ├── saved_checkpoints/           # 早停最佳 checkpoint
│   └── saved_models/                # 最终保存模型
├── link_models/                     # 链路预测
│   ├── main_link.py                 # EduTGT 链路预测（加载预训练 + 全模型微调；训练仅 RandEdgeSampler）
│   ├── data_utils.py                # 边/特征加载、划分、负采样（与 script 对齐）
│   ├── train_link_tgat.py           # TGAT 基线
│   ├── train_link_graphsage.py      # GraphSAGE 基线
│   ├── train_link_jodie.py         # JODIE 基线（KDD 2019，时序嵌入更新）
│   └── run_paper_eval.py            # 多 seed 一键评估并汇总
├── passfail_models/                  # 课程通过/不通过
│   ├── main_passfail.py             # EduTGT + MLP 头（课程级，含 NN_Test）
│   ├── passfail_data.py             # (u,i) 划分、原始特征（单向量+序列）
│   ├── extract_embeddings.py        # 预训练表示按 (u,i) 聚合成单向量/序列
│   ├── train_mlp.py                 # 纯 MLP（原始特征）
│   ├── train_logistic.py            # Logistic（pure / contratgt 表示用预训练）
│   ├── train_lstm.py                # LSTM（pure / contratgt 表示用预训练）
│   └── run_paper_eval.py            # 六种方法多 seed 汇总
└── result/
    ├── link/<data_dir>/             # 链路：link_edutgt_*.csv、link_jodie_*.csv 等
    └── passfail/<data_dir>/         # 通过不通过：edutgt_*.csv、mlp_pure_*.csv 等
```

---

## 二、环境与依赖

在**项目根目录（EduTGT）**下：

```bash
pip install -r requirements.txt
```

主要依赖：`torch`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`tqdm`。  
**可选**：GraphSAGE 需 `torch-geometric`；未安装时 EduTGT、TGAT、JODIE、passfail 仍可运行。

---

## 三、数据格式与流程

### 3.1 原始数据

将 [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) 解压后的 7 个 CSV 放入 `data/OULAD_rawdata/`：

- `studentVle.csv`、`studentInfo.csv`、`courses.csv`、`assessments.csv`
- `studentAssessment.csv`、`studentRegistration.csv`、`vle.csv`

### 3.2 转换脚本

在**项目根 EduTGT** 下执行：

```bash
# 全量 → data/all_data/data_1/
python data/convert_oulad.py

# 全量 10% 采样 → data/all_data/data_0.1/
python data/convert_oulad.py --sample_ratio 0.1

# 仅 AAA,BBB,CCC 模块，1% 采样 → data/all_data/data_abc_0.01/
python data/convert_oulad_abc.py --modules AAA,BBB,CCC --sample_ratio 0.01
```

### 3.3 统一输出格式

每个 `data/all_data/<data_dir>/` 下：

| 文件 | 说明 |
|------|------|
| **ml_oulad.csv** | 边表：`id,u,i,ts,label,idx`。u=学生节点 id，i=课程节点 id，ts=时间步，label=通过(1)/不通过(-1 或 0)，idx=边序号。 |
| **oulad.content** | 节点特征：每行一个节点（与边表 u/i 的 id 对应），逗号分隔的浮点数，行数 ≥ max(u,i)。 |

节点 id 在边表中从 1 开始；脚本内部会做 0-index 处理。预训练、链路、通过/不通过均使用同一套 `ml_oulad.csv` + `oulad.content`。

### 3.4 数据划分与对齐

- **默认**：按**时间**划分（quantile 0.1, 0.2）→ 约 10% 训练、10% 验证、80% 测试，与 ContraTGT 原代码一致；预训练、链路、基线在默认下共用该划分。
- **可选**：加 `--paper_eval` 时改为**按边随机 1:1:8**，预训练、链路、基线同样一致。
- **通过/不通过**：先按边得到 train/val/test 边集合，再按 (u,i) 归属到首条边所在集合，得到 (u,i) 级划分。

---

## 四、运行方式

**所有命令均在项目根目录（EduTGT）下执行。**

### 4.1 预训练（script）

```bash
python script/pretrain.py --data_dir data_abc_0.01
```

- 权重保存：`script/pretrain_model/<data_name>.pth`、`script/saved_checkpoints/`；消融时使用 `--ablation_suffix <name>` 则保存为 `<data_name>_<name>.pth`。
- **超参建议**（论文/实践）：`--ctx_sample`/`--tmp_sample` 建议 20~40（默认 30/21）；`--alpha` 建议 0.25~0.5（consistency > diversity，默认 0.35）；embedding 维度 128 或 256；masking ratio ρ∈[0.25,0.5] 时最稳定。
- 可选：`--no_topk`/`--no_student_consistency` 做消融；`--seed`、`--n_epoch`、`--top_k_steps`/`--model_steps`。

### 4.2 链路预测（EduTGT + 基线）

- **EduTGT**：需先完成上述预训练，再运行 `main_link.py`。
- **TGAT / GraphSAGE / JODIE**：无需预训练，直接运行对应脚本；默认与 EduTGT 同按时间划分，可选 `--paper_eval` 改为按边随机 1:1:8。

```bash
# 预训练（仅 EduTGT 需要）
python script/pretrain.py --data_dir data_abc_0.01

# EduTGT 链路预测（默认按时间划分；加 --paper_eval 则按边随机 1:1:8，与基线一致）
python link_models/main_link.py --data_dir data_abc_0.01
python link_models/main_link.py --data_dir data_abc_0.01 --paper_eval

# 基线（GraphSAGE 需 PyG；TGAT/JODIE 仅需 torch；与 EduTGT 同划分、同负采样协议）
python link_models/train_link_tgat.py --data_dir data_abc_0.01 --paper_eval
python link_models/train_link_graphsage.py --data_dir data_abc_0.01 --paper_eval
python link_models/train_link_jodie.py --data_dir data_abc_0.01 --paper_eval
```

**一键多 seed 汇总**（与论文评估一致）：

```bash
python link_models/run_paper_eval.py --data_dir data_abc_0.01 --seeds 42,43,44,45,46
# 仅汇总已有 CSV：--no_run
# 只跑部分方法：--methods edutgt,tgat
```

结果目录：`result/link/<data_dir>/`，含 `link_edutgt_*.csv`、`link_jodie_*.csv` 等及 `paper_eval_summary_<data_dir>.csv`。

**消融 + 三基线（一条命令跑齐 5 种消融与 TGAT/GraphSAGE/JODIE）**：

```bash
# 从项目根运行；会依次：预训练(各配置) → EduTGT 链路 → TGAT → GraphSAGE → JODIE → 汇总打印 AUC 表
python script/run_ablation.py --data_dir data_abc_0.01

# 单次运行：不传 --seed 时 run_ablation 会自动注入 --seed 60，使预训练/链路/三基线统一用同一 seed；也可显式指定 --seed 60
python script/run_ablation.py --data_dir data_abc_0.01
python script/run_ablation.py --data_dir data_abc_0.01 --seed 60

# 多 seed 跑 5 次并输出带 seed 的结果文件 + 汇总表（论文用 mean±std）
python script/run_ablation.py --data_dir data_abc_0.01 --seeds 42,43,44,45,46
# 结果：result/link/<data_name>/ablation_results_seed_42.csv … ablation_results_seed_46.csv，以及 ablation_summary_<data_name>.csv（含 Mean_AUC, Std_AUC）

# 可选：-d slashdot（公共数据集）；--quick 仅 2 epoch 快速试跑
```

| 模型 | 说明 |
|------|------|
| baseline | ContraTGT baseline（无 Top_k、无 student consistency、random neg） |
| + neg sampler | 同上 + 两阶段负采样 80% random + 20% hard |
| + topk | + Top_k + mid_model（无 student consistency） |
| + consistency | + 同学生时序一致性（无 Top_k） |
| neg_consistency | + neg + consistency（无 Top_k） |
| **TGAT** | TGAT 基线（无需预训练） |
| **GraphSAGE** | GraphSAGE 基线（需 PyG） |
| **JODIE** | JODIE 基线（KDD 2019，时序嵌入 + GRU 更新） |

**对比公平性**：EduTGT 与 TGAT/GraphSAGE/JODIE 使用**相同数据划分**（默认按时间 10%/10%/80%，或 `--paper_eval` 时按边随机 1:1:8）、**相同训练负采样池**（仅训练集目标节点）、**相同 Val/Test 负采样池**（全量边目标节点 `all_dst`）、**相同评估指标**（AUC/AP/Acc）。差异仅来自模型结构，便于公平对比。

### 4.3 课程通过/不通过（passfail）

六种方法：EduTGT+MLP、纯 MLP、EduTGT+Logistic、纯 Logistic、EduTGT+LSTM、纯 LSTM。

```bash
# 一键多 seed（需先预训练）
python passfail_models/run_paper_eval.py --data_dir data_abc_0.01 --seeds 42,43,44,45,46
```

单独运行示例：

```bash
python passfail_models/main_passfail.py --data_dir data_abc_0.01 --seed 42
python passfail_models/train_mlp.py --data_dir data_abc_0.01 --seed 42
python passfail_models/train_logistic.py --data_dir data_abc_0.01 --mode pure
python passfail_models/train_logistic.py --data_dir data_abc_0.01 --mode contratgt
python passfail_models/train_lstm.py --data_dir data_abc_0.01 --mode pure
python passfail_models/train_lstm.py --data_dir data_abc_0.01 --mode contratgt
```

**消融 / 多 seed 时按 seed 加载预训练**：若已用 `run_ablation.py --seeds 42,43,...` 跑过消融，预训练权重按 seed 存为 `script/pretrain_model/<data_name>_<ablation_suffix>_seed<seed>.pth`。通过/不通过下游可指定同一份权重做课程级评估：

- **main_passfail.py**：支持 `--ablation_suffix` + `--seed`，加载上述预训练路径；自身 checkpoint / saved_models 也按 `<data_name>_<ablation_suffix>_seed<seed>.pth` 保存，多 seed 不互相覆盖。
- **extract_embeddings.py**：增加 `--ablation_suffix`，与 `--seed` 一起拼出预训练路径，提取的表示可给 Logistic/LSTM 用。
- **train_logistic.py / train_lstm.py**：增加 `--ablation_suffix`，在 `--mode contratgt` 时按同一规则拼预训练路径并传给 `run_extract`。

示例（使用 seed 42 的 baseline 消融预训练做通过/不通过）：

```bash
python passfail_models/main_passfail.py --data_dir data_abc_0.01 --ablation_suffix baseline --seed 42
python passfail_models/extract_embeddings.py --data_dir data_abc_0.01 --ablation_suffix baseline --seed 42
python passfail_models/train_logistic.py --data_dir data_abc_0.01 --mode contratgt --ablation_suffix baseline --seed 42
python passfail_models/train_lstm.py --data_dir data_abc_0.01 --mode contratgt --ablation_suffix baseline --seed 42
```

结果目录：`result/passfail/<data_dir>/`。**NN_Test** 表示新节点归纳测试（仅 EduTGT+MLP 计算；纯基线在汇总表中显示为 `-`）。

---

## 五、模块说明（简要）

| 模块 | 作用 |
|------|------|
| **data/convert_oulad.py** | 读 OULAD 七张表，建学生/课程节点与边（时间、通过/不通过标签），生成节点特征，输出 `ml_oulad.csv` + `oulad.content`。 |
| **data/convert_oulad_abc.py** | 按模块筛选后复用 convert_oulad 逻辑，输出到 `data_abc_*`。 |
| **script/pretrain.py** | 在训练边上做对比预训练：正边 vs 负样本边；训练**仅 RandEdgeSampler**（与 ContraTGT 一致）；含 **Top_k + mid_model**、双阶段训练、**同学生时序一致性**；`--no_topk`/`--no_student_consistency`/`--ablation_suffix` 做消融。 |
| **script/run_ablation.py** | 一键消融 + 三基线：5 种消融（预训练→链路） + TGAT/GraphSAGE/JODIE，汇总打印 Test AUC 表。 |
| **script/negative_sampling.py** | 增强负采样（消融用）：伪模块（Jaccard）、时序共现、度感知；`EnhancedNegSampler`。 |
| **script/utils.py** | `get_args`、`get_data_paths`、`Dataset`（按边比/按时间/按 (u,i)）、EarlyStopping。 |
| **script/sampling.py** | `get_adj_list`、`init_offset`、`get_neighbor_list`、`get_unique_node_sequence`。 |
| **script/model.py** | SpatialTemporal（空间/时间 Transformer + MLP 聚合 + 链接头）、Top_k（可学习 top-k 选择）。 |
| **script/graph_transformer.py** | 多头自注意力、LayerNorm、Encoder。 |
| **link_models/main_link.py** | 加载预训练 SpatialTemporal，全模型微调；训练负采样仅 **RandEdgeSampler**（与 ContraTGT 一致）；支持 `--ablation_suffix` 加载消融权重、`--paper_eval` 按边随机划分。 |
| **link_models/data_utils.py** | 边/特征加载、按比例/按时间/按 (u,i) 划分、负采样（同 u、i 从目标节点集采样）。 |
| **link_models/train_link_*** | TGAT/GraphSAGE/JODIE 从零训练链路预测，`--paper_eval` 下与 EduTGT 同划分、同负采样。 |
| **passfail_models/main_passfail.py** | 加载预训练，用课程通过/不通过标签训练（BCE + 正类权重），输出课程级 Train/Val/Test/NN_Test；支持 `--ablation_suffix` + `--seed` 加载消融多 seed 的预训练，自身 checkpoint/saved_models 也按同名带 seed 保存。 |
| **passfail_models/passfail_data.py** | 从边划分得到 (u,i) 划分与标签，构建单向量与序列特征。 |
| **passfail_models/extract_embeddings.py** | 用预训练模型对边前向得到表示，按 (u,i) 聚合成单向量与序列，供 Logistic/LSTM 头使用；支持 `--ablation_suffix` + `--seed` 拼出预训练路径。 |
| **passfail_models/train_mlp.py** | 原始特征 → MLP，BCE + pos_weight。 |
| **passfail_models/train_logistic.py** | 原始特征或预训练表示 → LogisticRegression；`--mode contratgt` 时支持 `--ablation_suffix` + `--seed` 加载对应预训练。 |
| **passfail_models/train_lstm.py** | 原始序列或预训练序列 → LSTM 头；`--mode contratgt` 时支持 `--ablation_suffix` + `--seed` 加载对应预训练。 |

---

## 六、输入输出约定

- **数据**：`data/all_data/<data_dir>/` 下必须有 `ml_oulad.csv`、`oulad.content`；通过 `--data_dir` 指定子目录名（如 `data_abc_0.01`）。
- **预训练权重**：默认 `script/pretrain_model/<data_name>.pth`。消融时带后缀 `<data_name>_<ablation_suffix>.pth`；**多 seed 时按 seed 区分**，保存为 `<data_name>_<ablation_suffix>_seed<seed>.pth`（如 `data_abc_0.01_baseline_seed42.pth`），`script/saved_checkpoints` 同名。main_link、main_passfail、extract_embeddings、train_logistic、train_lstm 均支持 `--ablation_suffix` + `--seed` 加载对应预训练；通过/不通过下游可用指定 seed 的权重做课程级评估。
- **结果**：链路 → `result/link/<data_dir>/`；通过/不通过 → `result/passfail/<data_dir>/`。

---

## 七、评估指标说明

- **链路预测**：边级二分类，Report Train/Val/Test/NN_Test 的 AUC、AP、Acc。NN_Test 为测试集中至少一端为新节点的边。
- **通过/不通过**：课程级（每个 (u,i) 一个标签）。EduTGT+MLP 报告 NN_Test（新节点边）；纯 MLP/Logistic/LSTM 不计算 NN_Test，汇总表中以 `-` 表示。

---

## 八、调参与注意

- **Seed 约定**：pretrain/main_link/main_passfail 使用 `script/utils` 的 `--seed`（默认 60）；TGAT/GraphSAGE/JODIE 脚本默认 42。**run_ablation 单次运行**时若未传 `--seed` 会自动注入 60，保证预训练、链路、三基线同一 seed。可复现性：`init_seeds(seed)` 会设置 random、numpy、torch 及 CUDA 种子；若需严格可复现可另设 `torch.backends.cudnn.deterministic=True`（可能略降速）。
- **预训练**：可调 `--n_epoch`、`--ctx_sample`（建议 20~40）、`--tmp_sample`（建议 20~40）、`--alpha`（建议 0.25~0.5）、`--aug_len`；`--top_k_steps`、`--model_steps` 减小可加速。训练负采样**仅 RandEdgeSampler**（与 ContraTGT 一致）。
- **链路**：默认按时间划分；若需与「按边随机 1:1:8」的论文设置一致，请加 `--paper_eval`。`run_paper_eval.py` 默认不传 `--paper_eval`，即全部按时间划分。
- **通过/不通过**：基线已使用 pos_weight/class_weight 缓解类别不平衡；若仍过拟合可减小 epoch 或加强正则。

---

## 九、相关文档

- **IMPLEMENTATION_AND_DIFFERENCES.md**：EduTGT 做了什么、与 ContraTGT 论文的差异及与基线可比性说明。
