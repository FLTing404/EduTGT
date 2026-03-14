# EduTGT 与 ContraTGT 论文的关系与差异

本文档说明 **EduTGT**（本仓库算法名）在做什么、与 **ContraTGT 论文**（Contrastive Temporal Graph Transformer）的**目标一致性**与**主要差异**。EduTGT 与 ContraTGT 同一目标（时序图对比学习 + 链路预测），并与 TGAT/GraphSAGE/JODIE 等基线在相同划分与负采样下公平可比。**最稳**：baseline 与 EduTGT 使用相同时间划分、相同评估负采样池、相同 early stopping 规则；差异仅来自模型结构。

---

## 一、本代码在做什么（整体流程）

### 1. 数据与任务

- **数据**：OULAD 教育数据 → 时序二部图（学生–课程边，带时间戳与通过/不通过标签）。
- **两个下游**：
  1. **链路预测**：预测边是否存在（二分类），边级 AUC/AP/Acc。
  2. **课程通过/不通过**：对每个 (学生, 课程) 预测是否通过，课程级 AUC/AP/Acc。

流程概括：**先预训练得到图表示 → 再在链路上微调 / 在通过不通过上接分类头**，并与多种基线公平对比。

### 2. 预训练（script/，EduTGT 预训练）

- **输入**：边表 `ml_oulad.csv` + 节点特征 `oulad.content`，默认按**时间**划分（quantile 0.1, 0.2），只用约 10% 训练边；可选 `--paper_eval` 时按边随机 1:1:8。
- **做法**：
  - 对每条训练边 (u, i, ts)，构造**正样本** (u, i) 与**负样本** (u, fake_i)；fake_i 由 **RandEdgeSampler** 得到（与 ContraTGT 一致）。
  - 用 **SpatialTemporal** 编码 (u, i) 的时空邻居与时间序列（**neighbor_ts < current_ts**，避免 temporal leakage），得到表示，再经**链接头**得到正/负得分，用 BCE 做对比学习。
- **EduTGT 在预训练中增加**：Top_k 自适应增强、双阶段交替（mid_model 教师）、同学生时序一致性正则；消融可通过 `--no_topk` / `--no_student_consistency` 关闭对应部分。
- **输出**：默认 `script/pretrain_model/<data_name>.pth`。消融时 `--ablation_suffix` 则保存为 `<data_name>_<suffix>.pth`；**多 seed 消融时按 seed 区分**，保存为 `<data_name>_<suffix>_seed<seed>.pth`（如 `data_abc_0.01_baseline_seed42.pth`），`script/saved_checkpoints` 同名，供链路与通过/不通过按 `--ablation_suffix` + `--seed` 加载。

### 3. 链路预测（link_models/）

- **EduTGT**：加载预训练 SpatialTemporal，全模型微调；训练负采样**仅 RandEdgeSampler**（与 ContraTGT 一致）；Val/Test 负采样池为全量边目标节点；默认按时间划分，可选 `--paper_eval` 按边随机 1:1:8。支持 `--ablation_suffix` + `--seed` 加载消融多 seed 的预训练（`<data_name>_<suffix>_seed<seed>.pth`）；链路 checkpoint / saved_models 也按同一命名带 seed，多 seed 不互相覆盖。
- **基线**：TGAT、GraphSAGE、JODIE 从零训练链路模型，同一划分、同一 Val/Test 负采样规则，便于公平对比。

### 4. 课程通过/不通过（passfail_models/）

- **EduTGT+MLP**：加载预训练，用**通过/不通过标签**训练（BCE + 正类权重），按 (u,i) 聚合为课程级指标，并单独报告 **NN_Test**（新节点边）。支持 `--ablation_suffix` + `--seed` 加载消融多 seed 的预训练；自身 checkpoint / saved_models 也按 `<data_name>_<suffix>_seed<seed>.pth` 保存，多 seed 不覆盖。
- **extract_embeddings / train_logistic / train_lstm**：支持 `--ablation_suffix` + `--seed`，在 `mode=contratgt` 时按同一规则拼预训练路径并加载，便于用指定 seed 的消融预训练做课程级评估。
- **纯 MLP / Logistic / LSTM**：用原始特征（或预训练表示）接对应头，同样 (u,i) 划分；MLP/LSTM 使用 pos_weight，Logistic 使用 class_weight='balanced'。
- **run_paper_eval**：六种方法多 seed 跑并汇总；NN_Test 仅 EduTGT 有，其余在表中标为 `-`。

---

## 二、与 ContraTGT 论文的差异

下面按「预训练」和「下游使用」分开说。

### 2.1 预训练部分

| 方面 | ContraTGT 论文（典型描述） | EduTGT |
|------|---------------------------|--------|
| **训练方式** | 单阶段：一个模型 + 链接头，正/负边 BCE 对比 | 默认可**双阶段交替**（`--no_topk` 可关）：先训 Top_k 再训主模型，每 batch 内多轮；引入 mid_model 教师 |
| **负采样** | 随机从目标节点采样负目标（RandEdgeSampler） | **同论文**：仅 RandEdgeSampler |
| **数据增强** | 固定或简单随机增强 | **Top_k 可学习增强**（可关）：对加长空间/时间序列做 top-k 选择，一致性损失 + 多样性损失（cosine）；α 建议 0.25~0.5（consistency > diversity） |
| **损失** | BCE 对比（正 vs 负） | BCE + **一致性**（增强视图与原视图预测一致）+ **多样性**（α·cos）+ **同学生时序一致性**（可关） |
| **训练量** | 常见 50 epoch 左右 | 默认**至少 80 epoch**，early stopping max_round=3 |
| **时序约束** | neighbor 需 ts_neighbor < ts_current（防 leakage） | **一致**：sampling 中仅取 neighbor_ts < current_ts |

也就是说：**EduTGT 与 ContraTGT 在训练负采样上一致（仅 RandEdgeSampler）**；预训练中增加 Top_k、双阶段、同学生一致性等，可通过消融对比。

### 2.2 下游部分

- **链路预测**：与论文一致，加载预训练、全模型微调；**训练仅 RandEdgeSampler**（与 ContraTGT 一致）。与基线使用相同时间划分、相同评估负采样池、相同 early stopping；差异仅来自模型结构。
- **通过/不通过**：论文中未必细说；EduTGT 用同一预训练权重 + BCE 与正类权重，并区分课程级 Test 与 NN_Test（新节点归纳）。

### 2.3 小结表

| 模块 | 论文 ContraTGT | EduTGT |
|------|----------------|--------|
| 时空编码 + 链接头 | ✓ | ✓（同一套 SpatialTemporal + 链接头） |
| 对比学习（正/负边） | ✓ | ✓ |
| 训练负采样 | RandEdgeSampler | **同论文**：仅 RandEdgeSampler |
| 同学生时序一致性 | ✗ | 预训练中增加（消融可关） |
| Top_k 自适应增强 | ✗ | 预训练中增加（消融可关） |
| 双阶段 + 教师 mid_model | ✗ | 与 Top_k 同开同关 |
| 时序约束（ts_neighbor < ts） | 需满足防 leakage | ✓（sampling 已保证） |
| 预训练 epoch / early stop | 常见较少 | 80+ epoch，max_round=3 |
| 消融 / 超参 | - | `run_ablation.py` 一键跑 5 种配置 + 三基线；`--seeds 42,43,44,45,46` 多 seed 时输出 per-seed CSV 与汇总表（Mean_AUC, Std_AUC）；α、ctx_sample、tmp_sample 见 README |
| 按 seed 保存/加载 | - | 消融多 seed 时 pretrain/checkpoint 为 `<data_name>_<suffix>_seed<seed>.pth`；main_link、main_passfail、extract、train_logistic、train_lstm 均支持 `--ablation_suffix` + `--seed` 加载 |
| 链路/通过不通过下游 | 思路一致 | 一致，训练仅 RandEdgeSampler；通过/不通过可指定某 seed 的消融预训练做课程级评估 |

---

## 三、文件与功能对应（便于对照）

- **负采样**：训练**仅 RandEdgeSampler**（`script/utils.py`），与 ContraTGT 一致；**链路训练同样仅 RandEdgeSampler**。
- **Top_k 与多损失**：`script/model.py`（`Top_k` 类）、`script/pretrain.py` 中 Top_k 阶段与 model 阶段的损失（一致性、多样性、对比）；同学生时序一致性在 model 阶段按 batch 内学生分组做方差正则；`--no_topk` / `--no_student_consistency` 可关闭对应部分。
- **预训练入口**：`script/pretrain.py`（数据加载、划分、仅 RandEdgeSampler、训练循环；`--ablation_suffix` 时保存带后缀；多 seed 消融时路径带 `_seed<seed>`，与下游加载约定一致）。
- **消融**：`link_models/run_ablation.py` 一键跑 5 种配置 + TGAT/GraphSAGE/JODIE；`--seeds 42,43,...` 时每 seed 跑一轮并写 `result/link/<data_name>/ablation_results_seed_<seed>.csv`，最后写 `ablation_summary_<data_name>.csv`（Mean_AUC, Std_AUC 及各 seed 列）。
- **链路**：`link_models/main_link.py`（加载预训练、全模型微调、仅 RandEdgeSampler；`--ablation_suffix` + `--seed` 时加载/保存带 seed 的路径）、`link_models/data_utils.py`（与 script 对齐的划分与负采样）。
- **通过/不通过**：`passfail_models/main_passfail.py`（支持 `--ablation_suffix` + `--seed` 加载对应预训练，自身 checkpoint 也按 seed 区分）、`passfail_data.py`、`extract_embeddings.py`（支持 `--ablation_suffix` + `--seed`）、`train_logistic.py` / `train_lstm.py`（`mode=contratgt` 时支持 `--ablation_suffix` + `--seed`），以及各脚本中的 pos_weight/class_weight 处理。

若只想复现「与论文一致」的 ContraTGT 训练方式：加 `--no_topk`、`--no_student_consistency` 即等价单阶段 BCE；**训练负采样与论文一致（仅 RandEdgeSampler）**。

---

## 四、与 ContraTGT 原代码对齐情况

目标：在 OULAD 上跑 ContraTGT 算法，在**数据划分、训练/Val/Test 负采样、early stopping** 上与 ContraTGT 一致；EduTGT 在预训练中增加 Top_k、双阶段、同学生时序一致性等，差异仅来自模型与训练方式，评估协议一致。

### 4.1 预训练（script/pretrain.py）

| 项目 | ContraTGT 原 pretrain.py | EduTGT pretrain.py | 是否一致 |
|------|-------------------------|---------------------|----------|
| 模型结构 | SpatialTemporal + Top_k + mid_model | 同左（`--no_topk` 时不建 Top_k/mid_model） | ✓/可关 |
| 训练方式 | 双阶段（先训 Top_k，再训 model），一致性/多样性损失 | 同左 + **同学生时序一致性**（`--no_student_consistency` 可关） | 扩展/可关 |
| 负采样 | `RandEdgeSampler(train_data['idx'][:,1])` 随机 | **同左**：仅 RandEdgeSampler | ✓ |
| 负采样池 | 训练边目标节点 | 训练边目标节点（two_stage 仅改变采样分布） | ✓ |
| 数据划分 | 按**时间** quantile 0.1/0.2 | 按**时间** quantile 0.1/0.2（train_ratio=None） | ✓ |
| 时序约束 | neighbor_ts < current_ts | 同左（sampling 中保证） | ✓ |
| Early stopping / 保存路径 | max_round=5（原 pretrain） | max_round=3；`--ablation_suffix` 时 checkpoint 与 pretrain 保存路径带后缀；多 seed 时带 `_seed<seed>` | ✓ |

**结论**：预训练在划分、负采样（仅 RandEdgeSampler）、early stopping、时序约束上与 ContraTGT 一致；EduTGT 在预训练中增加 Top_k、双阶段、同学生一致性，支持消融对比。

### 4.2 链路预测（link_models/main_link.py）

| 项目 | ContraTGT 原 main.py | EduTGT main_link.py | 是否一致 |
|------|----------------------|----------------------|----------|
| 数据划分 | 按**时间** quantile 0.1/0.2 | 默认按**时间** quantile 0.1/0.2（不传 --paper_eval 时） | ✓ |
| 加载预训练 | 加载 pretrain_model/*.pth，全模型微调 | 同左；`--ablation_suffix` 时加载 `<data_name>_<suffix>.pth`；多 seed 时为 `<data_name>_<suffix>_seed<seed>.pth`，链路 checkpoint/saved_models 同名 | ✓ |
| 训练负采样 | `RandEdgeSampler(train_data['idx'][:,1])` | **同左**（仅 RandEdgeSampler，与论文一致） | ✓ |
| **Val/Test 负采样** | `RandEdgeSampler(edges['idx'][:,1])` 全量边目标节点 | 同左 | ✓ |
| 损失与前向 | BCE，linkPredict(pos, neg) | 同左 | ✓ |
| Early stopping | 默认 max_round=3 | max_round=3 | ✓ |

**结论**：链路在划分、训练/Val/Test 负采样、early stopping 上与 ContraTGT 一致；通过 `--ablation_suffix` 可加载消融预训练权重。

### 4.3 基线对比（TGAT / GraphSAGE / JODIE）与公平性

已与 EduTGT 统一，保证可比、公平（算法差异会直接体现为得分差异）：

| 项目 | 说明 |
|------|------|
| **数据划分** | 默认（不传 `--paper_eval`）时，预训练、main_link、TGAT、GraphSAGE、JODIE 均按**时间** quantile 0.1/0.2（10% train, 10% val, 80% test）。传 `--paper_eval` 时均按边随机 1:1:8。`run_ablation.py` / `run_paper_eval.py` 调用基线时不传 `--paper_eval`，故默认全部用时间划分。 |
| **训练负采样** | 所有基线训练时负样本目标节点均来自**训练集目标节点** `dst_train = get_dst_nodes(i_tr)`（与 EduTGT 的 RandEdgeSampler 池一致），同 u 下随机采 neg_i。 |
| **Val/Test 负采样池** | 所有方法 Val/Test 均使用**全量边目标节点** `all_dst = unique(i_tr ∪ i_val ∪ i_te)`，与 ContraTGT/EduTGT 一致；评估时每条正边配一条同 u 的随机负边。 |
| **评估指标** | 统一报告 Train/Val/Test/NN_Test 的 AUC、AP、Acc；NN_Test 为测试集中至少一端为新节点的边。 |
| **依赖** | TGAT、JODIE 仅需 torch；GraphSAGE 需 PyTorch Geometric。三者接口一致（`--data_dir`、`--seed`、`--paper_eval`），输出 `link_<name>_<data_name>.csv` 供消融/汇总读取。 |
| **Early stopping** | 预训练与 main_link 均为 max_round=3；基线按固定 epoch 训练（TGAT/GraphSAGE/JODIE 默认 80/100/80），与 ContraTGT 常见设置可比。 |

这样对比组与 EduTGT 同一划分、同一负采样协议、同一指标，无额外优势，得分差异来自方法本身。

---

## 五、Seed 贯穿逻辑（科研复现）

为满足论文级可复现与多 seed 报告 mean±std，**同一 seed 必须贯穿**：预训练 → 链路（或通过/不通过）→ 评估；且不同 seed 之间**仅随机性不同**，数据划分/协议一致。

### 5.1 谁提供 seed

| 脚本/入口 | 种子来源 | 默认值 |
|-----------|----------|--------|
| script/pretrain.py | `get_args()`（script/utils）的 `--seed` | 60 |
| link_models/main_link.py | 同上 | 60 |
| passfail_models/main_passfail.py | 同上 | 60 |
| TGAT/GraphSAGE/JODIE | 各自脚本的 `--seed` | 42 |
| link_models/run_ablation.py | 单次：未传则注入 `--seed 60`；多 seed：`--seeds` 每轮注入当前 seed | 60 / 用户指定 |
| link_models/run_paper_eval.py | 每轮传入当前 `--seed`（edutgt 先 pretrain 再 main_link，同一 seed） | 42,43,44,45,46 |

### 5.2 何处使用 seed（贯穿链）

- **预训练（pretrain.py）**：脚本开头 `init_seeds(seed)`（random / numpy / torch / CUDA）；`Dataset(..., random_state=seed)` 决定 train/val/test 划分；`EnhancedNegSampler(..., seed=seed)`（若有）；保存路径在有 `--ablation_suffix` 时带 `_seed<seed>`，如 `<data>_baseline_seed42.pth` 或 `<data>_paper_eval_seed42.pth`。
- **链路（main_link.py）**：开头 `init_seeds(seed)`；`Dataset(..., random_state=args.seed)`；加载预训练时路径与 pretrain 约定一致（`--ablation_suffix` + `--seed` → `_<suffix>_seed<seed>.pth`）；`eval_epoch` 内评估前再次 `init_seeds(seed)` 保证 Val/Test 负采样可复现。
- **通过/不通过（main_passfail.py）**：开头 `init_seeds(seed)`；`Dataset(..., random_state=args.seed)`；加载预训练路径同上；eval/collect 内 `init_seeds(seed)`。
- **基线（TGAT/GraphSAGE/JODIE）**：脚本内 `torch.manual_seed(seed)`、`np.random.seed(seed)`，划分与负采样均传入同一 seed。

### 5.3 两套多 seed 流程（无逻辑冲突）

1. **run_ablation.py（消融 + 三基线）**  
   - 单次：不传 `--seed` 时自动注入 `60`，pretrain / main_link / 三基线均收到 `--seed 60`。  
   - 多 seed：`--seeds 42,43,...`，每轮 base 带 `--seed <s>`，先跑 5 种消融（每种 pretrain(s)→main_link(s)），再跑 3 基线(s)。同一 seed 下预训练与链路一一对应，权重按 `<data>_<suffix>_seed<seed>.pth` 存/读。  
2. **run_paper_eval.py（论文 Table 多 seed 汇总）**  
   - 仅链路：对每个 seed 先 `pretrain --ablation_suffix paper_eval --seed <s>`（存 `<data>_paper_eval_seed<s>.pth`），再 `main_link --ablation_suffix paper_eval --seed <s>`（读同一文件）；TGAT/GraphSAGE/JODIE 仅 `--seed <s>`。  
   - 预训练按 seed 存盘，不与消融的 baseline/neg 等混用；汇总时用各 seed 的 CSV 算 mean±std。

### 5.4 科研角度的合理性

- **单 seed 可复现**：同一命令（含 `--seed`）多次运行，数据划分、模型初始化、训练/评估中的随机性一致（init_seeds 覆盖 random/numpy/torch/cuda）。  
- **多 seed 独立**：每个 seed 对应一次独立的「预训练→下游→评估」链，无跨 seed 泄露；报告 mean±std 时各 seed 等价于独立重复实验。  
- **消融与 paper_eval 隔离**：消融用 `baseline/neg/topk/...` 后缀，paper_eval 用 `paper_eval` 后缀，预训练权重与下游加载路径一一对应，无混用或覆盖错误。

---

## 六、超参与消融

- **超参建议**（论文/实践）：`ctx_sample` / `tmp_sample`（ls, lt）建议 20~40，默认 30/21；`alpha`（diversity 权重）建议 0.25~0.5，默认 0.35（consistency > diversity）；embedding 维度 128 或 256；masking ratio ρ∈[0.25,0.5] 时较稳定。
- **消融**：`python link_models/run_ablation.py --data_dir <dir>` 一键跑 5 种配置 + TGAT/GraphSAGE/JODIE，汇总 Test AUC。单次运行可用 `--seed 60` 统一种子；**多 seed** 用 `--seeds 42,43,44,45,46`，每 seed 跑一轮，写入 `result/link/<data_name>/ablation_results_seed_<seed>.csv`，最后生成 `ablation_summary_<data_name>.csv`（Mean_AUC, Std_AUC 及各 seed 列），便于论文报告 mean±std。单次预训练或链路用 `--ablation_suffix <name>`（及多 seed 时 `--seed <s>`）指定保存/加载的权重文件名。
- **下游按 seed 加载**：消融多 seed 后，预训练权重为 `script/pretrain_model/<data_name>_<suffix>_seed<seed>.pth`。main_link、main_passfail、extract_embeddings、train_logistic、train_lstm 均支持 `--ablation_suffix` + `--seed` 加载对应权重；通过/不通过可用指定 seed 的预训练做课程级评估（见 README 4.3）。
