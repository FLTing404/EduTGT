# 方法思路与 ContraTGT 对比

本文面向 **论文写作与答辩**：概括 **ContraTGT（IJCAI）** 类方法在动态图上的建模思路，说明 **EduTGT** 如何实现同一主线，以及在 **数据、工程与实验扩展** 上与并列 **ContraTGT 官方仓库** 的差异。

> 说明：具体公式与证明以 **ContraTGT 原文** 为准；此处侧重 **实现级对照** 与本仓库 **可写进论文的叙事结构**。

---

## 1. 问题设定：归纳式动态链路预测

- **图**：随时间演化的交互图，边带时间戳 \(t\)（离散步或编码后的相对时间）。  
- **目标**：对给定 **源节点 \(u\)、候选终点 \(v\)、时刻 \(t\)**，预测该交互是否会发生（或等价地，对正负边打分）。  
- **归纳性**：测试阶段可出现 **训练边中未出现过的节点**；因此除全量测试集外，常单独报告 **new node** 子集上的指标（本仓库 `utils.Dataset` 中的 **`nn_test`**，与 ContraTGT / EduTGT 的 `main.py` 一致）。

**时间切分（与官方一致）**：对全体边的时间戳取 **10% / 20% 分位数** 得到两个阈值，划分为 train / val / test 三段（按时间外推，而非随机边对留出）。详见 `README_oulad.md` 第 5 节。

---

## 2. 核心模型：时空双路编码 + 链路打分

EduTGT 中主体网络为 **`SpatialTemporal`**（`model.py`），与 ContraTGT 参考实现 **同构思想**：

### 2.1 空间视角（结构邻居）

- 对当前边 \((u,v,t)\)，在 **\(t\) 之前** 的图上，为 \(u\)、\(v\) 各采样一段 **空间邻居序列**（`sampling.get_neighbor_list`：按时间倒序的邻接事件）。  
- 邻居节点特征经 **时间编码**（`TimeEncode`：相对当前 \(t\) 的间隔与可学习谐波等）后，送入 **Graph Transformer 式 Encoder**（多头注意力 + 前馈，`graph_transformer.py`）。  
- 将序列输出聚合为 **空间侧表示**（实现中取首位置与除首外的均值等组合，与原文结构对应）。

### 2.2 时间视角（交互历史）

- 对 \(u\)、\(v\) 分别沿各自 **历史交互链** 采样序列（`get_unique_node_sequence`），同样经时间编码后进入 **另一路 Temporal Encoder**。  
- 得到 **时间侧表示**，与空间侧 **拼接**，经 **MLP** 得到该节点在当前上下文下的 **嵌入**（`getEmbed`）。

### 2.3 链路预测头

- 正样本：真实 \((u,v,t)\) 的嵌入拼接；负样本：将 \(v\) 换为 **随机采样的假终点**（`RandEdgeSampler`，从训练中出现过的终点集合抽样，与 ContraTGT 训练逻辑一致）。  
- 拼接后经线性层与 **`affinity_score`（小型 MLP）** 得到正负 **logit/sigmoid 分数**，用 **BCE** 训练。

**论文可强调的要点**：同时显式建模 **「空间邻域结构」** 与 **「沿时间轴的交互轨迹」**，再用 **Transformer 式编码** 融合，适用于教育场景中学生–资源交互的 **_burst 与周期性**。

---

## 3. 训练流程：预训练 + 微调

与 ContraTGT 实践一致，EduTGT 采用 **两阶段**：

1. **`pretrain.py`（自监督 / 辅助任务预训练）**  
   - 在 **仅训练时间窗** 的边上构造 batch；组合 **链路 BCE**、**Top-k 排序式损失**、以及基于 **中间模型快照** 的 **表示一致性 / 散度类正则**（余弦约束等，见 `pretrain.py` 循环内 `divloss_*`）。  
   - 产出 **`outputs/pretrain/<data_name>.pth`**，并配合 **`outputs/middle/`**、**`outputs/checkpoints/`** 早停。

2. **`main.py`（微调）**  
   - 加载同一 **`SpatialTemporal`** 的预训练权重，在相同数据与划分下继续 **链路 BCE** 训练，早停后写入 **`outputs/models/<data_name>.pth`**。  

**写作建议**：预训练可表述为 **「多任务驱动的图时空表示学习」**；微调为 **「面向链路预测的适配」**。若论文重点在 **OULAD 或消融**，无需把预训练单独列为第二贡献，但须在实验设置中写清 **两阶段与超参**（见 `使用指南.md`）。

---

## 4. 本仓库相对 ContraTGT **官方目录**的差异

下列对照针对并列 **`ContraTGT/ContraTGT`** 类上游仓库与 **`EduTGT/EduTGT`** 本工程（非论文理论差异，而是 **实现与数据管线**）。

| 维度 | ContraTGT 官方脚本习惯 | EduTGT |
|------|-------------------------|--------|
| **数据** | 内置数据集名（如 wiki、slashdot）与固定相对路径 | **OULAD** 预处理 → `data/processed/data_*`；`--data_dir` 指向含 `ml_*.csv`、`.content` 的目录 |
| **路径** | 常见 `data/ml_{name}.csv`、`node_feature/{name}.content` | 统一由 **`paths.py`** 管理 **`outputs/pretrain|checkpoints|models|middle|logs`** |
| **`Dataset` / 采样** | `utils.Dataset`、`RandEdgeSampler`、`get_neighbor_list` 同源逻辑 | **同一套** `utils.py` / `sampling.py` 语义，保证与论文设定一致 |
| **扩展实验** | 官方仓库以原设定为主 | **`main_ablation_pres_relation.py`**：训练期按 **presentation 关系矩阵 W** 偏置负样本；**`main_ablation_student_temporal.py`**：batch 内同源节点时序一致性项 |
| **基线对比** | 无（或自备） | **`baseline/`**：TGN、TGAT、JODIE 风格等，与主流程 **共享 `Dataset` 与划分** |
| **结果记录** | 终端为主 | **`outputs/logs/training_runs.jsonl`** 自动追加 JSON 行（`result_logger.py`） |

**结论（可写进论文）**：*EduTGT 在模型与优化主线上遵循 ContraTGT 的时空图 Transformer 链路预测框架；在 **教育数据构建、训练产物组织、以及面向课程实例的负采样与一致性消融** 上做了工程化与问题驱动的扩展。*

---

## 5. 任务语义边界（与「标准 link prediction」文献）

OULAD 构图多为 **学生 ↔ VLE 站点** 二部图，负样本为 **替换终点** 而非显式未观测边表；切分为 **时间分位** 而非随机边对留出。与部分 link prediction 论文的 **负例协议、切分协议** 可能不完全一致。详细讨论与改进方向见 **`README_oulad.md` 第 3–4 节**——建议在论文 **「实验设置 / 局限」** 中简要交代。

---

## 6. 引用与仓库关系（写作提示）

- **方法主线**：引用 **ContraTGT 原文**（IJCAI）作为 **动态图归纳表示 + 链路预测** 的方法来源。  
- **实现来源**：说明 EduTGT **基于公开 ContraTGT 代码演进**，并列出 **本仓库特有模块**（OULAD 预处理、`course_pres` / 消融脚本、`baseline`、`paths` 与日志等）。  
- 若投稿要求 **可复现**：给出 **`使用指南.md`** 中的命令链、`--data_dir` 与 **`outputs/`** 产物路径即可。

---

## 7. 延伸阅读

- **命令与目录**：`使用指南.md`  
- **OULAD 字段与切分**：`README_oulad.md`  
- **基线**：`../baseline/README.md`
