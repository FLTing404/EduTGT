# 2 EduTGT：面向教育时序图的归纳式链路预测方法

本章在 **ContraTGT** 所提出的时空双路图 Transformer 框架之上，面向 **OULAD 学生–VLE 站点** 时序二部图，给出 **EduTGT** 的问题形式化、网络结构、训练目标与实现要点。整体叙事结构参考《融合全局信息和对比学习的图神经网络推荐模型》一文第 2 节「主干 + 多任务/多视角」的组织方式：先明确图与任务，再分述各模块，最后给出联合优化与训练流程。

---

## 2.1 问题形式化与数据划分

将交互日志建模为带时间戳的动态二部图。以下 **全部使用块级公式**（`$$ … $$`），避免行内 `$…$` 在部分 Markdown 预览器中不渲染的问题。

**图与节点集合**

$$
\begin{aligned}
&G=(\mathcal{V},\mathcal{E}),\quad \mathcal{V}=\mathcal{S}\cup\mathcal{R}, \\
&\mathcal{S}\ \text{为学生节点集合},\quad \mathcal{R}\ \text{为 VLE 活动站点集合}.
\end{aligned}
$$

**单条有向交互（边）**

$$
(u,r,t,\ell),\qquad u\in\mathcal{S},\quad r\in\mathcal{R},\quad \ell\in\{0,1\}.
$$

**时间戳与特征**

$$
\begin{aligned}
&t\ \text{为时间戳；实现中由开课实例编码与相对课程日共同构成，与预处理脚本一致。} \\
&X\in\mathbb{R}^{|\mathcal{V}|\times d}\ \text{为节点静态特征矩阵（类别编码、连续特征标准化与维度对齐；见数据管线）。}
\end{aligned}
$$

**归纳式动态链路预测**：在给定时刻之前已观测子图的条件下，对候选边

$$
(u,r,t)
$$

是否发生进行预测。负样本构造为**保持源学生节点与时刻不变，仅将终点站点替换为随机采样站点**（`RandEdgeSampler`），与 ContraTGT 训练协议一致。

**时间外推划分**：对全体边的时间戳取 10% 与 20% 分位数，记为

$$
t_{10\%},\quad t_{20\%},
$$

划分规则为

$$
\begin{aligned}
&t\le t_{10\%}\ \Rightarrow\ \text{训练集}, \\
&t_{10\%}<t\le t_{20\%}\ \Rightarrow\ \text{验证集}, \\
&t>t_{20\%}\ \Rightarrow\ \text{测试集}.
\end{aligned}
$$

（实现见 `utils.Dataset`。）此外单独保留 **新节点子集** `nn_test` / `nn_val`，用于报告测试阶段出现训练边中未见节点的泛化指标。

---

## 2.2 时空双路编码器（Graph Transformer）

对每条训练样本边

$$
(u,r,t),
$$

模型同时利用 **结构邻域** 与 **沿时间轴的交互轨迹**，对应实现中的 **`SpatialTemporal`** 模块（`model.py`）。

### 2.2.1 空间视角：当前时刻之前的结构邻居序列

设当前边对应时刻为

$$
t.
$$

在时刻之前，分别从源节点、终点节点出发，按时间倒序从邻接表中截取长度为

$$
L_s
$$

的邻居序列（`get_neighbor_list`）。序列中各节点取特征后，与相对当前时刻的时间间隔一起输入 **时间编码** `TimeEncode`（支持拼接或求和式时间嵌入），得到张量表示及 mask。

该序列送入 **空间分支**——堆叠

$$
N
$$

层的 **Graph Transformer 编码器**（`graph_transformer.py`：多头自注意力 + 前馈子层 + 残差，掩码屏蔽无效位置），记为

$$
\mathrm{Enc}_{\mathrm{spa}}.
$$

### 2.2.2 时间视角：交互历史序列

对学生端与站点端分别沿各自历史交互链采样长度为

$$
L_t
$$

的序列（`get_unique_node_sequence`），同样经时间编码后输入 **时间分支**

$$
\mathrm{Enc}_{\mathrm{tmp}}.
$$

### 2.2.3 读出与融合

对空间、时间两路编码器输出，分别取 **首位 token 表示** 与 **除首位外的均值池化**，共四维拼接后经 MLP 映射为节点级嵌入（`SpatialTemporal.forward` / `getEmbed`）。该设计与 ContraTGT「空间结构 + 时间轨迹」双路一致，用于刻画教育场景中访问的 **突发性、周期性** 与 **个体行为轨迹**。

---

## 2.3 链路预测头与微调损失

将源节点与真实终点、源节点与负采样终点的嵌入拼接，经线性投影与 `LayerNorm` 后，输入 **亲和度 MLP** `affinity_score` 得到标量 logit，经 Sigmoid 得概率

$$
\hat{y}\in(0,1).
$$

微调阶段（`methods/main.py`）对正、负样本使用 **二元交叉熵（BCE）** 优化：

$$
\mathcal{L}_{\mathrm{BCE}} = -\frac{1}{|\mathcal{B}|}\sum_{(\cdot)\in\mathcal{B}}\bigl[y\log \hat{y}+(1-y)\log(1-\hat{y})\bigr],
$$

其中

$$
\mathcal{B}
$$

为 mini-batch，

$$
y\in\{0,1\}.
$$

验证集上可采用 **AP、AUC、ACC** 等指标；归纳设定下另报告 **新节点子集** 上的 `nn_test_*` 指标。

---

## 2.4 预训练：对比式辅助与序列增强

与 ContraTGT 一致，EduTGT 采用 **预训练 + 微调** 两阶段。预训练脚本为 `pretrain.py`，核心思想是在仅含训练时间窗的边上，学习对时空扰动稳健的表示。

1. **Top‑k 序列重加权（`Top_k`）**  
   以中间模型在空间/时间单路编码上的注意力为引导，对增强后的更长序列进行迭代式重加权与截断，突出信息量高的邻居位置，再与未增强序列组合。

2. **链路 BCE 一致性（教师–学生式）**  
   利用快照模型 `mid_model` 对增强序列的预测作为软标签，训练 `Top_k` 子模块，使增强视图下的链路预测与教师一致（实现中的 `criterion` + `pos_label/neg_label` 由教师预测阈值生成）。

3. **余弦一致性 / 对比项**  
   - 对 `Top_k` 分支：增强前后嵌入的 **余弦嵌入损失**（`CosineEmbeddingLoss`），与 BCE 项加权相加；权重记为

     $$
     \alpha,
     $$

     代码中默认取

     $$
     0.6.
     $$

   - 对主模型：将 **增强视图嵌入** 与 **原始视图嵌入** 拉近，将 **增强视图** 与 **随机负样本终点视图** 推远（`pos_loss` + `neg_loss`），以无额外标签方式强化判别边界。

4. **早停**  
   预训练阶段对验证目标采用耐心度（如 `max_round=5`）早停，保存最优权重至 `outputs/pretrain/`。

---

## 2.5 教育场景扩展：课程实例先验与消融变体（可选）

在 OULAD 整模块数据中，同一课程可能存在多个 **presentation（开课实例）**。工程上可由各实例下站点集合的 **Jaccard 相似度** 构建关系矩阵

$$
W
$$

（`course_pres.build_pres_relation_dict`），并写入 `pres_relation.json`。

- **Presentation 偏置负采样（PANS）**：训练时将随机负样本终点按

  $$
  W
  $$

  与当前边隐含的实例信息混合采样（`make_train_pres_biased_sampler`，可调均匀混合比例 `pres_mix_uniform`），使负例更贴近「同学段但语义相近」的硬负样本，对应扩展训练入口 `methods/main_ablation_pres_relation.py`（完整模型常记为 **EduTGT + PANS**）。

- **其它消融脚本（与第 3 章实验对应）**  
  - `main_ablation_student_temporal.py`：在 batch 内对同源节点的嵌入施加 **时序一致性** 正则；  
  - `main_ablation_course_progress.py`：将时间戳归一化到区间

    $$
    [0,1],
    $$

    以 **课程进度** 替代绝对时间尺度，用于分析时间语义消融。

具体符号命名以实验章节表格为准，本章仅说明其与代码模块的对应关系。

---

## 2.6 总体训练流程（算法式描述）

**输入**：`ml_*.csv`（边列表与时间戳）、`*.content`（节点特征）。  
**输出**：微调后的 `SpatialTemporal` 权重（`outputs/models/`）。

1. **划分**：按时间分位数得到 train/val/test 与 nn 子集。  
2. **预训练**：在训练边上采样时空序列 → 双路 Transformer 编码 → Top‑k 增强 + BCE/余弦多任务 → 早停保存。  
3. **微调**：加载预训练权重，在同一划分下以 BCE 优化链路预测头直至早停（如 `max_round=8`）。  
4. **推理**：对测试边与 nn 子集计算 AUC、AP、ACC 等。

