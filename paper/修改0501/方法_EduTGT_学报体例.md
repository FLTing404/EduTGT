# 1 本研究 EduTGT 方法

面向开放大学学习分析数据集（Open University Learning Analytics Dataset，OULAD）中学生与虚拟学习环境（Virtual Learning Environment，VLE）站点之间的时序交互日志，本文在对比式时空图 Transformer（Contrastive Pre-training and Learnable Data Augmentation of Temporal Graph Networks，**ContraTGT**）^[1]^ 的双路序列建模与对比预训练框架之上，给出 **教育场景时空图 Transformer（Educational Temporal Graph Transformer，EduTGT）**。EduTGT 保留 ContraTGT 的「空间邻域序列 + 时间历史序列」双视角 Graph Transformer 编码、可学习序列掩码增强与对比预训练—微调流程；在此基础上引入 **开课实例（presentation）** 层面的站点关联先验，并通过 **开课实例偏置负采样（presentation-biased negative sampling，PANS）** 构造更具课程语义难度的负例，以适配教育学场景下的归纳式动态链路预测任务。

---

## 1.1 教育时序二部图与动态链路预测问题

远程开放课程中的点击流可建模为带时间戳的动态二部图。设图

$$
G=\langle \mathcal{V},\mathcal{E}\rangle,\quad \mathcal{V}=\mathcal{S}\cup \mathcal{R},
$$

其中 $\mathcal{S}=\{s_1,s_2,\ldots,s_{|\mathcal{S}|}\}$ 为学生节点集合，$\mathcal{R}=\{r_1,r_2,\ldots,r_{|\mathcal{R}|}\}$ 为 VLE 站点节点集合；$\mathcal{E}$ 为有向交互边集合，每条边可写为四元组 $(u,i,t,\ell)$，表示时刻 $t$ 学生 $u\in\mathcal{S}$ 对站点 $i\in\mathcal{R}$ 发生访问交互，$\ell\in\{0,1\}$ 为链路标签（观测日志中为正样本时 $\ell=1$）。记节点静态特征矩阵为 $X\in\mathbb{R}^{|\mathcal{V}|\times F}$，由人口学与站点属性等经编码、标准化得到；$t$ 由各开课实例内的相对课程日与实例编码共同构成可排序时间戳，以保证多实例合并时的时序可比性。

本文关注 **归纳式动态链路预测（inductive dynamic link prediction）**：给定候选三元组 $(u,i,t)$，依据 $u$、$i$ 在历史时刻不超过 $t$ 的子图条件下得到的表示，预测该交互是否发生。微调阶段对未观测边采用 **二元交叉熵（Binary Cross Entropy，BCE）** 训练；负样本在均匀随机分布下替换终点站点 $i$ 而保持 $u$、$t$ 不变，与 ContraTGT 协议一致^[1]^。数据集按全体边时间戳的 $10\%$、$20\%$ 分位数划分训练 / 验证 / 测试集，实现时间外推评测；并可单独报告测试中出现训练未覆盖节点的 **新节点（new node）** 子集指标。

---

## 1.2 EduTGT 整体流程

如图 1 所示（示意图结构与 ContraTGT 文献 Fig.~1 一致^[1]^），EduTGT 整体包含：**（a）输入时序二部图；**（b）**双路序列采样**——分别从「当前时刻之前的结构邻域」与「目标节点沿交互链的历史轨迹」构造空间序列与时间序列，并为序列中各位置附加时间编码；**（c）对比预训练**——在可学习掩码增强生成的增强序列上，联合链路一致性、表示多样性及对比项优化 Graph Transformer；**（d）监督微调**——在链路标签上优化预测头。教育场景扩展模块在 **（c）（d）** 之间的负采样环节引入 presentation 级先验矩阵，见 **1.6 节**。

**Fig. 1  EduTGT 总体框架示意（双路采样—对比预训练—链路微调；教育扩展为偏置负采样）**  
**Fig. 1  Overall pipeline of EduTGT (dual-view sampling, contrastive pre-training, link fine-tuning; educational extension via biased negative sampling)**

---

## 1.3 双路序列采样与时空 Graph Transformer 编码

### 1.3.1 空间与时间序列构造

**空间采样**以目标节点为中心，自其在时刻 $t$ 之前的邻接交互出发，按与各邻居的最后交互时间 **递减** 排序，截取长度为 $l_s$ 的邻居序列，并在序列首位拼接目标节点本身，得到空间视角序列；对每个位置计算与当前时刻 $t$ 的时间间隔，经时间编码后与节点特征拼接，作为 Transformer 输入^[1]^。

**时间采样**以目标节点最近一次交互为锚，沿其历史交互链回溯，截取长度为 $l_t$ 的轨迹序列；同样对各位置施加时间编码^[1]^。对学生节点与站点节点可分别沿各自轨迹采样，以刻画「学习者行为演化」与「资源被访问的动态模式」。

### 1.3.2 Graph Transformer 层

记第 $l$ 层输入序列表示为 $H^{(l-1)}=[h^{(l-1)}_0,\ldots,h^{(l-1)}_{n-1}]\in\mathbb{R}^{n\times d_{l-1}}$。单层多头自注意力中，查询、键、值矩阵为^[1]^

$$
Q^{(l)}=H^{(l-1)}W_Q^{(l)},\quad K^{(l)}=H^{(l-1)}W_K^{(l)},\quad V^{(l)}=H^{(l-1)}W_V^{(l)},
\tag{1}
$$

注意力输出为^[1]^

$$
\mathrm{ATT}\bigl(H^{(l-1)}\bigr)=\mathrm{SOFTMAX}\!\left(\frac{Q^{(l)}\bigl(K^{(l)}\bigr)^{\!\top}}{\sqrt{d_K^{(l)}}}\right)V^{(l)}.
\tag{2}
$$

经前馈子层与残差、层归一化（Layer Normalization，LN）^[2]^ 堆叠 $L$ 层后得到序列表示 $H^{(L)}$^[1]^：

$$
H^{(l)}=\mathrm{LN}\!\left(\mathrm{FFN}\!\left(\mathrm{LN}\!\left(\mathrm{ATT}(H^{(l-1)})+H^{(l-1)}\right)\right)+e_H^{(l)}\right).
\tag{3}
$$

式中：$\mathrm{FFN}(\cdot)$ 为前馈网络；$e_H^{(l)}$ 为可选位置或类型嵌入；各矩阵维数与 ContraTGT 实现一致^[1]^。

### 1.3.3 双视角融合与节点表示

对空间分支与时间分支分别独立编码后，将同一目标位置上的两路表示拼接，得到用于链路预测的节点向量^[1]^。记空间、时间输出分别为 $H_s$、$H_t$，则节点 $v$ 的表示可取对应索引拼接：

$$
h_v = H_s[v]\ \|\ H_t[v],
\tag{4}
$$

其中 $\|$ 表示向量拼接。实现中还可对序列首位 token 与其余位置的均值池化进行组合读出（与 ContraTGT / 代码实现一致），以增强稳定性。

---

## 1.4 链路预测头与微调损失

对候选交互 $(u,i,t)$，由双路编码得到 $h_u$、$h_i$，链路存在概率写为^[1]^

$$
p_{(u,i)}=\sigma\!\bigl(\mathrm{MLP}(h_u\ \|\ h_i)\bigr),
\tag{5}
$$

式中：$\mathrm{MLP}(\cdot)$ 为多层感知机；$\sigma(\cdot)$ 为 Sigmoid 函数。给定训练边集合上的标签 $y\in\{0,1\}$，经验风险采用 BCE：

$$
\mathcal{L}_{\mathrm{tr}}=-\frac{1}{|\mathcal{B}|}\sum_{(u,i,t,y)\in\mathcal{B}}\Bigl[y\log p_{(u,i)}+(1-y)\log\bigl(1-p_{(u,i)}\bigr)\Bigr],
\tag{6}
$$

式中：$\mathcal{B}$ 为一个 mini-batch。该目标对应 ContraTGT 微调阶段的链路预测形式^[1]^。

---

## 1.5 对比预训练与可学习序列掩码增强

预训练在无额外标注条件下强化时空表示。沿用一致性—多样性准则^[1,3]^：给定增强算子 $\Psi(\cdot;\theta_\Psi)$ 与编码器输出表示 $h_x$、$h_{\hat{x}}$，理想增强应在 **预测一致** 与 **表示差异** 之间折中。

### 1.5.1 一致性与多样性度量

设链路预测为 $f(\cdot)$，指示函数为 $\mathbb{I}(\cdot)$，则一致性与多样性可写为^[1,3]^

$$
\mathcal{C}(x,\hat{x})=\mathbb{I}\!\bigl(f(h_x)=f(h_{\hat{x}})\bigr),
\tag{7}
$$

$$
\mathcal{D}(x,\hat{x})=d(h_x,h_{\hat{x}}),
\tag{8}
$$

式中：$d(\cdot,\cdot)$ 为距离函数，常取欧氏距离^[1]^。

### 1.5.2 可学习掩码增强

对长度为 $n$ 的原始序列 $s=(v_0,\ldots,v_{n-1})$ 及其特征序列 $s_x$，经 Transformer 得到表示序列 $S=(h_{v_0},\ldots,h_{v_{n-1}})$。可学习掩码模块生成增强特征序列^[1]^

$$
\hat{s}_x=\Psi_{\mathrm{lm}}(S;\theta_{\Psi})=\mathrm{SHARPEN}(\mathrm{ATTEN}(S,v);\rho)\odot s_x,
\tag{9}
$$

式中：$\mathrm{ATTEN}(S,v)$ 对各位置重要性打分；$\mathrm{SHARPEN}(\cdot;\rho)$ 在掩码率 $\rho$ 约束下将权重锐化为近似 $\{0,1\}$ 掩码且保持可反向传播^[1]^；$\odot$ 为逐元素乘法。

### 1.5.3 预训练辅助损失与对比损失

对真实链路 $(v,u)$，记增强后一端表示为 $\hat{v}$，链路概率分别为 $p_{(v,u)}$、$p_{(\hat{v},u)}$，则一致性损失与多样性损失可取^[1]^

$$
\mathcal{L}_{\Psi}^{c}=-\sum_{(v,u)\in\mathcal{L}_{\mathrm{pt}}}\Bigl[p_{(v,u)}\log p_{(\hat{v},u)}+\bigl(1-p_{(v,u)}\bigr)\log\bigl(1-p_{(\hat{v},u)}\bigr)\Bigr],
\tag{10}
$$

$$
\mathcal{L}_{\Psi}^{d}=-\sum_{(v,u)\in\mathcal{L}_{\mathrm{pt}}} d(h_v,h_{\hat{v}}),
\tag{11}
$$

$$
\mathcal{L}_{\mathrm{aug}}=\mathcal{L}_{\Psi}^{c}+\alpha\mathcal{L}_{\Psi}^{d}.
\tag{12}
$$

式中：$\mathcal{L}_{\mathrm{pt}}$ 为预训练链路集合；$\alpha$ 为权重系数^[1]^。

对比损失拉近原视图与增强视图、推远随机负样本 $u^-$^[1]^：

$$
\mathcal{L}_{\mathrm{con}}=-\sum_{v\in\mathcal{V}_{\mathrm{pt}},\,u^-\in\mathcal{V}_{\mathrm{pt}}\setminus\{v\}}\log\frac{\exp(h_v\cdot h_{\hat{v}})}{\exp(h_v\cdot h_{\hat{v}})+\exp(h_v\cdot h_{u^-})}.
\tag{13}
$$

式中：点积可替换为余弦相似度等实现变体。预训练中 $\mathcal{L}_{\mathrm{aug}}$ 与 $\mathcal{L}_{\mathrm{con}}$ 常 **交替优化** 不同子模块参数，以降低增强退化与表示坍塌^[1]^。**Top-$k$ 序列重加权** 等实现细节与官方代码一致，此处不赘述。

---

## 1.6 课程实例先验与偏置负采样（PANS）

同一课程模块（如 OULAD 中 `code_module`）下往往并行多个开课实例（presentation）。记各实例对应的 VLE 站点集合为 $\{\mathcal{R}^{(p)}\}_p$，构造实例间站点集合的 **Jaccard 相似度** 矩阵 $W\in[0,1]^{P\times P}$；$W_{pq}$ 越大表示两实例在站点层面的重叠越强，蕴含相似的课程结构先验。

微调或预训练阶段的随机负采样中，以一定比例将终点站点由全局均匀分布替换为 **与当前边所属 presentation 强相关的站点** 所诱导的分布（与 $W$ 及混合系数 `pres_mix_uniform` 一致），得到 **PANS**。直观上，负例更接近「同学段、结构相近却标签为负」的 **困难负样本**，有利于学习与课程组织相关的判别边界。该模块不改变式 (5)—(6) 的链路头形式，仅改变负例分布。

---

## 1.7 训练流程与复杂度说明

**输入**：边列表（含时间戳）、节点特征矩阵、可选 $W$。  
**输出**：微调后的 Graph Transformer 与链路预测头参数。

1. 按时间分位数划分训练 / 验证 / 测试集及新节点子集；  
2. **预训练**：在训练时间窗内采样双路序列 → 可学习掩码增强 → 计算式 (10)—(13) 及相关 Top-$k$ 辅助项 → 早停保存权重；  
3. **微调**：加载预训练参数，在式 (6) 下优化；若启用 PANS，在负采样步骤注入 $W$；  
4. **推断**：对测试边报告 AUC、AP、ACC 等。

记批大小为 $B$，隐维度为 $d$，空间 / 时间序列长度为 $l_s$、$l_t$，则单层 Transformer 主导的表示计算量级约为 $\mathcal{O}(B(l_s^2+l_t^2)d)$；序列长度为常数时随图规模近似线性扩展^[1]^。

---

**参考文献（References）**

（格式参照《浙江大学学报（工学版）》常用著录习惯；外文文献题名保留原文；标点按刊物最终稿统一校对。）

[1] Anonymous. Contrastive pre-training and learnable data augmentation of temporal graph networks[C]//Proceedings of the International Joint Conference on Artificial Intelligence. （投稿稿；录用后补：出版地: 出版者, 出版年: 起止页码.）全文见项目 `paper/ContraTGT_IJCAI.pdf`；代码 https://github.com/Jaff-hi/ContraTGT .

[2] BA J L, KIROS J R, HINTON G E. Layer normalization[J]. arXiv preprint arXiv:1607.06450, 2016.

[3] BO D, HU B, WANG X, et al. Regularizing graph neural networks via consistency-diversity graph augmentations[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(4): 3913-3921.

[4] KUZILEK J, HOLENKO DLABACOVÁ M, ZENDULKA J. Open university learning analytics dataset[J]. Scientific Data, 2017, 4: 170171.

[5] VASWANI A, SHAZEER N, PARMAR N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems. 2017, 30.

[6] XU D, RUAN C, KORPEOGLU E, et al. Inductive representation learning on temporal graphs[C]//International Conference on Learning Representations. 2020.

[7] ROSSI E, CHAMBERLAIN B, FRASCA F, et al. Temporal graph networks for deep learning on dynamic graphs[J]. arXiv preprint arXiv:2006.10637, 2020.

[8] KUMAR S, ZHANG X, LESKOVEC J. Predicting dynamic embedding trajectory in temporal interaction networks[C]//Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. New York: ACM, 2019: 1269-1278.

---

**说明**：公式编号 (1)—(13) 为本稿自成体系编号；其中式 (1)—(3)、(5)、(7)—(13) 的思想与符号与 ContraTGT 文献^[1]^ 对应；式 (6) 为 BCE 的标准写法，与文献中式 (2) 的链路预测训练目标一致。正式投稿时请根据编辑部模板将图表插入排版系统，并将文献 [1] 替换为正式录用信息。
