# TGAT（Temporal Graph Attention）

## 论文

Xu et al., *Inductive Representation Learning on Temporal Graphs*, ICLR 2020.

## 官方代码来源

- 论文作者仓库（TensorFlow）：**[StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs)**

## 本项目的接入方式

官方实现为 **TensorFlow**，无法在不引入整套 TF 图与数据管道的前提下「原样」嵌入本 PyTorch 项目。此处采用：

- **与论文一致的机制**：时间感知的 **多头注意力** 聚合历史邻居 + 中心节点特征，再经 MLP 做链路对数几率；  
- **与主项目一致的结构化邻居**：通过现有 **`sampling.get_neighbor_list`**（与 `main.py` 相同）构造时序邻居；负样本对 **假 destination** 单独取邻居。

## 与官方代码相比的差异

- 框架为 **PyTorch**，非官方 TF 代码的逐行移植。  
- 未复现官方仓库中的完整训练脚本与数据集适配，仅复现 **模型结构思想** 与 **邻居采样接口**（来自本仓库 `sampling.py`）。

## 运行

```bash
cd EduTGT/EduTGT
python baseline/tgat/runner.py --data_dir data/processed/data_AAA_2013J
```
