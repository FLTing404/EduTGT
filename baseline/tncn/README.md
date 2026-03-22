# 第三基线说明：TNCN → JODIE 风格

## 原计划：TNCN

**TNCN**（Temporal Neighborhood Convolutional Network 等）在公开实现上多与 **TGB（Temporal Graph Benchmark）** 数据格式与评估管道绑定（例如 **[GraphPKU/TNCN](https://github.com/GraphPKU/TNCN)**），与当前 OULAD 经 `utils.Dataset` 产出的 `ml_*.csv` + 分位数划分 **接入成本高**，且需大规模改写数据与负采样逻辑。

## 替代模型：JODIE

Kumar et al., *Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks*, KDD 2019.

- 作者参考实现：**[srijankr/jodie](https://github.com/srijankr/jodie)**（Python + PyTorch 友好，与动态链路预测任务一致）

## 本项目的接入方式

- 目录名保留为 **`tncn/`** 以符合仓库约定的三目录结构；**实现为精简 JODIE 风格**：可训练节点嵌入 + 双线性链路打分 + `GRUCell` 在 **无梯度** 下更新动态状态 `h`。  
- 训练损失当前主要反传到 **静态嵌入与线性层**；`evolve` 中的 RNN 更新为 **推理式刷新**（与完整 JODIE 的 BPTT 不完全等价），便于与当前 batch 训练对齐——实验对比时请注意该简化。

## 与官方 JODIE 的差异

- 未嵌入官方仓库的 `data/` 加载与 `train.py` 全流程，仅保留 **动态嵌入 + 交互更新 + 链路预测** 核心思想。  
- 数据与划分、负采样 **全部** 来自 `common/data_adapter.py`（与 `main.py` 一致）。

## 运行

```bash
cd EduTGT/EduTGT
python baseline/tncn/runner.py --data_dir data/processed/data_AAA_2013J
```
