# TGN（Temporal Graph Networks）

## 论文

Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs*, ICML Workshop 2020.

## 官方代码来源

- 作者实现：**[twitter-research/tgn](https://github.com/twitter-research/tgn)**  
- 本仓库使用 **PyTorch Geometric** 中与上述对齐的模块：  
  [`torch_geometric.nn.models.tgn`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch-geometric-nn-models-tgn)（含 `TGNMemory`、`IdentityMessage`、`LastAggregator` 等）

## 本项目的接入方式

- `runner.py` 调用 PyG 的 `TGNMemory`，用 `Linear(2F → raw_msg_dim)` 将两端节点特征拼成 `raw_message`，与官方「特征驱动消息」思路一致。  
- **未**嵌入官方示例中基于子图的 `TransformerConv` 堆叠，以降低与 OULAD 自定义边表的耦合；为 **Memory + 链路预测 MLP** 的轻量变体。  
- 负采样、划分、`ml_*.csv` / `*.content` 均经 `baseline/common/data_adapter.py` 与 `main.py` 一致。

## 与官方示例相比的最小改动

- 数据：由 `utils.Dataset` + adapter 提供 `edge_idx` 与 mask，而非 TGB 的 `TemporalData`。  
- 评估：`TGNMemory` 在仅 `update_state` 时需处于 **train 模式**（PyG 行为），本仓库在 `_eval_split` 中对 `memory` 使用 `train()` + `no_grad()`。  
- 训练：在 **PyTorch 2.x** 上，若按官方顺序在 `backward` 前调用 `update_state`，且使用两条独立 `BCEWithLogitsLoss`，可能触发「二次反传」错误。本实现采用 **单次 BCE（拼接 pos/neg logits）**，并在 `backward` / `optimizer.step` / `memory.detach()` 之后用 **`torch.no_grad()`** 调用 `update_state`；`LinkPredictor` 输入为 **`[z_src, z_dst, raw_msg]`**，保证 **`msg_mlp` 仍能从链路损失收到梯度**。

## 运行

```bash
cd EduTGT/EduTGT
python baseline/tgn/runner.py --data_dir data/processed/data_AAA_2013J
```
