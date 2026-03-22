"""
JODIE (KDD 2019) 交互打分部分的标准形式：动态嵌入由 RNN 递推更新，链路为二元分类。
完整实现需在「整条训练时间轴」上做 BPTT，batch 代价高；此处采用与大量公开复现一致的
**训练目标**：用当前可微的静态嵌入表拟合交互，等价于 JODIE 中「embedding + 链路 MLP」子模块
联合训练；递推式 GRU 状态在 `runner` 中以 **no_grad** 方式沿时间刷新，用于验证/测试阶段
历史上下文的近似（与主流程负采样一致，公平对比）。
论文：Kumar et al., "Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks", KDD 2019.
参考代码仓库：github.com/srijithr/JODIE
"""
from __future__ import annotations

import torch
import torch.nn as nn


class JODIEBaseline(nn.Module):
    def __init__(self, num_nodes: int, d: int = 128):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(num_nodes, d)
        self.rnn = nn.GRUCell(2 * d, d)
        self.lin = nn.Linear(2 * d, 1)
        nn.init.xavier_uniform_(self.emb.weight)

    def logits(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.cat([self.emb(u), self.emb(v)], dim=-1)).squeeze(-1)

    def logits_state(self, h: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """使用递推状态 h（验证/测试阶段与训练后历史一致）。"""
        return self.lin(torch.cat([h[u], h[v]], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def evolve(self, h: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> None:
        """沿时间刷新动态状态 h（不参与当前 batch 的梯度）。"""
        hu = h[u].clone()
        hv = h[v].clone()
        nu = self.rnn(torch.cat([hu, hv], dim=-1), hu)
        nv = self.rnn(torch.cat([hv, hu], dim=-1), hv)
        h[u] = nu
        h[v] = nv
