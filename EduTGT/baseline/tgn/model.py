"""链路预测头：在 PyG 示例基础上拼接 raw_msg，使 msg_mlp 对损失可反传。"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Linear


class LinkPredictor(torch.nn.Module):
    def __init__(self, memory_dim: int, raw_dim: int):
        super().__init__()
        self.lin = Linear(memory_dim * 2 + raw_dim, 1)

    def forward(self, z_src: Tensor, z_dst: Tensor, raw: Tensor) -> Tensor:
        return self.lin(torch.cat([z_src, z_dst, raw], dim=-1))
