"""
时序图注意力（TGAT, Xu et al., ICLR 2020）核心：对时间排序邻居做多头注意力聚合。
实现参考论文与官方仓库
https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs
的邻居编码思路；因官方为 TensorFlow 且数据管线不同，此处用 PyTorch 按公式重写。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncode(nn.Module):
    """与 TGAT 一致的正弦/线性时间编码（简化版）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: [*, ] float
        x = delta_t.float().unsqueeze(-1)
        return torch.cos(self.w(x))


class TGATNeighborAttention(nn.Module):
    """单中心节点对多邻居的缩放点积注意力（含时间编码），对齐 TGAT 论文形式。"""

    def __init__(self, in_dim: int, out_dim: int, time_dim: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.dk = out_dim // n_heads
        self.proj_q = nn.Linear(in_dim, out_dim)
        self.proj_k = nn.Linear(in_dim + time_dim, out_dim)
        self.proj_v = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.time_enc = TimeEncode(time_dim)

    def forward(
        self,
        z_center: torch.Tensor,
        z_ngh: torch.Tensor,
        t_center: torch.Tensor,
        t_ngh: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_center: [B, d]
        z_ngh: [B, K, d]
        t_center: [B], t_ngh: [B, K] 绝对时间戳
        mask: [B, K] 1 有效 0 padding
        """
        B, K, _ = z_ngh.shape
        dt = (t_center.unsqueeze(-1) - t_ngh).clamp(min=0).float()
        te = self.time_enc(dt.reshape(-1)).view(B, K, -1)
        kh = torch.cat([z_ngh, te], dim=-1)

        q = self.proj_q(z_center).view(B, self.n_heads, self.dk).unsqueeze(2)
        k = self.proj_k(kh).view(B, K, self.n_heads, self.dk).transpose(1, 2)
        v = self.proj_v(z_ngh).view(B, K, self.n_heads, self.dk).transpose(1, 2)

        att = (q * k).sum(-1) / math.sqrt(self.dk)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e9)
        w = F.softmax(att, dim=-1)
        w = self.dropout(w)
        out = (w.unsqueeze(-1) * v).sum(2).reshape(B, -1)
        return self.out_proj(out)


class TGATLinkBaseline(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 128, time_dim: int = 32, heads: int = 2):
        super().__init__()
        self.feat_embed = nn.Linear(feat_dim, hidden)
        self.src_att = TGATNeighborAttention(hidden, hidden, time_dim, heads)
        self.dst_att = TGATNeighborAttention(hidden, hidden, time_dim, heads)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(
        self,
        feat_u: torch.Tensor,
        feat_v: torch.Tensor,
        src_ngh_feat: torch.Tensor,
        dst_ngh_feat: torch.Tensor,
        t: torch.Tensor,
        src_ngh_t: torch.Tensor,
        dst_ngh_t: torch.Tensor,
        src_mask: torch.Tensor,
        dst_mask: torch.Tensor,
    ) -> torch.Tensor:
        zu = self.feat_embed(feat_u)
        zv = self.feat_embed(feat_v)
        zs = self.src_att(zu, self.feat_embed(src_ngh_feat), t, src_ngh_t, src_mask)
        zd = self.dst_att(zv, self.feat_embed(dst_ngh_feat), t, dst_ngh_t, dst_mask)
        return self.head(torch.cat([zs, zd], dim=-1)).squeeze(-1)
