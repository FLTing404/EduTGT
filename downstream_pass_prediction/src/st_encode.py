"""仅构造源端（学生）时空张量并调用 SpatialTemporal.getEmbed。"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
import torch

from repo_path import ensure_repo_on_path

ensure_repo_on_path()

from sampling import get_neighbor_list, get_unique_node_sequence  # noqa: E402

if TYPE_CHECKING:
    from st_graph import GraphBundle


def batch_data_from_idx(idx_mat: torch.Tensor) -> Dict[str, Any]:
    """idx_mat (B,4): u,v,ts,idx"""
    return {
        "idx": idx_mat,
        "labels": torch.ones(idx_mat.shape[0], dtype=torch.long),
    }


def build_src_tensors(
    bundle: "GraphBundle",
    batch_data: Dict[str, Any],
    ctx_sample: int,
    tmp_sample: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """与 main.py 源端一致：context_feature, temporal_feature, spa_mask, tmp_mask。"""
    features = bundle.features
    indim = bundle.indim
    edges = bundle.edge
    node_l = bundle.node_l
    ts_l = bundle.ts_l
    idx_l = bundle.idx_l
    offset_l = bundle.offset_l
    interaction_list = bundle.interaction_list
    time_encode = bundle.time_encode
    time_information = bundle.time_information

    node_sum = len(batch_data["idx"])
    batch_ngh_node, batch_ngh_ts, _batch_ngh_idx, batch_ngh_mask = get_neighbor_list(
        node_l,
        ts_l,
        idx_l,
        offset_l,
        batch_data["idx"][:, 0].cpu().numpy(),
        batch_data["idx"][:, 2].cpu().numpy(),
        num_sample=ctx_sample,
    )
    con_seq_fea = np.empty((node_sum, ctx_sample, indim))
    for ix, neigh in enumerate(batch_ngh_node):
        for idj, j in enumerate(neigh):
            con_seq_fea[ix, idj, :] = features[j]
    con_seq_fea = torch.FloatTensor(np.asarray(con_seq_fea))
    ts = batch_data["idx"][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ngh_ts)
    con_seq_fea = time_information(con_seq_fea, ts)
    context_feature = time_encode(con_seq_fea, torch.tensor(batch_ngh_ts)).to(device)
    batch_ngh_mask_t = torch.LongTensor(np.asarray(batch_ngh_mask)).to(device)

    batch_node_seq, batch_node_seq_mask, batch_ts = get_unique_node_sequence(
        batch_data, edges, tmp_sample, interaction_list, flag=True
    )
    temp_seq_fea = np.empty((node_sum, tmp_sample, indim))
    for ix, seq in enumerate(batch_node_seq):
        for idj, j in enumerate(seq):
            temp_seq_fea[ix, idj, :] = features[j]
    temp_seq_fea = torch.FloatTensor(np.asarray(temp_seq_fea))
    ts = batch_data["idx"][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ts)
    temp_seq_fea = time_information(temp_seq_fea, ts)
    temporal_feature = time_encode(temp_seq_fea, torch.tensor(batch_ts)).to(device)
    batch_node_seq_mask_t = torch.LongTensor(np.asarray(batch_node_seq_mask)).to(device)

    return context_feature, temporal_feature, batch_ngh_mask_t, batch_node_seq_mask_t


def get_src_embed(
    st_model: torch.nn.Module,
    bundle: "GraphBundle",
    idx_mat: torch.Tensor,
    ctx_sample: int,
    tmp_sample: int,
    device: torch.device,
) -> torch.Tensor:
    """idx_mat (B,4) on CPU；返回 (B, embed_dim)。"""
    bd = batch_data_from_idx(idx_mat)
    ctx, tmp, m1, m2 = build_src_tensors(bundle, bd, ctx_sample, tmp_sample, device)
    return st_model.getEmbed(ctx, tmp, m1, m2)
