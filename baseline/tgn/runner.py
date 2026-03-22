#!/usr/bin/env python3
"""
TGN：使用 torch_geometric.nn.models.tgn 中的 TGNMemory（论文官方实现由 Twitter Research 开源，
PyG 收录并与作者对齐的内存模块 + 训练流程，见 PyG examples/tgn.py）。
本 runner 采用 **Memory + 链路 MLP**（不显式堆 TransformerConv 子图，以降低与自定义数据的耦合），
负采样与 main.py 一致（训练集 dst 上的 RandEdgeSampler）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_BASE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "baseline"))

from common.data_adapter import load_baseline_bundle, sanity_check_nodes  # noqa: E402
from common.metrics import link_prediction_metrics  # noqa: E402
from common.seed import set_seed  # noqa: E402
from common.trainer import EarlyStoppingAP, save_json  # noqa: E402

from tgn.model import LinkPredictor  # noqa: E402


def _require_pyg():
    try:
        from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TGNMemory
    except ImportError as e:
        raise SystemExit(
            "未安装 torch_geometric。请执行: pip install -r baseline/requirements-baseline.txt\n" + str(e)
        ) from e
    return TGNMemory, IdentityMessage, LastAggregator


def _chrono_indices(bundle, mask: torch.Tensor) -> np.ndarray:
    m = mask.numpy()
    p = bundle.perm_chrono
    return p[m[p]]


def _train_one_epoch(
    bundle,
    memory,
    link_pred,
    msg_mlp,
    feat: torch.Tensor,
    optimizer,
    criterion,
    batch_size: int,
    device: torch.device,
) -> float:
    memory.train()
    link_pred.train()
    msg_mlp.train()
    memory.reset_state()
    idxs = _chrono_indices(bundle, bundle.train_mask)
    edge_idx = bundle.edge_idx
    total_loss = 0.0
    n_steps = 0
    assoc = torch.empty(bundle.num_nodes, dtype=torch.long, device=device)

    for start in range(0, len(idxs), batch_size):
        sl = idxs[start : start + batch_size]
        if len(sl) == 0:
            break
        optimizer.zero_grad()
        src = edge_idx[sl, 0].to(device)
        dst = edge_idx[sl, 1].to(device)
        t = edge_idx[sl, 2].to(device)
        neg = torch.as_tensor(bundle.train_rand_sampler.sample(len(sl)), device=device, dtype=torch.long)

        x_pos = torch.cat([feat[src], feat[dst]], dim=-1)
        x_neg = torch.cat([feat[src], feat[neg]], dim=-1)
        raw_pos = msg_mlp(x_pos)
        raw_neg = msg_mlp(x_neg)

        n_id = torch.cat([src, dst, neg]).unique()
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, _ = memory(n_id)
        pos_o = link_pred(z[assoc[src]], z[assoc[dst]], raw_pos)
        neg_o = link_pred(z[assoc[src]], z[assoc[neg]], raw_neg)

        logits = torch.cat([pos_o, neg_o], dim=0)
        y = torch.cat([torch.ones_like(pos_o), torch.zeros_like(neg_o)], dim=0)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        memory.detach()
        # PyTorch 2.x 下「先 backward 再在 no_grad 里 update_state」避免与 z 的图冲突；
        # 记忆仍用 raw_pos（与正边一致）更新。
        with torch.no_grad():
            memory.update_state(src, dst, t, raw_pos)
        total_loss += float(loss.detach()) * len(sl)
        n_steps += len(sl)
    return total_loss / max(n_steps, 1)


@torch.no_grad()
def _eval_split(
    bundle,
    memory,
    link_pred,
    msg_mlp,
    feat: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    batch_size: int,
    train_sampler,
) -> dict:
    # TGNMemory 需在 train 模式下 update_state 才会写回 GRU 记忆（eval 时仅写 msg_store）
    memory.train()
    link_pred.eval()
    msg_mlp.eval()
    memory.reset_state()
    idxs = _chrono_indices(bundle, bundle.train_mask)
    edge_idx = bundle.edge_idx
    assoc = torch.empty(bundle.num_nodes, dtype=torch.long, device=device)

    # 用训练集时间线填充 memory（与 PyG test 前需见过历史一致）
    for start in range(0, len(idxs), batch_size):
        sl = idxs[start : start + batch_size]
        src = edge_idx[sl, 0].to(device)
        dst = edge_idx[sl, 1].to(device)
        t = edge_idx[sl, 2].to(device)
        x = torch.cat([feat[src], feat[dst]], dim=-1)
        raw_msg = msg_mlp(x)
        memory.update_state(src, dst, t, raw_msg)

    pos_list, neg_list = [], []
    eidx = _chrono_indices(bundle, mask)
    for start in range(0, len(eidx), batch_size):
        sl = eidx[start : start + batch_size]
        src = edge_idx[sl, 0].to(device)
        dst = edge_idx[sl, 1].to(device)
        t = edge_idx[sl, 2].to(device)
        neg = torch.as_tensor(train_sampler.sample(len(sl)), device=device, dtype=torch.long)

        x = torch.cat([feat[src], feat[dst]], dim=-1)
        raw_msg = msg_mlp(x)

        n_id = torch.cat([src, dst, neg]).unique()
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, _ = memory(n_id)
        raw_neg = msg_mlp(torch.cat([feat[src], feat[neg]], dim=-1))
        pos_o = link_pred(z[assoc[src]], z[assoc[dst]], raw_msg)
        neg_o = link_pred(z[assoc[src]], z[assoc[neg]], raw_neg)
        pos_list.append(pos_o.cpu().numpy())
        neg_list.append(neg_o.cpu().numpy())
        memory.update_state(src, dst, t, raw_msg)

    pos_logits = np.concatenate(pos_list) if pos_list else np.array([])
    neg_logits = np.concatenate(neg_list) if neg_list else np.array([])
    return link_prediction_metrics(pos_logits, neg_logits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--memory_dim", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=64)
    p.add_argument("--raw_msg_dim", type=int, default=32)
    args = p.parse_args()

    TGNMemory, IdentityMessage, LastAggregator = _require_pyg()

    set_seed(args.seed)
    bundle = load_baseline_bundle(args.data_dir)
    sanity_check_nodes(bundle.edge_idx, bundle.num_nodes)
    device = torch.device(
        "cuda:0" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    out_dir = args.out_dir or os.path.join(_ROOT, "baseline", "results", bundle.data_name, "tgn")
    os.makedirs(out_dir, exist_ok=True)

    feat = torch.from_numpy(bundle.features).float().to(device)
    msg_mlp = nn.Linear(bundle.feat_dim * 2, args.raw_msg_dim).to(device)

    memory = TGNMemory(
        bundle.num_nodes,
        args.raw_msg_dim,
        args.memory_dim,
        args.time_dim,
        message_module=IdentityMessage(args.raw_msg_dim, args.memory_dim, args.time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    link_pred = LinkPredictor(args.memory_dim, args.raw_msg_dim).to(device)
    optimizer = torch.optim.Adam(
        list(memory.parameters()) + list(link_pred.parameters()) + list(msg_mlp.parameters()),
        lr=args.lr,
    )
    criterion = BCEWithLogitsLoss()

    es = EarlyStoppingAP(patience=8)
    t0 = time.time()
    best_state = None
    best_ep = 0

    val_sampler = bundle.train_rand_sampler
    test_sampler = bundle.train_rand_sampler

    for ep in range(1, args.epochs + 1):
        tr_loss = _train_one_epoch(
            bundle, memory, link_pred, msg_mlp, feat, optimizer, criterion, args.batch_size, device
        )
        set_seed(args.seed + ep)
        val_m = _eval_split(
            bundle, memory, link_pred, msg_mlp, feat, bundle.val_mask, device, args.batch_size, val_sampler
        )
        improved = es.step(ep, val_m["ap"])
        if improved:
            best_ep = ep
            best_state = {
                "memory": memory.state_dict(),
                "link_pred": link_pred.state_dict(),
                "msg_mlp": msg_mlp.state_dict(),
            }
        print(
            f"[TGN] epoch {ep} train_loss={tr_loss:.4f} val_auc={val_m['auc']:.4f} val_ap={val_m['ap']:.4f}"
        )
        if es.should_stop:
            break

    if best_state:
        memory.load_state_dict(best_state["memory"])
        link_pred.load_state_dict(best_state["link_pred"])
        msg_mlp.load_state_dict(best_state["msg_mlp"])

    set_seed(args.seed)
    val_m = _eval_split(
        bundle, memory, link_pred, msg_mlp, feat, bundle.val_mask, device, args.batch_size, val_sampler
    )
    test_m = _eval_split(
        bundle, memory, link_pred, msg_mlp, feat, bundle.test_mask, device, args.batch_size, test_sampler
    )
    nn_test_m = _eval_split(
        bundle, memory, link_pred, msg_mlp, feat, bundle.nn_test_mask, device, args.batch_size, test_sampler
    )

    elapsed = time.time() - t0
    rec = {
        "model": "tgn",
        "data_name": bundle.data_name,
        "val_auc": val_m["auc"],
        "val_ap": val_m["ap"],
        "test_auc": test_m["auc"],
        "test_ap": test_m["ap"],
        "test_acc": test_m["acc"],
        "nn_test_auc": nn_test_m["auc"],
        "nn_test_ap": nn_test_m["ap"],
        "nn_test_acc": nn_test_m["acc"],
        "train_time_sec": elapsed,
        "best_epoch": best_ep,
        "hyperparams": vars(args),
        "stats": bundle.stats,
    }
    print(
        f"test_auc={test_m['auc']:.6f} test_ap={test_m['ap']:.6f} test_acc={test_m['acc']:.6f} | "
        f"nn_test_auc={nn_test_m['auc']:.6f} nn_test_ap={nn_test_m['ap']:.6f} nn_test_acc={nn_test_m['acc']:.6f}"
    )
    save_json(os.path.join(out_dir, "result.json"), rec)
    torch.save(best_state or {}, os.path.join(out_dir, "checkpoint.pt"))
    print(json.dumps(rec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
