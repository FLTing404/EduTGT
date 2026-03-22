#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_BASE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "baseline"))

from common.data_adapter import load_baseline_bundle, sanity_check_nodes  # noqa: E402
from common.metrics import link_prediction_metrics  # noqa: E402
from common.seed import set_seed  # noqa: E402
from common.trainer import EarlyStoppingAP, save_json  # noqa: E402
from sampling import get_adj_list, get_neighbor_list, init_offset  # noqa: E402

from tgat.model import TGATLinkBaseline  # noqa: E402


def _pack_neighbors(
    ngh_node: np.ndarray,
    ngh_ts_delta: np.ndarray,
    ngh_mask: np.ndarray,
    t_center: torch.Tensor,
    features: np.ndarray,
    device: torch.device,
) -> tuple:
    """ngh_ts_delta: get_neighbor_list 返回的「当前边时刻 − 邻居时刻」。"""
    B, K = ngh_node.shape
    nf = np.empty((B, K, features.shape[1]), dtype=np.float32)
    for i in range(B):
        for j in range(K):
            nf[i, j] = features[ngh_node[i, j]]
    t_abs = t_center.unsqueeze(-1) - torch.tensor(ngh_ts_delta, device=device, dtype=torch.long).clamp(min=0)
    mask_t = torch.tensor(ngh_mask, device=device, dtype=torch.float32)
    return (
        torch.from_numpy(nf).to(device),
        t_abs,
        mask_t,
    )


def _run_epoch_train(
    model,
    bundle,
    adj,
    node_l,
    ts_l,
    idx_l,
    offset_l,
    features_np,
    feat_t: torch.Tensor,
    optimizer,
    criterion,
    batch_size: int,
    ctx: int,
    device: torch.device,
) -> float:
    model.train()
    idxs = bundle.perm_chrono[bundle.train_mask.numpy()[bundle.perm_chrono]]
    edge_idx = bundle.edge_idx
    total_loss = 0.0
    n = 0
    for start in range(0, len(idxs), batch_size):
        sl = idxs[start : start + batch_size]
        if len(sl) == 0:
            break
        u = edge_idx[sl, 0]
        v = edge_idx[sl, 1]
        t = edge_idx[sl, 2]
        neg = torch.as_tensor(bundle.train_rand_sampler.sample(len(sl)), dtype=torch.long)

        su, tu_d, _, mu = get_neighbor_list(node_l, ts_l, idx_l, offset_l, u.numpy(), t.numpy(), num_sample=ctx)
        sv, tv_d, _, mv = get_neighbor_list(node_l, ts_l, idx_l, offset_l, v.numpy(), t.numpy(), num_sample=ctx)

        t_dev = t.to(device)
        fn = features_np
        sf, st, sm = _pack_neighbors(su, tu_d, mu, t_dev, fn, device)
        df, dt, dm = _pack_neighbors(sv, tv_d, mv, t_dev, fn, device)

        fu = feat_t[u.to(device)]
        fv = feat_t[v.to(device)]
        pos = model(fu, fv, sf, df, t_dev, st, dt, sm, dm)
        fn_ = feat_t[neg.to(device)]
        sn, tn_d, _, mn = get_neighbor_list(
            node_l, ts_l, idx_l, offset_l, neg.numpy(), t.numpy(), num_sample=ctx
        )
        dnf, dnt, dnm = _pack_neighbors(sn, tn_d, mn, t_dev, fn, device)
        neg_l = model(fu, fn_, sf, dnf, t_dev, st, dnt, sm, dnm)

        loss = criterion(pos, torch.ones_like(pos)) + criterion(neg_l, torch.zeros_like(neg_l))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(sl)
        n += len(sl)
    return total_loss / max(n, 1)


@torch.no_grad()
def _eval_mask(
    model,
    bundle,
    node_l,
    ts_l,
    idx_l,
    offset_l,
    features_np,
    feat_t: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    ctx: int,
    device: torch.device,
    sampler,
) -> dict:
    model.eval()
    edge_idx = bundle.edge_idx
    pos_l, neg_l = [], []
    idxs = bundle.perm_chrono[mask.numpy()[bundle.perm_chrono]]
    for start in range(0, len(idxs), batch_size):
        sl = idxs[start : start + batch_size]
        u = edge_idx[sl, 0]
        v = edge_idx[sl, 1]
        t = edge_idx[sl, 2]
        neg = torch.as_tensor(sampler.sample(len(sl)), dtype=torch.long)
        su, tu_d, _, mu = get_neighbor_list(node_l, ts_l, idx_l, offset_l, u.numpy(), t.numpy(), num_sample=ctx)
        sv, tv_d, _, mv = get_neighbor_list(node_l, ts_l, idx_l, offset_l, v.numpy(), t.numpy(), num_sample=ctx)
        t_dev = t.to(device)
        sf, st, sm = _pack_neighbors(su, tu_d, mu, t_dev, features_np, device)
        df, dt, dm = _pack_neighbors(sv, tv_d, mv, t_dev, features_np, device)
        fu = feat_t[u.to(device)]
        fv = feat_t[v.to(device)]
        pos = model(fu, fv, sf, df, t_dev, st, dt, sm, dm)
        fn_ = feat_t[neg.to(device)]
        sn, tn_d, _, mn = get_neighbor_list(
            node_l, ts_l, idx_l, offset_l, neg.numpy(), t.numpy(), num_sample=ctx
        )
        dnf, dnt, dnm = _pack_neighbors(sn, tn_d, mn, t_dev, features_np, device)
        neg_o = model(fu, fn_, sf, dnf, t_dev, st, dnt, sm, dnm)
        pos_l.append(pos.cpu().numpy())
        neg_l.append(neg_o.cpu().numpy())
    if not pos_l:
        return {"auc": float("nan"), "ap": float("nan"), "acc": float("nan")}
    return link_prediction_metrics(np.concatenate(pos_l), np.concatenate(neg_l))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--ctx_sample", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)
    bundle = load_baseline_bundle(args.data_dir)
    sanity_check_nodes(bundle.edge_idx, bundle.num_nodes)

    edges_dict = {"idx": bundle.edge_idx, "labels": bundle.labels}
    adj = get_adj_list(edges_dict)
    node_l, ts_l, idx_l, offset_l = init_offset(adj)

    device = torch.device(
        "cuda:0" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    feat_t = torch.from_numpy(bundle.features).float().to(device)
    model = TGATLinkBaseline(bundle.feat_dim, hidden=128, time_dim=32, heads=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()

    out_dir = args.out_dir or os.path.join(_ROOT, "baseline", "results", bundle.data_name, "tgat")
    os.makedirs(out_dir, exist_ok=True)
    es = EarlyStoppingAP(patience=8)
    t0 = time.time()
    best_sd = None
    best_ep = 0

    for ep in range(1, args.epochs + 1):
        set_seed(args.seed + ep)
        tr = _run_epoch_train(
            model,
            bundle,
            adj,
            node_l,
            ts_l,
            idx_l,
            offset_l,
            bundle.features,
            feat_t,
            opt,
            crit,
            args.batch_size,
            args.ctx_sample,
            device,
        )
        set_seed(args.seed)
        val_m = _eval_mask(
            model,
            bundle,
            node_l,
            ts_l,
            idx_l,
            offset_l,
            bundle.features,
            feat_t,
            bundle.val_mask,
            args.batch_size,
            args.ctx_sample,
            device,
            bundle.train_rand_sampler,
        )
        if es.step(ep, val_m["ap"]):
            best_ep = ep
            best_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"[TGAT] epoch {ep} train_loss={tr:.4f} val_auc={val_m['auc']:.4f} val_ap={val_m['ap']:.4f}")
        if es.should_stop:
            break

    if best_sd:
        model.load_state_dict(best_sd)
    set_seed(args.seed)
    val_m = _eval_mask(
        model,
        bundle,
        node_l,
        ts_l,
        idx_l,
        offset_l,
        bundle.features,
        feat_t,
        bundle.val_mask,
        args.batch_size,
        args.ctx_sample,
        device,
        bundle.train_rand_sampler,
    )
    test_m = _eval_mask(
        model,
        bundle,
        node_l,
        ts_l,
        idx_l,
        offset_l,
        bundle.features,
        feat_t,
        bundle.test_mask,
        args.batch_size,
        args.ctx_sample,
        device,
        bundle.train_rand_sampler,
    )
    nn_test_m = _eval_mask(
        model,
        bundle,
        node_l,
        ts_l,
        idx_l,
        offset_l,
        bundle.features,
        feat_t,
        bundle.nn_test_mask,
        args.batch_size,
        args.ctx_sample,
        device,
        bundle.train_rand_sampler,
    )
    elapsed = time.time() - t0
    rec = {
        "model": "tgat",
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
    torch.save(best_sd or model.state_dict(), os.path.join(out_dir, "checkpoint.pt"))
    print(json.dumps(rec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
