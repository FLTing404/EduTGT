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

from tncn.model import JODIEBaseline  # noqa: E402


def _train_rows(bundle) -> np.ndarray:
    return bundle.perm_chrono[bundle.train_mask.numpy()[bundle.perm_chrono]]


def _mask_rows(bundle, m: torch.Tensor) -> np.ndarray:
    return bundle.perm_chrono[m.numpy()[bundle.perm_chrono]]


def train_epoch(model, bundle, opt, crit, batch_size, device):
    model.train()
    edge_idx = bundle.edge_idx
    rows = _train_rows(bundle)
    tot = 0.0
    n = 0
    h = torch.zeros(bundle.num_nodes, model.d, device=device)
    h.copy_(model.emb.weight.detach())

    for start in range(0, len(rows), batch_size):
        sl = rows[start : start + batch_size]
        u = edge_idx[sl, 0].to(device)
        v = edge_idx[sl, 1].to(device)
        neg = torch.as_tensor(bundle.train_rand_sampler.sample(len(sl)), device=device, dtype=torch.long)

        pos = model.logits(u, v)
        neg_l = model.logits(u, neg)
        loss = crit(pos, torch.ones_like(pos)) + crit(neg_l, torch.zeros_like(neg_l))
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            model.evolve(h, u, v)
        tot += float(loss.detach()) * len(sl)
        n += len(sl)
    return tot / max(n, 1)


@torch.no_grad()
def eval_mask(model, bundle, mask, batch_size, device, sampler):
    model.eval()
    edge_idx = bundle.edge_idx
    h = torch.zeros(bundle.num_nodes, model.d, device=device)
    h.copy_(model.emb.weight)

    for sl in np.array_split(_train_rows(bundle), max(1, len(_train_rows(bundle)) // batch_size)):
        if len(sl) == 0:
            continue
        u = edge_idx[sl, 0].to(device)
        v = edge_idx[sl, 1].to(device)
        model.evolve(h, u, v)

    pos_l, neg_l = [], []
    for sl in np.array_split(_mask_rows(bundle, mask), max(1, int(mask.sum()) // batch_size)):
        if len(sl) == 0:
            continue
        u = edge_idx[sl, 0].to(device)
        v = edge_idx[sl, 1].to(device)
        neg = torch.as_tensor(sampler.sample(len(sl)), device=device, dtype=torch.long)
        pos_l.append(model.logits_state(h, u, v).cpu().numpy())
        neg_l.append(model.logits_state(h, u, neg).cpu().numpy())
        model.evolve(h, u, v)
    if not pos_l:
        return {"auc": float("nan"), "ap": float("nan"), "acc": float("nan")}
    return link_prediction_metrics(np.concatenate(pos_l), np.concatenate(neg_l))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--dim", type=int, default=128)
    args = p.parse_args()

    set_seed(args.seed)
    bundle = load_baseline_bundle(args.data_dir)
    sanity_check_nodes(bundle.edge_idx, bundle.num_nodes)
    device = torch.device(
        "cuda:0" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    model = JODIEBaseline(bundle.num_nodes, args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()

    out_dir = args.out_dir or os.path.join(_ROOT, "baseline", "results", bundle.data_name, "tncn")
    os.makedirs(out_dir, exist_ok=True)
    es = EarlyStoppingAP(patience=8)
    t0 = time.time()
    best_sd = None
    best_ep = 0

    for ep in range(1, args.epochs + 1):
        set_seed(args.seed + ep)
        tr = train_epoch(model, bundle, opt, crit, args.batch_size, device)
        set_seed(args.seed)
        val_m = eval_mask(model, bundle, bundle.val_mask, args.batch_size, device, bundle.train_rand_sampler)
        if es.step(ep, val_m["ap"]):
            best_ep = ep
            best_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"[JODIE] epoch {ep} train_loss={tr:.4f} val_auc={val_m['auc']:.4f} val_ap={val_m['ap']:.4f}")
        if es.should_stop:
            break

    if best_sd:
        model.load_state_dict(best_sd)
    set_seed(args.seed)
    val_m = eval_mask(model, bundle, bundle.val_mask, args.batch_size, device, bundle.train_rand_sampler)
    set_seed(args.seed)
    test_m = eval_mask(model, bundle, bundle.test_mask, args.batch_size, device, bundle.train_rand_sampler)
    set_seed(args.seed)
    nn_test_m = eval_mask(model, bundle, bundle.nn_test_mask, args.batch_size, device, bundle.train_rand_sampler)
    elapsed = time.time() - t0
    rec = {
        "model": "jodie",
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
