#!/usr/bin/env python3
"""
E1：SpatialTemporal 随机初始化 + 线性分类头；不读预训练权重，ctx/tmp 由配置（与 main 量级一致）。
E2：加载 outputs/pretrain/<data_name>.pth；ctx/tmp 固定为 pretrain.py 内 30/21，与权重匹配。

只读 data/processed；E2 另只读预训练权重。标签与 split 在 downstream_pass_prediction/data/；
checkpoint 写在 downstream_pass_prediction/outputs/。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG / "src") not in sys.path:
    sys.path.insert(0, str(_PKG / "src"))

from repo_path import ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()

import paths  # noqa: E402
from model import SpatialTemporal  # noqa: E402

from metrics import binary_metrics, save_roc_curve_png  # noqa: E402
from paths_downstream import DOWNSTREAM_ROOT, REPO_ROOT, module_data_dir  # noqa: E402
from st_encode import get_src_embed  # noqa: E402
from io_data import load_stats  # noqa: E402
from st_graph import build_anchor_lists, load_graph_bundle, pack_anchors  # noqa: E402

# 仅 E2：与仓库根目录 pretrain.py 写死的 ctx_sample / tmp_sample 一致（勿改单边）
PRETRAIN_CTX_SAMPLE = 30
PRETRAIN_TMP_SAMPLE = 21


def _load_yaml(p: Path) -> dict:
    if yaml is None or not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_students(p: Path) -> set:
    s = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                s.add(int(line))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", type=str, required=True, help="如 data_AAA")
    ap.add_argument(
        "--init",
        type=str,
        choices=("scratch", "pretrain"),
        required=True,
        help="scratch=E1 随机初始化；pretrain=E2 加载预训练权重",
    )
    ap.add_argument("--config", type=str, default=str(_PKG / "configs" / "default.yaml"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument(
        "--freeze-epochs",
        type=int,
        default=None,
        help=(
            "仅 E2：开头 N 轮冻结编码器、只训练分类头（热身），之后再解冻全量微调。"
            "默认 E2=5，E1=0"
        ),
    )
    ap.add_argument(
        "--head-lr-factor",
        type=float,
        default=None,
        help=(
            "分类头学习率 = lr × factor；E2 默认 10.0，让头部更快适应预训练特征空间；"
            "E1 默认 1.0"
        ),
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=10,
        help="验证集 AUPRC 连续无提升的容忍轮数，达到则早停",
    )
    ap.add_argument("--t-star-relative-day", type=int, default=None)
    ap.add_argument(
        "--acc-threshold",
        type=float,
        default=0.5,
        help="将预测概率 >= 该阈值视为正类（通过）以计算 ACC",
    )
    ap.add_argument(
        "--no-roc-plot",
        action="store_true",
        help="不保存 ROC 曲线 PNG（仍输出 test_auroc）",
    )
    ap.add_argument(
        "--dump-run-json",
        type=str,
        default=None,
        help="将本次运行的指标等写入该 JSON 文件（便于批处理汇总）",
    )
    args = ap.parse_args()
    wall_t0 = time.perf_counter()

    cfg = _load_yaml(Path(args.config))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 60))
    st_cfg = cfg.get("st") or {}
    if args.init == "pretrain":
        ctx_sample = PRETRAIN_CTX_SAMPLE
        tmp_sample = PRETRAIN_TMP_SAMPLE
    else:
        ctx_sample = int(st_cfg.get("ctx_sample", cfg.get("ctx_sample", 40)))
        tmp_sample = int(st_cfg.get("tmp_sample", cfg.get("tmp_sample", 31)))
    dropout = float(st_cfg.get("drop_out", st_cfg.get("dropout", cfg.get("drop_out", 0.2))))
    nheads = int(st_cfg.get("n_heads", 4))
    n_layers = int(st_cfg.get("N", 2))
    out_dim = int(st_cfg.get("out_dim", 128))
    batch_size = int(args.batch_size or st_cfg.get("batch_size", 32))
    lr = float(args.lr if args.lr is not None else st_cfg.get("lr", 1e-4))
    # 两阶段微调参数：E2 默认先冻结 5 轮热身头，再差分 LR 解冻；E1 不冻结
    if args.init == "pretrain":
        freeze_epochs = args.freeze_epochs if args.freeze_epochs is not None else 5
        head_lr_factor = args.head_lr_factor if args.head_lr_factor is not None else 10.0
    else:
        freeze_epochs = args.freeze_epochs if args.freeze_epochs is not None else 0
        head_lr_factor = args.head_lr_factor if args.head_lr_factor is not None else 1.0
    t_star = args.t_star_relative_day
    if t_star is None and cfg.get("t_star_relative_day") is not None:
        t_star = cfg.get("t_star_relative_day")

    torch.manual_seed(seed)
    np.random.seed(seed)

    mod = args.module.strip()
    ddir = module_data_dir(mod)
    man_path = ddir / "manifest.json"
    if not man_path.is_file():
        raise SystemExit(f"请先运行 build_pass_labels.py：缺少 {man_path}")
    splits_dir = ddir / "splits"
    for name in ("train_students.txt", "val_students.txt", "test_students.txt"):
        if not (splits_dir / name).is_file():
            raise SystemExit(f"请先运行 build_student_splits.py：缺少 {splits_dir / name}")

    with open(man_path, encoding="utf-8") as f:
        manifest = json.load(f)

    def _p(rel: str) -> Path:
        return REPO_ROOT.joinpath(*rel.replace("\\", "/").split("/"))

    proc = _p(manifest["processed_dir"])
    ml_path = _p(manifest["ml_csv"])
    content_path = _p(manifest["content_csv"])
    labels_path = _p(manifest["labels_csv"])

    stats = load_stats(proc)
    presentations = list(stats["code_presentations_merged"])
    data_name = manifest.get("module") or mod

    bundle = load_graph_bundle(str(ml_path), str(content_path))
    indim = bundle.indim
    labels_df = pd.read_csv(labels_path)
    ml_df = pd.read_csv(ml_path)

    anchors, stu_ids = build_anchor_lists(
        str(ml_path),
        labels_df,
        presentations,
        t_star_relative_day=t_star,
        block=int(manifest.get("ts_pres_block", 1000)),
    )
    if not anchors:
        raise SystemExit("无有效锚点边（检查 t* 或标签与 ml）")

    idx_all, y_all = pack_anchors(bundle.edge["idx"], ml_df, anchors)
    stu_np = np.array(stu_ids, dtype=np.int64)

    train_s = _read_students(splits_dir / "train_students.txt")
    val_s = _read_students(splits_dir / "val_students.txt")
    test_s = _read_students(splits_dir / "test_students.txt")

    def _mask(sids: set) -> np.ndarray:
        return np.array([s in sids for s in stu_np], dtype=bool)

    tr_m, va_m, te_m = _mask(train_s), _mask(val_s), _mask(test_s)
    if not tr_m.any() or not va_m.any() or not te_m.any():
        raise SystemExit("train/val/test 锚点为空")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    st_model = SpatialTemporal(
        in_dim=indim,
        out_dim=out_dim,
        n_heads=nheads,
        dropout=dropout,
        N=n_layers,
    ).to(device)
    if args.init == "pretrain":
        ckpt = paths.pretrain_ckpt_path(data_name)
        if not ckpt.is_file():
            raise SystemExit(f"E2 需要预训练权重文件：{ckpt}")
        st_model.load_state_dict(torch.load(str(ckpt), map_location=device), strict=True)
    head = nn.Linear(indim, 1).to(device)

    pos = float(y_all[tr_m].sum().item())
    neg = float((y_all[tr_m] == 0).sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    wd = float(st_cfg.get("weight_decay", 1e-5))

    def _build_opt(frozen: bool) -> torch.optim.Optimizer:
        """frozen=True：只优化头部（编码器梯度被 requires_grad 关掉）。
        frozen=False：差分 LR——编码器用 lr，头部用 lr × head_lr_factor。"""
        if frozen:
            return torch.optim.Adam(head.parameters(), lr=lr * head_lr_factor, weight_decay=wd)
        if head_lr_factor == 1.0:
            return torch.optim.Adam(
                list(st_model.parameters()) + list(head.parameters()),
                lr=lr,
                weight_decay=wd,
            )
        return torch.optim.Adam(
            [
                {"params": list(st_model.parameters()), "lr": lr, "weight_decay": wd},
                {"params": list(head.parameters()), "lr": lr * head_lr_factor, "weight_decay": wd},
            ]
        )

    # Phase-1 setup：若 freeze_epochs > 0 先冻结编码器
    currently_frozen = freeze_epochs > 0
    if currently_frozen:
        st_model.requires_grad_(False)
    opt = _build_opt(currently_frozen)

    idx_all_cpu = idx_all.cpu()
    y_all_cpu = y_all.cpu()

    def run_split(mask: np.ndarray) -> tuple:
        st_model.eval()
        head.eval()
        m = torch.tensor(mask, dtype=torch.bool)
        ii = idx_all_cpu[m]
        yy = y_all_cpu[m]
        logits = []
        with torch.no_grad():
            for s in range(0, len(ii), batch_size):
                chunk = ii[s : s + batch_size]
                emb = get_src_embed(
                    st_model, bundle, chunk, ctx_sample, tmp_sample, device
                )
                logits.append(head(emb).squeeze(-1))
        log = torch.cat(logits, dim=0)
        prob = torch.sigmoid(log).cpu().numpy()
        return yy.numpy(), prob

    best_ap = -1.0
    bad = 0
    best_state = None

    for epoch in range(int(args.epochs)):
        # Phase-2 切换：到达 freeze_epochs 时解冻编码器，换差分 LR 优化器
        if currently_frozen and epoch >= freeze_epochs:
            st_model.requires_grad_(True)
            currently_frozen = False
            opt = _build_opt(frozen=False)

        st_model.train()
        head.train()
        tr_idx = np.where(tr_m)[0]
        np.random.shuffle(tr_idx)
        total_loss = 0.0
        n_batches = 0
        for s in range(0, len(tr_idx), batch_size):
            bi = tr_idx[s : s + batch_size]
            chunk = idx_all_cpu[torch.from_numpy(bi)]
            yb = y_all_cpu[torch.from_numpy(bi)].to(device)
            opt.zero_grad()
            emb = get_src_embed(st_model, bundle, chunk, ctx_sample, tmp_sample, device)
            logit = head(emb).squeeze(-1)
            loss = criterion(logit, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        y_va, p_va = run_split(va_m)
        m_va = binary_metrics(y_va, p_va, acc_threshold=args.acc_threshold)
        ap = float(m_va.get("auprc") or 0.0)
        if ap != ap:  # NaN
            ap = 0.0
        if ap > best_ap:
            best_ap = ap
            bad = 0
            best_state = {
                "st": {k: v.cpu().clone() for k, v in st_model.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }
        else:
            bad += 1
        if bad >= int(args.patience):
            break

    if best_state is not None:
        st_model.load_state_dict(best_state["st"])
        head.load_state_dict(best_state["head"])

    y_te, p_te = run_split(te_m)
    m_te = binary_metrics(y_te, p_te, acc_threshold=args.acc_threshold)

    out_root = DOWNSTREAM_ROOT / "outputs" / mod
    out_root.mkdir(parents=True, exist_ok=True)
    tag = "e2_pretrain" if args.init == "pretrain" else "e1_scratch"
    ckpt_out = out_root / f"st_pass_{tag}.pt"
    torch.save(
        {
            "st_state": st_model.state_dict(),
            "head_state": head.state_dict(),
            "init": args.init,
            "indim": indim,
            "out_dim": out_dim,
            "ctx_sample": ctx_sample,
            "tmp_sample": tmp_sample,
        },
        ckpt_out,
    )

    roc_name = f"roc_test_{tag}.png"
    roc_path = out_root / roc_name
    roc_saved = False
    if not args.no_roc_plot:
        roc_saved = save_roc_curve_png(
            roc_path,
            y_te,
            p_te,
            title=f"{mod} test ROC ({tag})",
        )

    rec = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "module": mod,
        "init": args.init,
        "ctx_sample": ctx_sample,
        "tmp_sample": tmp_sample,
        "seed": seed,
        "t_star_relative_day": t_star,
        "freeze_epochs": freeze_epochs,
        "head_lr_factor": head_lr_factor,
        "test_acc": m_te.get("acc"),
        "acc_threshold": m_te.get("acc_threshold"),
        "test_auroc": m_te.get("auroc"),
        "test_auprc": m_te.get("auprc"),
        "test_f1": m_te.get("f1"),
        "wall_time_sec": round(time.perf_counter() - wall_t0, 3),
        "epochs_ran": epoch + 1,
        "checkpoint": str(ckpt_out.relative_to(DOWNSTREAM_ROOT)).replace("\\", "/"),
        "roc_plot": str(roc_path.relative_to(DOWNSTREAM_ROOT)).replace("\\", "/")
        if roc_saved
        else None,
    }
    with open(out_root / "runs_st.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if args.dump_run_json:
        dj = Path(args.dump_run_json)
        dj.parent.mkdir(parents=True, exist_ok=True)
        with open(dj, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False))
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    if not args.no_roc_plot and not roc_saved:
        print(
            "[提示] 未保存 ROC 图：请 pip install matplotlib，或测试集仅含单一类别。"
        )


if __name__ == "__main__":
    main()
