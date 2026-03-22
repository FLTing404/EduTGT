"""与 main.py 一致的链路二分类指标（pos/neg 拼接后计算）。"""
from __future__ import annotations

import numpy as np
import scipy.special
from sklearn.metrics import average_precision_score, roc_auc_score


def link_prediction_metrics(pos_logits: np.ndarray, neg_logits: np.ndarray) -> dict:
    """输入为 BCEWithLogits 的 logits；AUC/AP 用 logits 与用 sigmoid 等价于秩一致。"""
    pos_logits = np.asarray(pos_logits).reshape(-1)
    neg_logits = np.asarray(neg_logits).reshape(-1)
    if pos_logits.size == 0 or neg_logits.size == 0:
        return {"auc": float("nan"), "ap": float("nan"), "acc": float("nan")}
    y_score = np.concatenate([pos_logits, neg_logits])
    y_prob = scipy.special.expit(y_score)
    y_true = np.concatenate([np.ones(len(pos_logits)), np.zeros(len(neg_logits))])
    pred = y_prob > 0.5
    acc = float((pred == y_true).mean())
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y_true, y_prob))
    except ValueError:
        ap = float("nan")
    return {"auc": auc, "ap": ap, "acc": acc}
