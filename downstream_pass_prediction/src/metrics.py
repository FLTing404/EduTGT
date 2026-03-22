from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        auc as sk_auc,
        average_precision_score,
        roc_auc_score,
        roc_curve,
    )
except ImportError:
    roc_auc_score = None  # type: ignore
    average_precision_score = None  # type: ignore
    accuracy_score = None  # type: ignore
    roc_curve = None  # type: ignore
    sk_auc = None  # type: ignore


def binary_accuracy(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> float:
    """将分数按阈值二值化后与真实标签比较。"""
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    y_pred = (y_score >= threshold).astype(np.int32)
    if accuracy_score is not None:
        return float(accuracy_score(y_true, y_pred))
    return float(np.mean(y_pred == y_true))


def binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, acc_threshold: float = 0.5
) -> dict:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    out: dict = {
        "acc": binary_accuracy(y_true, y_score, threshold=acc_threshold),
        "acc_threshold": float(acc_threshold),
    }
    if len(np.unique(y_true)) < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
        return out
    if roc_auc_score is not None:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def save_roc_curve_png(
    path: Path,
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: Optional[str] = None,
) -> bool:
    """
    保存 ROC 曲线图（横轴 FPR、纵轴 TPR）；与 test_auroc 同一套 sklearn 定义。
    若仅单一类别或缺少依赖则跳过并返回 False。
    """
    if roc_curve is None:
        return False
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    if len(np.unique(y_true)) < 2:
        return False
    if sk_auc is None:
        return False
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = float(sk_auc(fpr, tpr))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("假正率 (FPR)")
    plt.ylabel("真正率 (TPR)")
    plt.title(title or "ROC 曲线 (test)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return True
