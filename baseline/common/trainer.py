"""通用早停与日志（各 baseline runner 可复用）。"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Optional


class EarlyStoppingAP:
    """以验证集 AP 为准的早停（与主项目思路类似）。"""

    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.best_epoch = 0
        self.num_bad = 0
        self.should_stop = False

    def step(self, epoch: int, val_ap: float) -> bool:
        if val_ap > self.best + self.min_delta:
            self.best = val_ap
            self.best_epoch = epoch
            self.num_bad = 0
            return True
        self.num_bad += 1
        if self.num_bad >= self.patience:
            self.should_stop = True
        return False


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def format_metrics_row(name: str, m: Dict[str, float], t_sec: float, ep: int) -> str:
    return (
        f"{name:12s}  val_auc={m.get('val_auc', float('nan')):.4f}  val_ap={m.get('val_ap', float('nan')):.4f}  "
        f"test_auc={m.get('test_auc', float('nan')):.4f}  test_ap={m.get('test_ap', float('nan')):.4f}  "
        f"time={t_sec:.1f}s  best_ep={ep}"
    )
