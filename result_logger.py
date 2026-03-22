"""
训练结束写一条 JSON 记录到 outputs/logs/training_runs.jsonl，并打印一行摘要。
设计方案见 docs/training_result_log.md。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import paths


def _default_json_path() -> str:
    paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return str(paths.TRAINING_RUNS_JSONL)


def log_training_result(
    *,
    run_type: str,
    data_name: str,
    data_dir: Optional[str],
    hyperparams: Dict[str, Any],
    test_auc: float,
    test_ap: float,
    test_acc: float,
    test_loss: float,
    nn_test_auc: float,
    nn_test_ap: float,
    nn_test_acc: float,
    nn_test_loss: float,
    early_stopped: bool,
    best_epoch: Optional[int],
    out_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    run_type: baseline | ablation_pres_relation | ablation_student_temporal
    """
    path = out_path or _default_json_path()
    record: Dict[str, Any] = {
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "run_type": run_type,
        "data_name": data_name,
        "data_dir": data_dir,
        "hyperparams": hyperparams,
        "metrics": {
            "test": {
                "auc": float(test_auc),
                "ap": float(test_ap),
                "acc": float(test_acc),
                "loss": float(test_loss),
            },
            "nn_test": {
                "auc": float(nn_test_auc),
                "ap": float(nn_test_ap),
                "acc": float(nn_test_acc),
                "loss": float(nn_test_loss),
            },
        },
        "training": {
            "early_stopped": early_stopped,
            "best_epoch": best_epoch,
        },
    }
    if extra:
        record["extra"] = extra

    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    summary = (
        f"[RESULT] {run_type} | {data_name} | "
        f"test_auc={test_auc:.6f} test_ap={test_ap:.6f} | "
        f"nn_test_auc={nn_test_auc:.6f} nn_test_ap={nn_test_ap:.6f} | "
        f"-> {path}"
    )
    print(summary)
    return path
