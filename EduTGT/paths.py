"""
EduTGT 项目路径：代码与数据位于 EduTGT/，训练产物统一到仓库 results/。
所有训练脚本应由此模块解析路径，避免硬编码散落。
"""
from __future__ import annotations

from pathlib import Path

# 主代码目录（本文件与 pretrain.py 同级）
ROOT: Path = Path(__file__).resolve().parent
REPOSITORY_ROOT: Path = ROOT.parent
RESULTS_ROOT: Path = REPOSITORY_ROOT / "results"

# ---------- 训练产出 ----------
OUTPUT_ROOT: Path = RESULTS_ROOT / "link_prediction"
RESULT_DIR: Path = OUTPUT_ROOT / "batches"
BASELINE_RESULTS_DIR: Path = OUTPUT_ROOT / "baselines"
PRETRAIN_DIR: Path = OUTPUT_ROOT / "pretrain"
CHECKPOINTS_DIR: Path = OUTPUT_ROOT / "checkpoints"
MODELS_DIR: Path = OUTPUT_ROOT / "models"
MIDDLE_DIR: Path = OUTPUT_ROOT / "middle"
LOGS_DIR: Path = OUTPUT_ROOT / "logs"
TRAINING_RUNS_JSONL: Path = LOGS_DIR / "training_runs.jsonl"

# ---------- 数据（方案 C）----------
DATA_ROOT: Path = ROOT / "data"
PROCESSED_DATA_DIR: Path = DATA_ROOT / "processed"  # data/processed/data_AAA/ ...
RAW_OULAD_DIR: Path = DATA_ROOT / "raw" / "OULAD"
DATA_SCRIPTS_DIR: Path = DATA_ROOT / "scripts"


def ensure_output_dirs() -> None:
    for d in (PRETRAIN_DIR, CHECKPOINTS_DIR, MODELS_DIR, MIDDLE_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def pretrain_ckpt_path(data_name: str) -> Path:
    return PRETRAIN_DIR / f"{data_name}.pth"


def checkpoint_ckpt_path(stem: str) -> Path:
    return CHECKPOINTS_DIR / f"{stem}.pth"


def middle_ckpt_path(data_name: str) -> Path:
    return MIDDLE_DIR / f"{data_name}.pth"


def model_save_path(filename_stem: str) -> Path:
    """filename_stem 如 data_AAA、data_AAA_pres_rel、data_AAA_stc（含后缀名不含 .pth）。"""
    return MODELS_DIR / f"{filename_stem}.pth"
