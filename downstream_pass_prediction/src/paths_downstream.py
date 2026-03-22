"""下游子项目内路径解析（不修改仓库其它目录）。"""
from __future__ import annotations

from pathlib import Path

# downstream_pass_prediction 根目录
DOWNSTREAM_ROOT: Path = Path(__file__).resolve().parent.parent

# EduTGT 代码根目录（与 main.py、data/ 同级）
REPO_ROOT: Path = DOWNSTREAM_ROOT.parent

DATA_DOWNSTREAM: Path = DOWNSTREAM_ROOT / "data"


def processed_dir(module_name: str) -> Path:
    """module_name 如 data_AAA。"""
    return REPO_ROOT / "data" / "processed" / module_name


def module_data_dir(module_name: str) -> Path:
    """下游产出：downstream_pass_prediction/data/data_AAA/"""
    return DATA_DOWNSTREAM / module_name
