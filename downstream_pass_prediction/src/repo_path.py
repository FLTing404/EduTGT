"""将 EduTGT 代码根目录加入 sys.path（仅用于 import，不修改该目录下文件）。"""
from __future__ import annotations

import sys
from pathlib import Path

_DOWNSTREAM_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = _DOWNSTREAM_ROOT.parent
CODE_ROOT = REPO_ROOT / "EduTGT"


def ensure_repo_on_path() -> None:
    r = str(CODE_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)
