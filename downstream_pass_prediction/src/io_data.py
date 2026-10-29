"""只读加载 processed 图与 stats（不写回任何文件）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

TS_PRES_BLOCK = 1000


def load_stats(processed: Path) -> Dict[str, Any]:
    p = processed / "stats.json"
    if not p.is_file():
        raise FileNotFoundError(f"缺少 stats.json: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def find_ml_csv(processed: Path) -> Path:
    files = sorted(processed.glob("ml_*.csv"))
    if not files:
        raise FileNotFoundError(f"{processed} 下无 ml_*.csv")
    return files[0]


def find_content_csv(processed: Path, ml_stem: str) -> Path:
    """与 utils.resolve_training_data 类似：优先匹配 ml  stem。"""
    files = sorted(processed.glob("*.content"))
    if not files:
        raise FileNotFoundError(f"{processed} 下无 *.content")
    tag = ml_stem[3:-4] if ml_stem.startswith("ml_") and ml_stem.endswith(".csv") else ""
    preferred = [f for f in files if tag and tag in f.name]
    return preferred[0] if preferred else files[0]


def load_ml_triplets(ml_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(ml_path)
    # 首列常为无名列索引 0,1,2...
    cols = {c.lower().strip(): c for c in df.columns}
    u_col = cols.get("u") or "u"
    i_col = cols.get("i") or "i"
    ts_col = cols.get("ts") or "ts"
    u = df[u_col].to_numpy(dtype=np.int64)
    i = df[i_col].to_numpy(dtype=np.int64)
    ts = df[ts_col].to_numpy(dtype=np.int64)
    return u, i, ts


def load_content_matrix(content_path: Path) -> np.ndarray:
    return np.loadtxt(content_path, delimiter=",", dtype=np.float64)


def load_node_student_map(node_map_path: Path) -> Dict[int, int]:
    """ml 节点 id -> id_student（仅 student 节点）。"""
    df = pd.read_csv(node_map_path)
    out: Dict[int, int] = {}
    for _, row in df.iterrows():
        nid = int(row["node_id"])
        raw = str(row["raw_id"])
        if raw.startswith("student::"):
            sid = int(raw.split("::", 1)[1])
            out[nid] = sid
    return out


def student_to_ml_node(node_map_path: Path) -> Dict[int, int]:
    """id_student -> ml 节点 id。"""
    df = pd.read_csv(node_map_path)
    out: Dict[int, int] = {}
    for _, row in df.iterrows():
        raw = str(row["raw_id"])
        if raw.startswith("student::"):
            sid = int(raw.split("::", 1)[1])
            out[sid] = int(row["node_id"])
    return out


def presentation_to_block(code_presentation: str, stats: Dict[str, Any]) -> int:
    pres: List[str] = stats["code_presentations_merged"]
    if code_presentation not in pres:
        raise ValueError(f"presentation {code_presentation!r} 不在 stats.code_presentations_merged: {pres}")
    return pres.index(code_presentation)
