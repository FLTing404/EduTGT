from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_TIMESTAMP_UNITS_PER_SECOND = {
    "assist12": 1_000_000_000.0,  # pandas datetime64[ns]
    "ednet": 1_000.0,             # EdNet timestamps are milliseconds
    "oulad": 1.0 / 86_400.0,      # OULAD timestamps are integer days
    "xes3g5m": 1_000.0,          # official timestamp tokens are milliseconds
    "assist17": 1.0,             # startTime epoch seconds
    "nips34": 1_000_000_000.0,   # pandas datetime64[ns], source precision one minute
}
_MAX_TIME_GAP_SECONDS = 365.0 * 86_400.0


def _normalized_time_gaps(
    timestamps: np.ndarray, dataset_name: str, time_valid: np.ndarray | None = None
) -> np.ndarray:
    """Return log-scaled time since the previous interaction in [0, 1]."""
    values = timestamps.astype(np.float64, copy=False)
    gaps = np.zeros(len(values), dtype=np.float64)
    if len(values) > 1:
        gaps[1:] = np.maximum(values[1:] - values[:-1], 0.0)
    units_per_second = _TIMESTAMP_UNITS_PER_SECOND.get(dataset_name, 1.0)
    seconds = np.clip(gaps / units_per_second, 0.0, _MAX_TIME_GAP_SECONDS)
    if time_valid is not None:
        valid = time_valid.astype(bool)
        seconds[~valid] = 0.0
        if len(values) > 1:
            seconds[1:][~valid[:-1]] = 0.0
    if dataset_name == "xes3g5m":
        seconds[gaps / units_per_second > _MAX_TIME_GAP_SECONDS] = 0.0
    return (np.log1p(seconds) / np.log1p(_MAX_TIME_GAP_SECONDS)).astype(np.float32)


class SequenceDataset(Dataset):
    def __init__(self, prepared_dir: str | Path, split: str, max_length: int, min_length: int = 3):
        prepared_dir = Path(prepared_dir)
        frame = pd.read_parquet(prepared_dir / "interactions.parquet")
        if "event_idx" not in frame:
            frame["event_idx"] = np.arange(len(frame), dtype=np.int64)
        users = set(np.load(prepared_dir / "splits.npz")[split].tolist())
        frame = frame[frame["user_idx"].isin(users)]
        if "sequence_idx" in frame:
            ordering = "event_order" if "event_order" in frame else "timestamp"
            frame = frame.sort_values(["user_idx", "sequence_idx", ordering], kind="mergesort")
            group_columns = ["user_idx", "sequence_idx"]
        else:
            frame = frame.sort_values(["user_idx", "timestamp"], kind="mergesort")
            group_columns = ["user_idx"]
        self.max_length = int(max_length)
        self.samples: list[dict[str, np.ndarray]] = []
        for _, group in frame.groupby(group_columns, sort=False):
            items = group["item_idx"].to_numpy(np.int64)
            concepts = group["concept_idx"].to_numpy(np.int64)
            correct = group["correct"].to_numpy(np.float32)
            valid_time = group["time_valid"].to_numpy(bool) if "time_valid" in group else None
            time_gaps = _normalized_time_gaps(group["timestamp"].to_numpy(), prepared_dir.name, valid_time)
            target_valid = group["target_valid"].to_numpy(bool) if "target_valid" in group else np.ones(len(group), dtype=bool)
            event_ids = group["event_idx"].to_numpy(np.int64)
            user_idx = int(group["user_idx"].iloc[0])
            if len(items) < min_length:
                continue
            for start in range(0, len(items) - 1, self.max_length):
                end = min(start + self.max_length + 1, len(items))
                if end - start < min_length:
                    continue
                length = end - start - 1
                sample = {
                    "item": np.zeros(self.max_length, dtype=np.int64),
                    "concept": np.zeros(self.max_length, dtype=np.int64),
                    "correct": np.zeros(self.max_length, dtype=np.float32),
                    "next_item": np.zeros(self.max_length, dtype=np.int64),
                    "next_concept": np.zeros(self.max_length, dtype=np.int64),
                    "target": np.zeros(self.max_length, dtype=np.float32),
                    "time_gap": np.zeros(self.max_length, dtype=np.float32),
                    "next_time_gap": np.zeros(self.max_length, dtype=np.float32),
                    "mask": np.zeros(self.max_length, dtype=np.bool_),
                    "user_idx": np.full(self.max_length, -1, dtype=np.int64),
                    "event_idx": np.full(self.max_length, -1, dtype=np.int64),
                    "history_position": np.full(self.max_length, -1, dtype=np.int64),
                }
                sl = slice(start, end)
                sample["item"][:length] = items[sl][:-1]
                sample["concept"][:length] = concepts[sl][:-1]
                sample["correct"][:length] = correct[sl][:-1]
                sample["next_item"][:length] = items[sl][1:]
                sample["next_concept"][:length] = concepts[sl][1:]
                sample["target"][:length] = correct[sl][1:]
                sample["time_gap"][:length] = time_gaps[sl][:-1]
                sample["next_time_gap"][:length] = time_gaps[sl][1:]
                sample["mask"][:length] = target_valid[sl][1:]
                sample["user_idx"][:length] = user_idx
                sample["event_idx"][:length] = event_ids[sl][1:]
                sample["history_position"][:length] = np.arange(start + 1, end, dtype=np.int64)
                if sample["mask"].any():
                    self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: torch.from_numpy(value) for key, value in self.samples[index].items()}


def load_graph(prepared_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prepared_dir = Path(prepared_dir)
    graph = np.load(prepared_dir / "hypergraph.npz")
    difficulty = np.load(prepared_dir / "difficulty.npy")
    return graph["item"], graph["concept"], difficulty
