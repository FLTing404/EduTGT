#!/usr/bin/env python3
"""
一键消融实验批跑脚本（4 种消融方法 × N 个 seed → CSV）

消融方法（共 4 种）：
  1. ablation_pres          — pres 关系偏置负采样（λ=0.35），原始绝对时间戳
  2. ablation_stc           — 学生时序一致性损失（在 ContraTGT 基础上加 L_TC）
  3. ablation_no_temporal   — 去掉时序分支（tmp_sample=1），仅保留空间上下文
  4. ablation_course_progress — pres 偏置采样 + 课程进度归一化时间戳（[0,1]）

前置条件：
  - 已存在 outputs/pretrain/data_<DATASET>.pth（提前跑过 pretrain.py）
  - 已存在 data/processed/data_<DATASET>/pres_relation.json
    （方法 1/3/4 需要；可用 python data/scripts/build_pres_relation.py --data_dir data/processed/data_<DATASET> 生成）

用法（在 EduTGT/EduTGT 下）：
  python run_ablation_batch.py --dataset AAA
  python run_ablation_batch.py --dataset CCC --seeds 1,2,3,4,5
  python run_ablation_batch.py --dataset AAA --seeds 42,2,2026,7,99 --gpu 0 --n_epoch 50

输出：
  outputs/result/ablation_<DATASET>_<时间戳>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
METHODS_DIR = ROOT / "methods"
RESULT_DIR = ROOT / "outputs" / "result"
JSONL_PATH = ROOT / "outputs" / "logs" / "training_runs.jsonl"

DATASET_MODULES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")

CSV_FIELDS = [
    "dataset",
    "method",
    "seed",
    "pres_mix_uniform",
    "AUC",
    "ACC",
    "AP",
    "Loss",
    "epoch",
    "time_cost",
    "nn_test_auc",
    "nn_test_ap",
    "nn_test_acc",
    "status",
]


def _env() -> dict:
    e = os.environ.copy()
    e["PYTHONPATH"] = str(ROOT) + os.pathsep + e.get("PYTHONPATH", "")
    return e


def _jsonl_byte_offset() -> int:
    if not JSONL_PATH.is_file():
        return 0
    return JSONL_PATH.stat().st_size


def _parse_new_jsonl_record(start_byte: int) -> Optional[Dict[str, Any]]:
    if not JSONL_PATH.is_file():
        return None
    with open(JSONL_PATH, "rb") as f:
        f.seek(start_byte)
        chunk = f.read().decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _row_from_jsonl(
    rec: Dict[str, Any],
    dataset: str,
    method: str,
    seed: int,
    pres_mix: str,
    elapsed: float,
) -> Dict[str, Any]:
    mt = rec.get("metrics", {})
    test = mt.get("test", {})
    nn = mt.get("nn_test", {})
    tr = rec.get("training", {})
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "pres_mix_uniform": pres_mix,
        "AUC": test.get("auc", ""),
        "ACC": test.get("acc", ""),
        "AP": test.get("ap", ""),
        "Loss": test.get("loss", ""),
        "epoch": tr.get("best_epoch", ""),
        "time_cost": round(elapsed, 3),
        "nn_test_auc": nn.get("auc", ""),
        "nn_test_ap": nn.get("ap", ""),
        "nn_test_acc": nn.get("acc", ""),
        "status": "ok",
    }


def _fail_row(
    dataset: str, method: str, seed: int, pres_mix: str, elapsed: float, msg: str
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "pres_mix_uniform": pres_mix,
        "AUC": "",
        "ACC": "",
        "AP": "",
        "Loss": "",
        "epoch": "",
        "time_cost": round(elapsed, 3),
        "nn_test_auc": "",
        "nn_test_ap": "",
        "nn_test_acc": "",
        "status": msg[:200],
    }


def _run_job(
    cmd: List[str],
    *,
    dataset: str,
    method: str,
    seed: int,
    pres_mix: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    off = _jsonl_byte_offset()
    try:
        subprocess.run(cmd, cwd=str(ROOT), env=_env(), check=True)
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - t0
        return _fail_row(dataset, method, seed, pres_mix, elapsed, f"exit_{e.returncode}")

    elapsed = time.perf_counter() - t0
    rec = _parse_new_jsonl_record(off)
    if not rec:
        return _fail_row(dataset, method, seed, pres_mix, elapsed, "no_jsonl_record")
    return _row_from_jsonl(rec, dataset, method, seed, pres_mix, elapsed)


def main() -> None:
    p = argparse.ArgumentParser(
        description="一键消融批跑：4 种消融方法 × N 个 seed → CSV（不含预训练）"
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_MODULES,
        help="整模块代码，对应 data/processed/data_<MODULE>",
    )
    p.add_argument(
        "--seeds",
        default="1,2,3,4,5",
        help="逗号分隔的整数 seed，默认 1,2,3,4,5",
    )
    p.add_argument("--n_epoch", type=int, default=50)
    p.add_argument("--bs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ctx_sample", type=int, default=40)
    p.add_argument("--tmp_sample", type=int, default=31)
    p.add_argument("--drop_out", type=float, default=0.2)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--lambda_tc", type=float, default=0.1, help="ablation_stc 时序一致性损失权重")
    p.add_argument(
        "--out_csv",
        default=None,
        help="输出 CSV 路径，默认 outputs/result/ablation_<DATASET>_<时间戳>.csv",
    )
    args = p.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        print("错误：--seeds 为空", file=sys.stderr)
        sys.exit(1)

    data_name = f"data_{args.dataset}"
    data_dir = ROOT / "data" / "processed" / data_name
    if not data_dir.is_dir():
        print(f"错误：数据目录不存在 {data_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        from paths import pretrain_ckpt_path
        pt = pretrain_ckpt_path(data_name)
        if not pt.is_file():
            print(f"警告：未找到预训练权重 {pt}，pres/course 消融将报错；请先跑 pretrain.py", file=sys.stderr)
    except ImportError:
        pass

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_csv) if args.out_csv else RESULT_DIR / f"ablation_{args.dataset}_{tag}.csv"
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()

    def common_args(seed: int, tmp_override: Optional[int] = None) -> List[str]:
        tmp = tmp_override if tmp_override is not None else args.tmp_sample
        return [
            "--data_dir", str(data_dir),
            "--n_epoch", str(args.n_epoch),
            "--bs", str(args.bs),
            "--lr", str(args.lr),
            "--ctx_sample", str(args.ctx_sample),
            "--tmp_sample", str(tmp),
            "--drop_out", str(args.drop_out),
            "--gpu", str(args.gpu),
            "--seed", str(seed),
        ]

    # ------------------------------------------------------------------ #
    # 构建任务列表：(method_label, pres_mix_str, cmd)
    # ------------------------------------------------------------------ #
    jobs: List[tuple] = []
    for seed in seeds:
        # 1. ablation_pres (λ=0.35)
        jobs.append((
            "ablation_pres", seed, "0.35",
            [sys.executable, str(METHODS_DIR / "main_ablation_pres_relation.py"),
             *common_args(seed), "--pres_mix_uniform", "0.35"],
        ))
        # 2. ablation_stc
        jobs.append((
            "ablation_stc", seed, "",
            [sys.executable, str(METHODS_DIR / "main_ablation_student_temporal.py"),
             "--lambda_tc", str(args.lambda_tc),
             *common_args(seed)],
        ))
        # 3. ablation_no_temporal (去掉时序分支：tmp_sample=1)
        jobs.append((
            "ablation_no_temporal", seed, "0.35",
            [sys.executable, str(METHODS_DIR / "main_ablation_pres_relation.py"),
             *common_args(seed, tmp_override=1), "--pres_mix_uniform", "0.35"],
        ))
        # 4. ablation_course_progress (归一化时间戳)
        jobs.append((
            "ablation_course_progress", seed, "0.35",
            [sys.executable, str(METHODS_DIR / "main_ablation_course_progress.py"),
             *common_args(seed), "--pres_mix_uniform", "0.35"],
        ))

    total = len(jobs)
    print(
        f"\n[run_ablation_batch] 数据集={args.dataset}  seed={seeds}  共 {total} 个任务\n"
        f"  输出 CSV → {out_path}\n",
        file=sys.stderr,
    )

    rows: List[Dict[str, Any]] = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()

        for i, (method, seed, pres_mix, cmd) in enumerate(jobs, 1):
            print(
                f"[{i}/{total}] {method}  seed={seed}  pres_mix={pres_mix or '-'} ...",
                file=sys.stderr,
            )
            row = _run_job(cmd, dataset=args.dataset, method=method, seed=seed, pres_mix=pres_mix)
            rows.append(row)
            writer.writerow(row)
            f.flush()
            status = row.get("status", "?")
            auc = row.get("AUC", "")
            print(f"    → status={status}  AUC={auc}", file=sys.stderr)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(
        f"\n[run_ablation_batch] 完成 {ok}/{total}  CSV → {out_path}\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
