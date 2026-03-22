#!/usr/bin/env python3
"""
一键顺序运行两项微调消融（需已存在 outputs/pretrain/{data_dir 文件夹名}.pth）：
1) main_ablation_pres_relation.py — 训练期按 pres_relation.json 偏置负样本 dst（需先有 pres_relation.json）
2) main_ablation_student_temporal.py — batch 内同 u 源嵌入时序一致性

产物：
- outputs/models/<data_name>_pres_rel.pth、<data_name>_stc.pth
- outputs/checkpoints/<同上>.pth（EarlyStopping）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="顺序运行 presentation 关系负采样与同学生时序一致性消融微调")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="与 main.py 相同，如 data/processed/data_AAA_2013J",
    )
    p.add_argument("--n_epoch", type=int, default=50)
    p.add_argument("--bs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ctx_sample", type=int, default=40)
    p.add_argument("--tmp_sample", type=int, default=31)
    p.add_argument("--drop_out", type=float, default=0.2)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--lambda_tc", type=float, default=0.1, help="仅 student temporal 脚本")
    p.add_argument(
        "--pres_mix_uniform",
        type=float,
        default=0.35,
        help="仅 pres_relation 脚本：负样本 dst 均匀混合比例",
    )
    p.add_argument("--skip_pres", action="store_true", help="跳过 presentation 关系消融")
    p.add_argument(
        "--skip_edge",
        action="store_true",
        help="已弃用，等同于 --skip_pres（保留以兼容旧命令）",
    )
    p.add_argument("--skip_stc", action="store_true", help="跳过时序一致性消融")
    args = p.parse_args()
    skip_pres = args.skip_pres or args.skip_edge

    common = [
        "--data_dir",
        args.data_dir,
        "--n_epoch",
        str(args.n_epoch),
        "--bs",
        str(args.bs),
        "--lr",
        str(args.lr),
        "--ctx_sample",
        str(args.ctx_sample),
        "--tmp_sample",
        str(args.tmp_sample),
        "--drop_out",
        str(args.drop_out),
        "--gpu",
        str(args.gpu),
        "--pres_mix_uniform",
        str(args.pres_mix_uniform),
    ]

    if not skip_pres:
        print("========== [1/2] presentation 关系负采样 main_ablation_pres_relation.py ==========")
        subprocess.check_call(
            [sys.executable, os.path.join(root, "main_ablation_pres_relation.py")] + common,
            cwd=root,
        )
    else:
        print("已跳过 presentation 关系消融 (--skip_pres)")

    if not args.skip_stc:
        print("========== [2/2] 同学生时序一致性 main_ablation_student_temporal.py ==========")
        subprocess.check_call(
            [
                sys.executable,
                os.path.join(root, "main_ablation_student_temporal.py"),
                "--lambda_tc",
                str(args.lambda_tc),
            ]
            + common,
            cwd=root,
        )
    else:
        print("已跳过时序一致性消融 (--skip_stc)")

    print("消融运行结束。")


if __name__ == "__main__":
    main()
