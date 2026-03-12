"""
方案 A：全程按边 1:1:8，多 seed 运行并汇总 mean±std。
前提：已先运行 script/pretrain.py 生成对应数据集的预训练权重。

用法（在项目根 EduTGT 下）：
  # 多个 seed 跑 ContraTGT 下游，通过/不通过
  python passfail_models/run_paper_eval.py --data_dir data_abc_0.01 --seeds 42,43,44,45,46

  # 仅汇总已有 CSV，不重新跑 main_passfail
  python passfail_models/run_paper_eval.py --data_dir data_abc_0.01 --no_run
"""
import os
import sys
import argparse
import subprocess
import shutil
import pandas as pd
import numpy as np

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(script_name, data_dir, seed, code_root):
    """运行 main_passfail 单次（默认按边 1:1:8）。"""
    cmd = [
        sys.executable,
        os.path.join(code_root, 'passfail_models', script_name),
        '--data_dir', data_dir,
        '--seed', str(seed),
    ]
    ret = subprocess.run(cmd, cwd=code_root)
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='方案 A：按边 1:1:8 多 seed 评估并汇总 mean±std')
    parser.add_argument('--data_dir', type=str, required=True, help='如 data_abc_0.01')
    parser.add_argument('--seeds', type=str, default='42,43,44,45,46', help='逗号分隔的随机种子')
    parser.add_argument('--no_run', action='store_true', help='仅汇总已有 CSV，不重新跑')
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(',')]
    result_dir = os.path.join(CODE_ROOT, 'result', 'passfail', args.data_dir)
    data_name = args.data_dir
    os.makedirs(result_dir, exist_ok=True)

    if not args.no_run:
        for seed in seeds:
            print(f"运行 main_passfail seed={seed} ...")
            ok = run_one('main_passfail.py', args.data_dir, seed, CODE_ROOT)
            if not ok:
                print(f"  warning: main_passfail seed={seed} 返回非 0")
            src = os.path.join(result_dir, f'script_{data_name}.csv')
            dst = os.path.join(result_dir, f'passfail_contratgt_{data_name}_s{seed}.csv')
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"  已保存 {dst}")

    summary = []
    collected = []
    for s in seeds:
        p = os.path.join(result_dir, f'passfail_contratgt_{data_name}_s{s}.csv')
        if not os.path.exists(p):
            p = os.path.join(result_dir, f'script_{data_name}.csv')
        if os.path.exists(p):
            collected.append(pd.read_csv(p).iloc[0])
    if collected:
        agg_df = pd.DataFrame(collected)
        means = agg_df.mean(numeric_only=True)
        stds = agg_df.std(numeric_only=True)
        row = {'Method': 'contratgt'}
        for c in ['Test_AUC', 'Test_AP', 'Test_Acc', 'NN_Test_AUC', 'NN_Test_AP', 'NN_Test_Acc']:
            if c in means.index:
                m, s = means[c], stds.get(c, np.nan)
                row[c] = f"{m:.4f}±{s:.4f}" if pd.notna(s) and np.isfinite(s) else f"{m:.4f}"
        summary.append(row)

    if summary:
        out_df = pd.DataFrame(summary)
        out_path = os.path.join(result_dir, f'paper_eval_summary_{data_name}.csv')
        out_df.to_csv(out_path, index=False)
        print(f"\n汇总已保存: {out_path}")
        print(out_df.to_string())
    else:
        print("未找到可汇总的 CSV。请先不带 --no_run 运行本脚本，或先运行 main_passfail 并保存 passfail_contratgt_*_s{seed}.csv。")


if __name__ == '__main__':
    main()
