"""
与论文 Section V-B 一致的评估：多 seed 运行并汇总 mean ± std。
用法（在项目根 EduTGT 下）：
  # 默认：跑当前支持的所有链路模型（EduTGT + TGAT + GraphSAGE + JODIE），并按 seeds 汇总
  python link_models/run_paper_eval.py --data_dir data_abc_0.01 --seeds 42,43,44,45,46

  # 只跑部分方法（如仅 EduTGT 和 TGAT）
  python link_models/run_paper_eval.py --data_dir data_abc_0.01 --seeds 42 --methods edutgt,tgat
"""
import os
import sys
import argparse
import subprocess
import shutil
import pandas as pd
import numpy as np

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# EduTGT 多 seed 时预训练按 seed 存盘，使用固定后缀 paper_eval，与消融的 baseline/neg 等区分
PAPER_EVAL_ABLATION_SUFFIX = 'paper_eval'


def run_pretrain(data_dir, seed, code_root):
    """先跑预训练，供 edutgt 的 main_link 加载。带 ablation_suffix 以便按 seed 存为 <data>_paper_eval_seed<seed>.pth。"""
    pretrain_script = os.path.join(code_root, 'script', 'pretrain.py')
    cmd = [
        sys.executable, pretrain_script,
        '--data_dir', data_dir,
        '--seed', str(seed),
        '--ablation_suffix', PAPER_EVAL_ABLATION_SUFFIX,
    ]
    ret = subprocess.run(cmd, cwd=code_root)
    return ret.returncode == 0


def run_one(script_name, data_dir, seed, code_root, extra_args=None):
    """运行单次，返回是否成功。extra_args 会追加到命令（如 edutgt 需 --ablation_suffix paper_eval）。"""
    # 与 ContraTGT 及基线统一：按时间划分（quantile 0.1, 0.2），不传 --paper_eval
    cmd = [
        sys.executable,
        os.path.join(code_root, 'link_models', script_name),
        '--data_dir', data_dir,
        '--seed', str(seed),
    ]
    if extra_args:
        cmd.extend(extra_args)
    ret = subprocess.run(cmd, cwd=code_root)
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='论文 Section V-B 评估：多 seed 运行并汇总 mean±std')
    parser.add_argument('--data_dir', type=str, required=True, help='如 data_abc_0.01')
    parser.add_argument('--seeds', type=str, default='42,43,44,45,46', help='逗号分隔的随机种子')
    # 默认跑所有支持的方法：EduTGT + TGAT + GraphSAGE + JODIE
    parser.add_argument('--methods', type=str, default='edutgt,tgat,graphsage,jodie',
                        help='逗号分隔：edutgt,tgat,graphsage,jodie')
    parser.add_argument('--no_run', action='store_true', help='仅汇总已有 CSV，不重新跑')
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(',')]
    methods = [x.strip().lower() for x in args.methods.split(',')]
    result_dir = os.path.join(CODE_ROOT, 'result', 'link', args.data_dir)
    data_name = args.data_dir
    os.makedirs(result_dir, exist_ok=True)

    method_script = {
        'edutgt': 'main_link.py',
        'tgat': 'train_link_tgat.py',
        'graphsage': 'train_link_graphsage.py',
        'jodie': 'train_link_jodie.py',
    }
    method_tag = {'edutgt': 'edutgt', 'tgat': 'tgat', 'graphsage': 'graphsage', 'jodie': 'jodie'}

    if not args.no_run:
        for method in methods:
            script_name = method_script.get(method)
            if not script_name:
                continue
            tag = method_tag.get(method, method)
            for seed in seeds:
                if method == 'edutgt':
                    # EduTGT 需先预训练再链路；预训练按 seed 存为 <data>_paper_eval_seed<seed>.pth，main_link 用同一 suffix+seed 加载
                    print(f"运行 {method} seed={seed}（先预训练再链路）...")
                    if not run_pretrain(args.data_dir, seed, CODE_ROOT):
                        print(f"  warning: edutgt 预训练 seed={seed} 返回非 0")
                    if not run_one(script_name, args.data_dir, seed, CODE_ROOT,
                                   extra_args=['--ablation_suffix', PAPER_EVAL_ABLATION_SUFFIX]):
                        print(f"  warning: {method} seed={seed} 返回非 0")
                else:
                    print(f"运行 {method} seed={seed} ...")
                    ok = run_one(script_name, args.data_dir, seed, CODE_ROOT)
                    if not ok:
                        print(f"  warning: {method} seed={seed} 返回非 0")
                src = os.path.join(result_dir, f'link_{tag}_{data_name}.csv')
                dst = os.path.join(result_dir, f'link_{tag}_{data_name}_s{seed}.csv')
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    print(f"  已保存 {dst}")

    summary = []
    for method in methods:
        tag = method_tag.get(method, method)
        collected = []
        for s in seeds:
            p = os.path.join(result_dir, f'link_{tag}_{data_name}_s{s}.csv')
            if not os.path.exists(p):
                p = os.path.join(result_dir, f'link_{tag}_{data_name}.csv')
            if os.path.exists(p):
                collected.append(pd.read_csv(p).iloc[0])
        if not collected:
            continue
        agg_df = pd.DataFrame(collected)
        means = agg_df.mean(numeric_only=True)
        stds = agg_df.std(numeric_only=True)
        row = {'Method': method}
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
        print("未找到可汇总的 CSV。请先不带 --no_run 运行本脚本，或手动跑各方法并保存 link_*_s{seed}.csv。")


if __name__ == '__main__':
    main()
