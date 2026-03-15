"""
方案 A：全程按边 1:1:8，多 seed 运行六种方法并汇总 mean±std。
六种：EduTGT+MLP、纯 MLP、EduTGT+Logistic、纯 Logistic、EduTGT+LSTM、纯 LSTM。
前提：已先运行 script/pretrain.py（或 link_models/run_ablation.py）生成对应数据集的预训练权重。

用法（在项目根 EduTGT 下）：
  # 无消融后缀时加载 script/pretrain_model/<data_name>.pth（需事先单独 pretrain）
  python passfail_models/run_paper_eval.py --data_dir data_abc_0.01 --seeds 42,43,44,45,46

  # 复用 run_ablation 的预训练：指定 --ablation_suffix，每个 seed 加载 <data>_<suffix>_seed<seed>.pth（跑 5 个 seed）
  python passfail_models/run_paper_eval.py --data_dir data_0.01 --seeds 42,43,44,45,46 --ablation_suffix baseline

  python passfail_models/run_paper_eval.py --data_dir data_abc_0.01 --seeds 42 --methods edutgt,logistic_pure,logistic_contratgt
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

# 六种方法：带预测头（EduTGT+头）与不带（纯）
# 脚本名、额外参数、运行后结果文件名（不含路径）
METHOD_CONFIG = {
    'edutgt': ('main_passfail.py', [], 'edutgt'),   # 输出 edutgt_{data_name}.csv
    'mlp_pure': ('train_mlp.py', [], 'mlp_pure'),
    'logistic_pure': ('train_logistic.py', ['--mode', 'pure'], 'logistic_pure'),
    'logistic_contratgt': ('train_logistic.py', ['--mode', 'contratgt'], 'logistic_contratgt'),
    'lstm_pure': ('train_lstm.py', ['--mode', 'pure'], 'lstm_pure'),
    'lstm_contratgt': ('train_lstm.py', ['--mode', 'contratgt'], 'lstm_contratgt'),
}

DEFAULT_METHODS = 'edutgt,mlp_pure,logistic_pure,logistic_contratgt,lstm_pure,lstm_contratgt'

# 需要加载预训练权重的方法（传 --ablation_suffix 时加载 <data>_<suffix>_seed<seed>.pth）
METHODS_USE_PRETRAIN = {'edutgt', 'logistic_contratgt', 'lstm_contratgt'}


def run_one(script_name, data_dir, seed, code_root, extra_args=None):
    """运行单个脚本一次，返回是否成功。"""
    cmd = [
        sys.executable,
        os.path.join(code_root, 'passfail_models', script_name),
        '--data_dir', data_dir,
        '--seed', str(seed),
    ]
    if extra_args:
        cmd.extend(extra_args)
    ret = subprocess.run(cmd, cwd=code_root)
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='方案 A：六种预测头/纯 多 seed 评估并汇总 mean±std')
    parser.add_argument('--data_dir', type=str, required=True, help='如 data_abc_0.01')
    parser.add_argument('--seeds', type=str, default='42,43,44,45,46', help='逗号分隔的随机种子')
    parser.add_argument('--methods', type=str, default=DEFAULT_METHODS,
                        help='逗号分隔，默认六种：edutgt,mlp_pure,logistic_pure,logistic_contratgt,lstm_pure,lstm_contratgt')
    parser.add_argument('--ablation_suffix', type=str, default='',
                        help='消融预训练后缀，如 baseline/neg/topk；指定后 edutgt/logistic_contratgt/lstm_contratgt 加载 script/pretrain_model/<data>_<suffix>_seed<seed>.pth，与 run_ablation 产出一致')
    parser.add_argument('--no_run', action='store_true', help='仅汇总已有 CSV，不重新跑')
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(',')]
    methods = [x.strip().lower() for x in args.methods.split(',')]
    result_dir = os.path.join(CODE_ROOT, 'result', 'passfail', args.data_dir)
    data_name = args.data_dir
    os.makedirs(result_dir, exist_ok=True)

    if not args.no_run:
        for method in methods:
            cfg = METHOD_CONFIG.get(method)
            if not cfg:
                continue
            script_name, base_extra, out_prefix = cfg
            for seed in seeds:
                extra_args = list(base_extra)
                if (args.ablation_suffix or '').strip() and method in METHODS_USE_PRETRAIN:
                    extra_args.extend(['--ablation_suffix', args.ablation_suffix.strip()])
                print(f"运行 {method} seed={seed} ...")
                ok = run_one(script_name, args.data_dir, seed, CODE_ROOT, extra_args if extra_args else None)
                if not ok:
                    print(f"  warning: {method} seed={seed} 返回非 0")
                # 脚本写出的文件名（如 edutgt_{data_name}.csv 或 logistic_pure_{data_name}.csv）
                src_name = f'{out_prefix}_{data_name}.csv'
                src = os.path.join(result_dir, src_name)
                dst = os.path.join(result_dir, f'{method}_{data_name}_s{seed}.csv')
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    print(f"  已保存 {dst}")

    summary = []
    for method in methods:
        collected = []
        for s in seeds:
            p = os.path.join(result_dir, f'{method}_{data_name}_s{s}.csv')
            if not os.path.exists(p):
                cfg = METHOD_CONFIG.get(method)
                if cfg:
                    _, _, out_prefix = cfg
                    p = os.path.join(result_dir, f'{out_prefix}_{data_name}.csv')
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
            elif c.startswith('NN_Test_'):
                row[c] = '-'  # 纯基线不计算 NN_Test
        summary.append(row)

    if summary:
        out_df = pd.DataFrame(summary)
        # 仅 EduTGT 有 NN_Test（新节点归纳测试）；纯基线不计算，已显式设为 -
        out_path = os.path.join(result_dir, f'paper_eval_summary_{data_name}.csv')
        out_df.to_csv(out_path, index=False)
        print(f"\n汇总已保存: {out_path}")
        print("说明: NN_Test=新节点归纳测试(仅 EduTGT 支持)，- 表示该方法不计算")
        print(out_df.to_string())
    else:
        print("未找到可汇总的 CSV。请先不带 --no_run 运行本脚本。")


if __name__ == '__main__':
    main()
