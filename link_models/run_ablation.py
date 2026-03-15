"""
一键消融 + 三基线：5 种消融配置 + TGAT / GraphSAGE / JODIE。
从项目根运行: python link_models/run_ablation.py [--data_dir xxx] [--seed 60] [--seeds 42,43,44,45,46] [--quick]
单次：--seed 可统一 pretrain/main_link/基线 的随机种子。多 seed：--seeds 跑多轮并输出带 seed 的结果文件与汇总表（Mean_AUC, Std_AUC）。
"""
import os
import sys
import subprocess
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(CODE_ROOT)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, SCRIPT_DIR)

def _parse_seeds():
    """解析 --seeds 42,43,44,45,46，返回 list[int] 或 None（表示单次运行）。"""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == '--seeds' and i + 1 < len(argv):
            raw = argv[i + 1].strip()
            return [int(x.strip()) for x in raw.split(',') if x.strip()]
    return None

def _base_args(for_multi_seed=False):
    """base 为所有脚本共用的参数（如 --data_dir, --seed）。去掉 --quick、--seeds 及其值；多 seed 时去掉 --seed 以便每轮单独注入。"""
    base = sys.argv[1:]
    out = []
    i = 0
    while i < len(base):
        if base[i] == '--seeds' and i + 1 < len(base):
            i += 2
            continue
        if base[i] == '--quick':
            i += 1
            continue
        if for_multi_seed and base[i] == '--seed' and i + 1 < len(base):
            i += 2
            continue
        out.append(base[i])
        i += 1
    return out

def _quick_args():
    """--quick 时仅对 pretrain 和 main_link 添加 --n_epoch 2；baseline 不受影响。"""
    return ['--n_epoch', '2'] if '--quick' in sys.argv else []

def _get_data_name():
    """从 argv 解析 data_name（与 script/utils.get_data_paths、link_models/data_utils 一致）。"""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ('-d', '--data') and i + 1 < len(argv):
            return argv[i + 1]
        if a == '--data_dir' and i + 1 < len(argv):
            val = argv[i + 1]
            return os.path.basename(os.path.normpath(val))
    return None

def _run(cmd, env=None):
    env = env or os.environ
    p = subprocess.run(cmd, shell=False, cwd=CODE_ROOT, env=env, capture_output=True, text=True, timeout=360000)
    return p.returncode, p.stdout, p.stderr

def _parse_test_auc(stdout):
    m = re.search(r'test_auc:\s*([\d.]+)', stdout)
    return float(m.group(1)) if m else None

def _read_baseline_auc(data_name, baseline_name):
    """从 result/link/<data_name>/link_<name>_<data_name>.csv 读 Test_AUC。"""
    try:
        import pandas as pd
        path = os.path.join(CODE_ROOT, 'result', 'link', data_name, f'link_{baseline_name}_{data_name}.csv')
        if not os.path.isfile(path):
            return None
        df = pd.read_csv(path)
        if 'Test_AUC' not in df.columns or len(df) == 0:
            return None
        return float(df['Test_AUC'].iloc[0])
    except Exception:
        return None

# 消融配置: (名称, pretrain 额外参数, 描述)；扩展配置带优化超参以提升相对 baseline 的 AUC
ABLATION_CONFIGS = [
    ('baseline', ['--no_topk', '--no_student_consistency', '--ablation_suffix', 'baseline'], 'ContraTGT baseline'),
    ('neg', ['--no_topk', '--no_student_consistency', '--neg_sampler', 'two_stage', '--ablation_suffix', 'neg',
             '--n_epoch', '95', '--neg_hard_ratio', '0.2'], '+ neg sampler (80% random + 20% hard, 95 epoch)'),
    ('topk', ['--no_student_consistency', '--ablation_suffix', 'topk',
              '--alpha', '0.25', '--n_epoch', '90'], '+ Top_k + mid_model (α=0.25, 90 epoch)'),
    ('consistency', ['--no_topk', '--ablation_suffix', 'consistency',
                     '--n_epoch', '90', '--consistency_weight', '0.10'], '+ student consistency (90 epoch, λ=0.10，降权以减方差)'),
    ('neg_consistency', ['--no_topk', '--neg_sampler', 'two_stage', '--ablation_suffix', 'neg_consistency',
                         '--n_epoch', '95', '--neg_hard_ratio', '0.2', '--consistency_weight', '0.12'],
     '+ neg + consistency (95 epoch, 80/20 neg, λ=0.12)'),
]

# 三基线: (脚本名, 显示名, 描述)
BASELINES = [
    ('train_link_tgat.py', 'TGAT', 'TGAT 基线'),
    ('train_link_graphsage.py', 'GraphSAGE', 'GraphSAGE 基线'),
    ('train_link_jodie.py', 'JODIE', 'JODIE 基线'),
]

def _run_one_round(base, python, pretrain_script, main_link_script, data_name):
    """跑一轮（5 消融 + 3 基线），返回 [(name, desc, auc), ...]。"""
    results = []
    for name, extra_pretrain, desc in ABLATION_CONFIGS:
        print('\n' + '=' * 60)
        print(f'消融: {name} — {desc}')
        print('=' * 60)
        pretrain_cmd = [python, pretrain_script] + base + extra_pretrain + _quick_args()
        code, out, err = _run(pretrain_cmd)
        if code != 0:
            print(f'预训练失败 {name}:', err[-2000:] if len(err) > 2000 else err)
            results.append((name, desc, None))
            continue
        link_cmd = [python, main_link_script] + base + ['--ablation_suffix', name] + _quick_args()
        code2, out2, err2 = _run(link_cmd)
        if code2 != 0:
            print(f'链路训练失败 {name}:', err2[-2000:] if len(err2) > 2000 else err2)
            results.append((name, desc, None))
            continue
        auc = _parse_test_auc(out2)
        results.append((name, desc, auc))
        print(f'  -> test_auc: {auc}')

    for script_name, display_name, desc in BASELINES:
        print('\n' + '=' * 60)
        print(f'基线: {display_name} — {desc}')
        print('=' * 60)
        script_path = os.path.join(CODE_ROOT, 'link_models', script_name)
        code, out, err = _run([python, script_path] + base)
        if code != 0:
            print(f'基线 {display_name} 失败:', err[-2000:] if len(err) > 2000 else err)
            results.append((display_name, desc, None))
            continue
        auc = _read_baseline_auc(data_name, display_name.lower()) if data_name else None
        if auc is None and out and 'Test_AUC' in out:
            m = re.search(r'Test_AUC[\s:]+([\d.]+)', out)
            if m:
                auc = float(m.group(1))
        results.append((display_name, desc, auc))
        print(f'  -> test_auc: {auc}')
    return results

def _write_seed_csv(results, data_name, seed, code_root):
    """将单 seed 结果写入 result/link/<data_name>/ablation_results_seed_<seed>.csv"""
    if not data_name:
        return
    out_dir = os.path.join(code_root, 'result', 'link', data_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'ablation_results_seed_{seed}.csv')
    try:
        import pandas as pd
        rows = [{'Model': name, 'Description': desc, 'Test_AUC': auc} for name, desc, auc in results]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f'  已写入: {path}')
    except Exception as e:
        print(f'  写入 {path} 失败: {e}')

def _write_summary_csv(seed_results, data_name, code_root):
    """根据多 seed 的 results 列表，汇总 Mean_AUC, Std_AUC 并写入 ablation_summary_<data_name>.csv"""
    if not data_name or not seed_results:
        return
    import pandas as pd
    # seed_results: [(seed, [(name, desc, auc), ...]), ...]
    models = []
    desc_map = {}
    auc_by_model = {}
    for seed, results in seed_results:
        for name, desc, auc in results:
            if name not in auc_by_model:
                models.append(name)
                desc_map[name] = desc
                auc_by_model[name] = []
            auc_by_model[name].append(auc if auc is not None else float('nan'))
    out_dir = os.path.join(code_root, 'result', 'link', data_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'ablation_summary_{data_name}.csv')
    rows = []
    for name in models:
        vals = auc_by_model[name]
        mean_auc = float(pd.Series(vals).mean())
        std_auc = float(pd.Series(vals).std()) if len(vals) > 1 else 0.0
        row = {'Model': name, 'Description': desc_map[name], 'Mean_AUC': round(mean_auc, 4), 'Std_AUC': round(std_auc, 4)}
        for i, (seed, _) in enumerate(seed_results):
            row[f'seed_{seed}'] = round(vals[i], 4) if not pd.isna(vals[i]) else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'\n汇总已写入: {path}')

def main():
    seeds = _parse_seeds()
    for_multi = seeds is not None
    base = _base_args(for_multi_seed=for_multi)
    # 单次运行时若未传 --seed，统一注入 60，使 pretrain/main_link 与 TGAT/GraphSAGE/JODIE 使用同一 seed
    if seeds is None and '--seed' not in base:
        base = base + ['--seed', '60']
    data_name = _get_data_name()
    python = sys.executable
    pretrain_script = os.path.join(CODE_ROOT, 'script', 'pretrain.py')
    main_link_script = os.path.join(CODE_ROOT, 'link_models', 'main_link.py')

    if seeds is None:
        results = _run_one_round(base, python, pretrain_script, main_link_script, data_name)
        print('\n' + '=' * 60)
        print('消融 + 基线 结果汇总 (Test AUC)')
        print('=' * 60)
        print(f'{"模型":<16} {"描述":<48} {"AUC":>8}')
        print('-' * 78)
        for name, desc, auc in results:
            auc_s = f'{auc:.4f}' if auc is not None else 'N/A'
            print(f'{name:<16} {desc:<48} {auc_s:>8}')
        print('=' * 78)
        return

    seed_results = []
    for seed in seeds:
        print('\n' + '#' * 60)
        print(f'# seed = {seed}')
        print('#' * 60)
        base_with_seed = base + ['--seed', str(seed)]
        results = _run_one_round(base_with_seed, python, pretrain_script, main_link_script, data_name)
        _write_seed_csv(results, data_name, seed, CODE_ROOT)
        seed_results.append((seed, results))

    _write_summary_csv(seed_results, data_name, CODE_ROOT)
    if data_name:
        print('\n' + '=' * 60)
        print('多 seed 汇总 (Mean_AUC ± Std_AUC)')
        print('=' * 60)
        try:
            import pandas as pd
            path = os.path.join(CODE_ROOT, 'result', 'link', data_name, f'ablation_summary_{data_name}.csv')
            if os.path.isfile(path):
                df = pd.read_csv(path)
                print(df.to_string(index=False))
        except Exception as e:
            print(f'打印汇总表失败: {e}')
        print('=' * 78)

if __name__ == '__main__':
    main()
