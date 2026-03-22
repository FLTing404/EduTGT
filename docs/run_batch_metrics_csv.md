# `run_batch_metrics_csv.py` 使用说明

本文档说明 **`EduTGT/EduTGT/run_batch_metrics_csv.py`** 的用途、前置条件、命令行参数、输出格式与常见注意事项。该脚本为**辅助批跑工具**，**不修改** `main.py`、消融与 baseline 的内部训练逻辑。

---

## 1. 脚本做什么

在 **项目根目录**（与 `main.py` 同级）下，对选定的 **整模块数据集 `AAA` 或 `BBB`** 做一轮批量实验：

对每个 **seed**（默认三个：`42,1,2`），按固定顺序依次执行：

| 顺序 | `method`（CSV 中名称） | 实际命令 |
|------|------------------------|----------|
| 1 | `main` | `main.py` |
| 2 | `ablation_pres` | `main_ablation_pres_relation.py` |
| 3 | `ablation_stc` | `main_ablation_student_temporal.py` |
| 4 | `tgn` | `baseline/tgn/runner.py` |
| 5 | `tgat` | `baseline/tgat/runner.py` |
| 6 | `jodie` | `baseline/tncn/runner.py`（JODIE 风格，目录名为 `tncn`） |

- **不会**调用 `pretrain.py`。需你事先完成预训练，并存在 **`outputs/pretrain/<data_name>.pth`**（例如 `data_AAA` → `outputs/pretrain/data_AAA.pth`）。
- 所有子进程 **`cwd`** 为项目根，并设置 **`PYTHONPATH`** 指向该根目录，保证 `import utils`、`import paths` 等与手动运行一致。
- **数据目录**固定为：`data/processed/data_AAA` 或 `data/processed/data_BBB`（由 `--dataset` 决定）。若目录不存在，脚本直接退出并报错。

---

## 2. 前置条件检查清单

1. **工作目录**：在 **`EduTGT/EduTGT`** 下执行（与文档 [使用指南.md](./使用指南.md) 一致）。
2. **数据**：已存在 `data/processed/data_AAA/` 或 `data/processed/data_BBB/`，且含 `ml_*.csv`、`.content` 等（见 [README_oulad.md](./README_oulad.md)）。
3. **预训练权重**：`outputs/pretrain/data_AAA.pth` 或 `data_BBB.pth`。缺失时脚本会打印 **警告**，随后 `main`/消融通常会报错退出。
4. **presentation 消融**：`ablation_pres` 需要数据目录下存在 **`pres_relation.json`**。若未生成，请先执行：  
   `python data/scripts/build_pres_relation.py --data_dir data/processed/data_AAA`（路径按你的数据集改）。
5. **TGN**：需安装 PyG，例如：  
   `pip install -r baseline/requirements-baseline.txt`
6. **GPU**：`main`/消融使用 `--gpu`；baseline 使用 `--device cuda` 或 `cpu`。无 CUDA 时请 `--device cpu`，避免 baseline 子进程失败。

---

## 3. 命令行参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | **必填** | 仅支持 **`AAA`** 或 **`BBB`**，对应 `data/processed/data_AAA`、`data_BBB`。 |
| `--seeds` | `42,1,2` | 逗号分隔的整数列表，**不要**加空格（或仅数字间逗号）。 |
| `--n_epoch` | `50` | 传给 `main.py` 与两个消融的 `--n_epoch`。 |
| `--bs` | `800` | 传给上述三者的 `--bs`。 |
| `--lr` | `0.001` | 传给上述三者及 baseline 的 `--lr`。 |
| `--ctx_sample` | `40` | 空间采样长度（main/消融）。 |
| `--tmp_sample` | `31` | 时间采样长度（main/消融）。 |
| `--drop_out` | `0.2` | Dropout（main/消融）。 |
| `--gpu` | `0` | `main`/消融 的 `--gpu`（CUDA 设备号）。 |
| `--lambda_tc` | `0.1` | **仅** `ablation_stc`：放在子进程参数最前，与 [使用指南.md](./使用指南.md) 一致。 |
| `--pres_mix_uniform` | `0.35` | **仅** `ablation_pres`。 |
| `--baseline_epochs` | `50` | 三个 baseline 的 `--epochs`。 |
| `--baseline_bs` | `200` | 三个 baseline 的 `--batch_size`（与 main 默认 800 区分开，避免 OOM）。 |
| `--device` | `cuda` | baseline 的 `--device`：`cuda` 或 `cpu`。 |
| `--out_csv` | 无 | 指定汇总 CSV 路径；**默认** `outputs/result/batch_<DATASET>_<YYYYMMDD_HHMMSS>.csv`。相对路径相对于项目根解析。 |

查看内置帮助：

```bash
python run_batch_metrics_csv.py -h
```

---

## 4. 使用示例

```bash
cd EduTGT/EduTGT

# AAA，默认 seeds=42,1,2，默认超参
python run_batch_metrics_csv.py --dataset AAA

# BBB，自定义种子与 CPU 跑 baseline
python run_batch_metrics_csv.py --dataset BBB --seeds 42,2025,7 --device cpu

# 缩短 baseline 轮数、指定 CSV 输出位置
python run_batch_metrics_csv.py --dataset AAA --baseline_epochs 20 --out_csv outputs/result/my_sweep.csv
```

**Windows PowerShell** 路径同理；若一行多条命令，用 `;` 分隔。

---

## 5. 输出文件布局

### 5.1 汇总 CSV（主要结果）

- **默认路径**：`outputs/result/batch_AAA_20250322_143052.csv`（时间戳随运行变化）。
- **编码**：UTF-8，带表头。
- **列说明**：

| 列名 | 含义 |
|------|------|
| `dataset` | 命令行中的 `AAA` 或 `BBB`（不是 `data_AAA` 字符串）。 |
| `method` | `main` / `ablation_pres` / `ablation_stc` / `tgn` / `tgat` / `jodie`。 |
| `seed` | 本次子进程传入的 `--seed`，与 main/消融的 `init_seeds(args.seed)` 及 baseline 的 `set_seed` 一致（见第 7 节）。 |
| `AUC` | 全量 **test** 集 AUC（main/消融来自 `training_runs.jsonl`；baseline 来自 `result.json` 的 `test_auc`）。 |
| `ACC` | test ACC。 |
| `AP` | test AP。 |
| `Loss` | test BCE loss（**仅 main/消融** 有；baseline 脚本不统计该项，列为空）。 |
| `epoch` | 早停记录的最佳 epoch（`best_epoch`）。 |
| `time_cost` | 该子进程**墙钟时间**（秒，保留 3 位小数），由批跑脚本计时，**不是** JSON 里的字段。 |
| `nn_test_auc` | 新节点子集 test 的 AUC。 |
| `nn_test_ap` | 同上，AP。 |
| `nn_test_acc` | 同上，ACC。 |
| `status` | `ok` 表示子进程退出码 0 且成功解析指标；否则为简短错误码（如 `exit_1`、`no_jsonl_record`、`no_result_json`）。 |

### 5.2 中间产物目录（baseline）

为避免覆盖 `baseline/results/` 下其它实验，每次 baseline 写入：

`outputs/result/_runs/<data_name>_s<seed>/<tgn|tgat|jodie>/`

内含该次运行的 `result.json`、`checkpoint.pt` 等（由各 baseline `runner.py` 原有逻辑写出）。

### 5.3 main / 消融 的日志

成功时仍会向 **`outputs/logs/training_runs.jsonl`** **追加**一行（与单独跑 `main.py` 行为一致）。本脚本通过「运行前记录文件字节偏移 → 运行后读取新增内容中的**最后一行** JSON」来取数；因此**请避免**在批跑进行时**并行**其它也向该 jsonl 追加的程序，否则可能错配记录。

---

## 6. 运行规模与耗时

- 每个 seed：**6** 次子进程；默认 3 个 seed → **共 18 次**。
- 总时间 ≈ 各次 `time_cost` 之和；完整 epoch 下可能长达数小时，建议先用小 `--n_epoch` / `--baseline_epochs` 试跑。

---

## 7. 关于 `seed` 的重要说明（必读）

- **`main.py`**、**`main_ablation_pres_relation.py`**、**`main_ablation_student_temporal.py`** 在 **进入训练循环前** 与 **`eval_epoch` 内** 调用 **`init_seeds(args.seed)`**，与命令行 **`--seed`**（`get_args()` 默认 `60`）一致。
- 因此：对这三个任务，**不同 `--seed` 会改变训练与验证/测试阶段的随机性**；`run_batch_metrics_csv.py` 传入的 seed 与 CSV 中 **`seed` 列** 含义一致。
- **`tgn` / `tgat` / `jodie`** 的 runner **同样使用** `--seed`，不同 seed 结果一般不同。

---

## 8. 故障排查

| 现象 | 可能原因 |
|------|----------|
| `status=exit_1`（或非 0） | 缺预训练权重、缺 `pres_relation.json`、CUDA OOM、数据路径错误等；查看终端子进程完整报错。 |
| `no_jsonl_record` | `main`/消融跑完但未向 jsonl 写入（异常提前退出）或文件被截断/权限问题。 |
| `no_result_json` | baseline 子进程未在约定 `--out_dir` 生成 `result.json`。 |
| TGN 导入错误 | 未安装 `torch-geometric`，见第 2 节。 |
| 指标为空但 `status=ok` | 极少见（JSON 字段缺失）；可手动打开对应 jsonl 行或 `result.json` 核对。 |

---

## 9. 与仓库其它文档的关系

- 训练流程、目录结构（`data/processed`、`outputs/`）：[使用指南.md](./使用指南.md)、[README.md](./README.md)。  
- 方法说明：[方法与ContraTGT对比.md](./方法与ContraTGT对比.md)。  
- `training_runs.jsonl` 字段设计：[training_result_log.md](./training_result_log.md)。

---

## 10. 版本与维护

- 脚本内逻辑以仓库中 **`run_batch_metrics_csv.py` 源码为准**；若未来修改子进程列表或解析方式，请同步更新本节与参数表。
