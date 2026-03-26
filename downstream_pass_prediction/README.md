# 下游课程通过预测（SpatialTemporal：E1 / E2）

所有脚本与产出均在 **`downstream_pass_prediction/`** 内；**只读** `data/processed/`、`data/raw/OULAD/` 与 **`outputs/pretrain/<data_name>.pth`（E2）**；不修改仓库其余目录。

## 任务

- **E2**（`--init pretrain`）：**加载** `outputs/pretrain/<module>.pth`；固定 `ctx_sample=30`、`tmp_sample=21` 与预训练几何一致。
- **E1**（`--init scratch`）：为保证 E1/E2 公平对比，scratch 也使用与 E2 相同的采样窗口 `ctx_sample=30`、`tmp_sample=21`（不再使用 `configs/default.yaml` 里的 `st.ctx_sample/st.tmp_sample`）。
- 为防止灾难性遗忘，采用**两阶段微调**：① 冻结编码器、**只训练头部**（轮数由 `st.e2_freeze_epochs` 控制，默认 **10**，可用 `--freeze-epochs` 覆盖）；② 解冻后以**差分学习率**微调——编码器用 `lr × st.e2_encoder_lr_scale`（默认 **0.5**），头部用 `lr × st.e2_head_lr_factor`（默认 **5.0**）。E2 默认 `lr` 取 **`st.e2_lr`（5e-5）**，未传 `--lr` 时生效；早停耐心可用 **`st.e2_patience`（默认 15）**。

锚点边：每个 `(学生, presentation)` 样本取该生在对应 presentation 下（可选 `t*` 截断）**时间上最后一条** `ml` 边，用其 **源端时空上下文** 经 `getEmbed` 得到表示，再接分类头。

## 依赖

在 `EduTGT/EduTGT`（与 `main.py` 同级）执行：

```bash
pip install -r downstream_pass_prediction/requirements-downstream.txt
```

## 流程

```bash
# 1) 标签 + manifest
python downstream_pass_prediction/scripts/build_pass_labels.py --module data_AAA

# 2) 按学生划分
python downstream_pass_prediction/scripts/build_student_splits.py --module data_AAA --seed 60

# 3a) E1 随机初始化（ctx/tmp 与 E2 一致：30/21）
python downstream_pass_prediction/scripts/run_train_st_pass.py --module data_AAA --init scratch

# 3b) E2：在同一 processed 目录上预训练（若已有 outputs/pretrain/data_AAA.pth 可跳过本行）
python pretrain.py --data_dir data/processed/data_AAA --n_epoch 50 --bs 800
# E2 默认已用 st.e2_lr=5e-5；若仍不如 E1，可再试 --lr 3e-5 或增大 --freeze-epochs
python downstream_pass_prediction/scripts/run_train_st_pass.py --module data_AAA --init pretrain
```

**权重与日志**：`downstream_pass_prediction/outputs/data_<MODULE>/st_pass_e1_scratch.pt`、`st_pass_e2_pretrain.pt`、`runs_st.jsonl`。

**指标**：`runs_st.jsonl` 每条含 **`test_acc`**、**`test_auroc`**、**`test_auprc`**、**`test_f1`**（F1 与 ACC 共用 `--acc-threshold` 得预测类别）、**`wall_time_sec`**（整次训练+评测墙钟秒）。**`test_auroc`** 与 ROC 图 AUC 一致。ROC 保存为 **`roc_test_e1_scratch.png` / `roc_test_e2_pretrain.png`**（`--no-roc-plot` 可关）。批处理可用 **`--dump-run-json <路径>`** 将本条指标写入单文件 JSON。

## 一键批跑 AAA / BBB / CCC（E1+E2 × 各 3 随机 seed）

在 **`EduTGT/EduTGT`** 下执行；每个模块抽取 **3 个（或 `--seeds-per-cell`）互不重复随机 seed**，**E1 与 E2 共用同一组 seed**：对每个 seed 先 `build_student_splits`，再跑 **1 次 E1(scratch)**，再跑 **1 次 E2(pretrain)**（默认 E2 为脚本内置单组；如需多组请传 `--e2-presets-json`）。汇总 **`UTF-8 BOM` CSV**、**`_seeds.json`**、**`_e2_presets.json`**（本次实际使用的 E2 列表）到 **`downstream_pass_prediction/outputs/`**。

### 常用命令（含解释）

```bash
# 1) 默认批跑 AAA+BBB+CCC：每个 seed 跑 1 次 E1 + 1 次 E2（单组超参）
python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py
```

说明：
- 不传参数时 `--modules` 默认是 `data_AAA,data_BBB,data_CCC`。
- `--seeds-per-cell` 默认是 `3`，即每个数据集随机取 3 个 seed。
- E2 默认使用脚本内置单组预设（不是从 yaml 动态枚举）。

```bash
# 2) 只跑 AAA，使用 GPU 0，并关闭 ROC 画图（更快）
python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py --modules data_AAA --device cuda:0 --no-roc-plot
```

说明：
- `--modules data_AAA`：只对 AAA 执行批跑。
- `--device cuda:0`：训练脚本传给 `torch.device`，改成 `cpu` 可走 CPU。
- `--no-roc-plot`：不保存 ROC PNG，减少 I/O 和画图开销。

```bash
# 3) 只跑 AAA：使用默认 E2 单组预设
python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py --modules data_AAA
```

说明：
- 若你想显式覆盖默认 E2 单组参数，可使用 `--e2-once` + 对应的 `--e2-lr` / `--e2-freeze-epochs` / `--e2-head-lr-factor` / `--e2-encoder-lr-scale` / `--e2-patience`。
- 未显式传的项会回退到 `configs/default.yaml` 的 `st.*` 默认值（仅影响 E2 的学习率/解冻/早停等）。

```bash
# 4) 只跑 AAA，并使用你自定义的“多组 E2 超参列表”
python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py --modules data_AAA --e2-presets-json d:/study/2026-大三下/EduTGT/EduTGT/downstream_pass_prediction/my_e2.json --device cuda:0
```

说明：
- `--e2-presets-json`：传入 JSON 数组，每个元素是一组 E2 配置。
- 每个元素可写字段：`preset`（或 `name`）、`lr`、`freeze_epochs`、`head_lr_factor`、`encoder_lr_scale`、`patience`。
- 该参数与 `--e2-once` 互斥；二者只能选一个。

```bash
# 5) 严格模式：只要某模块缺失预训练权重就直接失败退出
python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py --strict-pretrain
```

说明：
- 默认行为是：缺少 `outputs/pretrain/<module>.pth` 时，E2 记为 `no_pretrain_ckpt` 并跳过。
- 加 `--strict-pretrain` 后改为立即退出，适合正式批量实验前检查环境完整性。

### `--e2-presets-json` 文件示例

```json
[
  {
    "preset": "lr5e-5_f10_h5_enc0.5_p15",
    "lr": 5e-5,
    "freeze_epochs": 10,
    "head_lr_factor": 5.0,
    "encoder_lr_scale": 0.5,
    "patience": 15
  },
  {
    "preset": "lr3e-5_f15_h3_enc0.25_p20",
    "lr": 3e-5,
    "freeze_epochs": 15,
    "head_lr_factor": 3.0,
    "encoder_lr_scale": 0.25,
    "patience": 20
  }
]
```

字段解释：
- `preset`：该组配置名称，会写入结果 CSV 的 `e2_preset` 列。
- `lr`：E2 微调基础学习率。
- `freeze_epochs`：前 N 轮冻结编码器，仅训练分类头。
- `head_lr_factor`：解冻后分类头学习率倍率（头部 lr = `lr * head_lr_factor`）。
- `encoder_lr_scale`：解冻后编码器学习率倍率（编码器 lr = `lr * encoder_lr_scale`）。
- `patience`：E2 早停耐心（验证 AUPRC 连续无提升的容忍轮数）。

CSV 列含 **`e2_preset`**（E2 配置名；scratch 为空）、**`lr` / `encoder_lr_scale` / `patience`**（与 `freeze_epochs`、`head_lr_factor` 一并便于筛选最优行）。

**配置补充**：`configs/default.yaml` 含 `st.*` 与 `t_star_relative_day`。为了保证公平对比，**E1** 与 **E2** 都使用 `ctx_sample=30`、`tmp_sample=21`。单跑训练脚本时可用 `--encoder-lr-scale`、`--e2-patience` 覆盖 yaml；批跑中 E2 超参来自脚本内置单组或你提供的 `--e2-presets-json`。

## 标签

`is_pass` 来自 **`studentInfo.csv`** 的 `final_result`（见 `src/labels.py`）。

详细设计见 **`方案.md`**。
