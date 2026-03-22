# 下游课程通过预测（SpatialTemporal：E1 / E2）

所有脚本与产出均在 **`downstream_pass_prediction/`** 内；**只读** `data/processed/`、`data/raw/OULAD/` 与 **`outputs/pretrain/<data_name>.pth`（E2）**；不修改仓库其余目录。

## 任务

- **E1**（`--init scratch`）：`SpatialTemporal` **随机初始化** + 线性头；**不加载** `outputs/pretrain/*.pth`，邻居采样 **`ctx_sample` / `tmp_sample` 仅来自配置**（默认与 `main` 量级一致），与上游预训练实验无耦合。
- **E2**（`--init pretrain`）：**加载** `outputs/pretrain/<module>.pth`；固定 `ctx_sample=30`、`tmp_sample=21` 与预训练几何一致。为防止随机初始化的分类头在初期产生大梯度破坏预训练编码器（灾难性遗忘），采用**两阶段微调**：① 先冻结编码器若干轮**只训练头部**（`--freeze-epochs`，默认 5）；② 再以**差分学习率**全量微调——编码器用基础 `lr`，头部用 `lr × head_lr-factor`（默认 10.0）。

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

# 3a) E1 随机初始化（ctx/tmp 来自 configs/default.yaml，与 main 量级一致）
python downstream_pass_prediction/scripts/run_train_st_pass.py --module data_AAA --init scratch

# 3b) E2：在同一 processed 目录上预训练（若已有 outputs/pretrain/data_AAA.pth 可跳过本行）
python pretrain.py --data_dir data/processed/data_AAA --n_epoch 50 --bs 800
# 学习率可略低于 E1，减轻破坏已学 backbone（默认 lr 仍来自 default.yaml）
python downstream_pass_prediction/scripts/run_train_st_pass.py --module data_AAA --init pretrain --lr 5e-5
```

**权重与日志**：`downstream_pass_prediction/outputs/data_<MODULE>/st_pass_e1_scratch.pt`、`st_pass_e2_pretrain.pt`、`runs_st.jsonl`。

**指标**：`runs_st.jsonl` 每条含 **`test_acc`**、**`test_auroc`**、**`test_auprc`**、**`test_f1`**（F1 与 ACC 共用 `--acc-threshold` 得预测类别）、**`wall_time_sec`**（整次训练+评测墙钟秒）。**`test_auroc`** 与 ROC 图 AUC 一致。ROC 保存为 **`roc_test_e1_scratch.png` / `roc_test_e2_pretrain.png`**（`--no-roc-plot` 可关）。批处理可用 **`--dump-run-json <路径>`** 将本条指标写入单文件 JSON。

## 一键批跑 AAA / BBB / CCC（E1+E2 × 各 3 随机 seed）

在 **`EduTGT/EduTGT`** 下执行；每个模块抽取 **3 个（或 `--seeds-per-cell`）互不重复随机 seed**，**E1 与 E2 共用同一组 seed**：对每个 seed 先 **`build_student_splits`**，再依次跑 **scratch** 与 **pretrain**（中间不重建划分，保证配对可比）。汇总 **`UTF-8 BOM` CSV** 与 **`_seeds.json`** 写入 **`downstream_pass_prediction/outputs/`**。

```bash
# 默认：data_AAA + data_BBB + data_CCC，各 E1/E2 × 3 seed；E2 缺 pth 时跳过该行（status=no_pretrain_ckpt）
python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py

python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py --device cuda:0 --no-roc-plot --e2-lr 5e-5

# 若要求三个模块都必须有预训练权重，否则直接退出：
python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py --strict-pretrain
```

CSV 列：`dataset, init, seed, wall_time_sec, test_acc, test_auroc, test_auprc, test_f1, epochs_ran, ctx_sample, tmp_sample, status, error`。

**配置**：`configs/default.yaml` 的 `st.*`、`t_star_relative_day`。**E1** 使用 `ctx_sample` / `tmp_sample`；**E2** 写死 30/21。命令行覆盖：`--epochs`、`--patience`、`--lr`、`--batch-size`、`--t-star-relative-day`；E2 专用：`--freeze-epochs`（默认 5）、`--head-lr-factor`（默认 10.0）。

## 标签

`is_pass` 来自 **`studentInfo.csv`** 的 `final_result`（见 `src/labels.py`）。

详细设计见 **`方案.md`**。
