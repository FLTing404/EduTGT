# 下游课程通过预测（SpatialTemporal：E1 / E2）

所有脚本与产出均在 **`downstream_pass_prediction/`** 内；**只读** `data/processed/`、`data/raw/OULAD/` 与 **`outputs/pretrain/<data_name>.pth`（E2）**；不修改仓库其余目录。

## 任务

- **E1**（`--init scratch`）：`SpatialTemporal` **随机初始化** + 线性头，预测 `is_pass`。
- **E2**（`--init pretrain`）：加载 **`outputs/pretrain/<module>.pth`** 后与 E1 相同训练协议。

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

# 3a) E1 随机初始化
python downstream_pass_prediction/scripts/run_train_st_pass.py --module data_AAA --init scratch

# 3b) E2 预训练初始化（需已存在 outputs/pretrain/data_AAA.pth）
python downstream_pass_prediction/scripts/run_train_st_pass.py --module data_AAA --init pretrain
```

**权重与日志**：`downstream_pass_prediction/outputs/data_<MODULE>/st_pass_e1_scratch.pt`、`st_pass_e2_pretrain.pt`、`runs_st.jsonl`。

**指标**：`runs_st.jsonl` 中含 **`test_acc`**（默认概率 ≥ **0.5** 视为通过，可用 `--acc-threshold` 改）、**`test_auroc`**（与 ROC 图 AUC 一致）。测试集 **ROC 曲线** 保存为同目录下 **`roc_test_e1_scratch.png` / `roc_test_e2_pretrain.png`**（`--no-roc-plot` 可关闭；需安装 `matplotlib`）。

**配置**：`configs/default.yaml`（`st.*` 与 `t_star_relative_day`）。命令行可覆盖：`--epochs`（默认 50）、`--patience`（默认 10，验证 AUPRC 早停）、`--lr`、`--batch-size`、`--t-star-relative-day`。

## 标签

`is_pass` 来自 **`studentInfo.csv`** 的 `final_result`（见 `src/labels.py`）。

详细设计见 **`方案.md`**。
