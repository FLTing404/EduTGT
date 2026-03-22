# Baseline 对比实验（TGN / TGAT / JODIE）

本目录与 **主项目 `main.py` 解耦**：不修改默认训练流程；数据与划分通过 `common/data_adapter.py` **复用 `utils.Dataset`**，与 `README_oulad.md` 一致（按边时间戳分位数 train/val/test，负采样为训练集 `dst` 上的 `RandEdgeSampler`）。

**与 ContraTGT（IJCAI）论文设定对齐**：并列仓库 **`ContraTGT/`** 中 `main.py` 与 **`EduTGT/EduTGT/main.py`** 使用同一套 `Dataset` 逻辑：除全量 **test** 外，另报告 **new node test（`nn_test`）**——测试时间窗内、**至少一端节点未在训练边中出现** 的边。三个 baseline 在最终评估阶段与主模型一致，同时输出 `test_*` 与 `nn_test_*`（AUC / AP / ACC）；`summary.csv` 含 `nn_test_auc`、`nn_test_ap`、`nn_test_acc`。若某 split 无样本则指标为 `nan`。

## 支持的模型

| 目录 | 模型 | 说明 |
|------|------|------|
| `tgn/` | **TGN** | 使用 PyTorch Geometric 收录的 `TGNMemory`（与 Twitter Research 开源实现一致） |
| `tgat/` | **TGAT** | 时序邻居注意力 + 链路 MLP；论文官方代码为 TensorFlow，此处为 **PyTorch 复现论文结构**（见 `tgat/README.md`） |
| `tncn/` | **JODIE 风格** | 原计划 **TNCN** 与 TGB 生态强绑定，接入成本高；采用 **JODIE**（RNN 用户/物品动态嵌入）作为第三基线，见 `tncn/README.md` |

## 依赖

主项目已有 `torch`；**TGN 额外需要**：

```bash
pip install -r baseline/requirements-baseline.txt
```

## 一键运行（三个基线顺序执行）

在 **`EduTGT/EduTGT`** 下：

```bash
python baseline/run_all_baselines.py --data_dir data/processed/data_AAA_2013J
```

常用参数：`--epochs`、`--batch_size`、`--lr`、`--seed`（默认 42）、`--device`（`cuda` / `cpu`）。  
跳过某一基线：`--skip_tgn`、`--skip_tgat`、`--skip_tncn`。

## 单模型运行

```bash
python baseline/tgn/runner.py --data_dir data/processed/data_AAA_2013J --out_dir baseline/results/my_run/tgn
python baseline/tgat/runner.py --data_dir data/processed/data_AAA_2013J --out_dir baseline/results/my_run/tgat
python baseline/tncn/runner.py --data_dir data/processed/data_AAA_2013J --out_dir baseline/results/my_run/tncn
```

## 数据格式

与主流程相同：目录内需 `ml_*.csv`、`*.content`；由 `Dataset` 完成节点映射与时间切分。`edge_attr.csv` 等若存在，adapter 会记录日志并 **不强制用于** 上述基线。

## 结果输出

`run_all_baselines.py` 写入：

```
baseline/results/<数据目录名>/
  tgn/       result.json, checkpoint.pt
  tgat/      result.json, checkpoint.pt
  tncn/      result.json, checkpoint.pt（JODIE 实现）
  summary.csv
  summary.json
```

`result.json` 含 `val_*`、`test_*`、`nn_test_*`（及 `stats.num_nn_test` 等）、`train_time_sec`、`best_epoch` 等。

## 指标

统一由 `common/metrics.py` 计算（AP、AUC 等），与基线 runner 共用。
