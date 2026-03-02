# ContraTGT 使用方式与实验结果汇总

本文档说明本项目**如何使用 ContraTGT**，以及 **`code/result/`** 下基线模型（LSTM、XGBoost）的**结果对比**。ContraTGT 为主模型，基线用于对比或消融。

---

## 一、ContraTGT 在本项目中的使用方式

### 1.1 整体流程

```
OULAD 原始数据 (OULAD-main/data/)
        ↓
  数据转换 (convert_oulad_abc.py)
        ↓
  图数据 all_data/data_abc_* (ml_oulad.csv + oulad.content)
        ↓
  【ContraTGT】script/ 预训练 + 微调 + 导出
        ↓
  增强表 process_data/data_abc_*_enhanced/ (边表 + emb_u_* + emb_i_*)
        ↓
  基线模型 (LSTM / XGBoost) 在「原始」或「增强」数据上训练，结果写入 result/
```

### 1.2 ContraTGT 具体用法（主代码 `code/script/`）

| 步骤 | 脚本 | 作用 | 输出 |
|------|------|------|------|
| 1 | `pretrain.py` | 在时序边上做对比学习预训练 | `pretrain_model/<data_dir>.pth` |
| 2 | `main.py` | 加载预训练权重，BCE 链接预测微调，按验证集 AP 早停 | `saved_checkpoints/<data_dir>.pth`，控制台打印 **Test AP / AUC** 等 |
| 3 | `run_enhance.py` | 一键执行 1→2，再按时间顺序为每条边生成 (u,i) embedding，拼表 | `process_data/<data_dir>_enhanced/ml_oulad.csv`（含 emb_u_*, emb_i_*） |

**常用命令（均在 `code` 目录下）：**

```bash
# 一键：预训练 + 微调 + 导出增强表（你当前用法对应 data_abc_0.01）
python script/run_enhance.py --data_dir data_abc_0.01 --gpu 0

# 仅导出（已有 checkpoint 时）
python script/run_enhance.py --data_dir data_abc_0.01 --skip_finetune --gpu 0
```

- **ContraTGT 的最终评估指标**（Test AP、AUC 等）来自运行 **`script/main.py`** 时在 test / nn_test 上的输出，不会写入 `result/` 目录。
- **`result/`** 中保存的是 **LSTM、XGBoost 基线**在指定数据上的训练/验证/测试指标。

---

## 二、`code/result/` 当前文件说明

| 文件 | 含义 |
|------|------|
| `lstm_results_data_abc_0.01.csv` | LSTM 在**原始**图数据 `data_abc_0.01` 上的结果（仅节点特征 + 序列） |
| `lstm_results_data_abc_0.01_enhanced.csv` | LSTM 在 **ContraTGT 增强**数据 `data_abc_0.01_enhanced` 上的结果 |
| `xgboost_results_data_abc_0.01.csv` | XGBoost 在**原始**图数据 `data_abc_0.01` 上的结果 |
| `xgboost_results_data_abc_0.01_enhanced.csv` | XGBoost 在 **ContraTGT 增强**数据 `data_abc_0.01_enhanced` 上的结果（使用边表内 emb_u_*、emb_i_*） |
| `xgboost_results_data_abc_0.1.csv` | XGBoost 在**原始**数据 `data_abc_0.1`（10% 采样）上的结果 |

---

## 三、基线结果对比（Test 集）

### 3.1 数据 `data_abc_0.01`：原始 vs 增强

**LSTM**

| 数据 | Test AUC | Test AP | Test Acc | Test Loss |
|------|----------|---------|----------|-----------|
| 原始 `data_abc_0.01` | 0.6299 | 0.6450 | 0.5934 | 0.6695 |
| 增强 `data_abc_0.01_enhanced` | 0.6305 | 0.6419 | 0.5739 | 0.6736 |

**XGBoost**

| 数据 | Test AUC | Test AP | Test Acc | Test Loss |
|------|----------|---------|----------|-----------|
| 原始 `data_abc_0.01` | 0.6180 | 0.7496 | 0.6118 | 0.6635 |
| 增强 `data_abc_0.01_enhanced` | **0.6313** | **0.7579** | 0.5766 | 0.6549 |

- **XGBoost + 增强**：Test AUC、AP 均高于原始数据（+embedding 后模型利用了图表示）。
- **LSTM**：原始与增强在 Test 上接近，增强略升 AUC、略降 Acc，可能和序列构造/超参有关。

### 3.2 不同数据规模（XGBoost）

| 数据 | Test AUC | Test AP | Test Acc |
|------|----------|---------|----------|
| `data_abc_0.01`（原始） | 0.6180 | 0.7496 | 0.6118 |
| `data_abc_0.1`（原始，约 10 倍边） | 0.6161 | 0.7042 | 0.6194 |

---

## 四、小结

- **ContraTGT 使用方式**：用 `script/run_enhance.py --data_dir data_abc_0.01` 得到预训练+微调模型和增强表；ContraTGT 自身的 Test 指标看 `main.py` 控制台输出。
- **基线对比**：在 `data_abc_0.01` 上，**XGBoost + 增强**优于 **XGBoost + 原始**（Test AUC/AP 提升），说明 ContraTGT 导出的 embedding 对表格模型有增益；LSTM 在原始与增强上表现接近。
- **结果文件**：所有基线数值均来自 `code/result/` 下上述 CSV，指标列为 Train/Val/Test 的 AUC、AP、Acc、Loss 等。
