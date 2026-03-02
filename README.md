# 基于 ContraTGT 的教育学数据预测

本项目在 **ContraTGT**（时序图对比学习）基础上，面向 **OULAD** 教育学数据做学生–课程链接预测（如是否通过）。**主代码位于 `code/script/`**，包含预训练、微调与增强表导出；根目录下提供数据转换脚本与 LSTM/XGBoost 基线。所有命令均在 **`code`** 目录下执行。

---

## 如何使用（快速上手）

1. **进入目录**：`cd code`
2. **准备并转换数据**：将 OULAD 原始 CSV 放到 `OULAD-main/data/`，然后运行  
   `python convert_oulad_abc.py --sample_ratio 0.1`（或 `convert_oulad.py`），得到 `all_data/data_abc_0.1/` 等。
3. **跑主模型（script，需 GPU）**：
   - 一键：`python script/run_enhance.py --data_dir data_abc_0.1 --gpu 0`  
   - 或分步：`python script/pretrain.py --data_dir data_abc_0.1 --gpu 0` → `python script/main.py --data_dir data_abc_0.1 --gpu 0`  
   - 增强表输出在 `process_data/<data_dir>_enhanced/`。
4. **跑基线（可选）**：`python models/train_lstm.py --data_dir data_abc_0.1`、`python models/train_xgboost.py --data_dir data_abc_0.1`，结果在 `result/`。

更多参数见下文；各脚本支持 `--help`。

---

## 一、目录结构

```
code/
├── README.md                   # 本说明
├── script/                     # 【主代码】基于 ContraTGT 的图模型
│   ├── pretrain.py            # 对比学习预训练
│   ├── main.py                # 有监督链接预测微调与评估
│   ├── run_enhance.py         # 一键微调 + 导出 embedding 增强表
│   ├── model.py               # SpatialTemporal 双塔 Transformer
│   ├── graph_transformer.py   # Transformer 编码器
│   ├── utils.py               # 参数、路径、Dataset、早停等
│   └── sampling.py            # 时空邻居与序列采样
├── convert_oulad.py            # 数据转换：全量/按比例采样
├── convert_oulad_abc.py        # 数据转换：按模块筛选（如 AAA,BBB,CCC）
├── OULAD-main/data/            # 原始 OULAD CSV
├── all_data/                   # 转换后的图数据（按 data_dir 分子目录）
├── pretrain_model/             # 预训练权重（script 输出）
├── saved_checkpoints/          # 微调最佳权重（script 输出）
├── process_data/               # 增强表（run_enhance 输出）
├── models/                     # 基线：LSTM、XGBoost
└── result/                     # 基线结果 CSV
```

---

## 二、主代码（script/）

基于 **ContraTGT** 的时序图表示学习：在 OULAD 学生–课程交互边上做**对比学习预训练** → **链接预测微调** → 可选 **embedding 拼表** 得到增强数据。数据从 **`all_data/<data_dir>/`** 读取（`ml_oulad.csv` + `oulad.content`），通过 **`--data_dir`** 指定目录名。

### 2.1 pretrain.py（对比学习预训练）

- 在时序边上做对比学习，为微调提供初始化。
- 输出：`pretrain_model/<data_dir>.pth`，早停时也会写入 `saved_checkpoints/<data_dir>.pth`。

### 2.2 main.py（有监督微调 + 评估）

- 加载预训练权重，BCE 链接预测微调，按验证集 AP 早停（max_round=8）。
- 最佳模型：`saved_checkpoints/<data_dir>.pth`；在 test / nn_test 上评估 AP、AUC。

### 2.3 run_enhance.py（一键微调 + 导出 / 仅导出）

- **一键模式**：pretrain → main → 用最佳 checkpoint 按时间顺序生成 (u,i) embedding，拼成 [h_u ‖ h_i] 写入 `process_data/<data_dir>_enhanced/`。
- **仅导出**（`--skip_finetune`）：不训练，直接加载已有 checkpoint 生成增强表。

### 2.4 常用命令（在 code 目录下）

```bash
cd code
# 一键微调并导出增强表
python script/run_enhance.py --data_dir data_abc_0.01 --gpu 0
# 分步
python script/pretrain.py --data_dir data_abc_0.01 --gpu 0
python script/main.py --data_dir data_abc_0.01 --gpu 0
# 仅导出（需已有 saved_checkpoints）
python script/run_enhance.py --data_dir data_abc_0.01 --skip_finetune --gpu 0
```

数据根目录非 `code/all_data` 时可加 `--all_data_root <路径>`。

---

## 三、数据转换（OULAD → 四文件格式）

### 前置条件

- 原始 OULAD 数据放在 **`code/OULAD-main/data/`**，需包含：
  - `studentVle.csv`、`studentInfo.csv`、`courses.csv`、`assessments.csv`
  - `studentAssessment.csv`、`studentRegistration.csv`、`vle.csv`

### 3.1 convert_oulad.py（全量或按比例采样）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--sample_ratio` | float | 1.0 | 边数据采样比例 (0.0–1.0) |
| `--max_edges` | int | None | 最大边数上限（可选） |

- 输出目录：`sample_ratio=1.0` → **`all_data/data_1/`**；`0.1` → **`all_data/data_0.1/`**；其它 → **`all_data/data_<sample_ratio>/`**

```bash
cd code
python convert_oulad.py
python convert_oulad.py --sample_ratio 0.1
python convert_oulad.py --sample_ratio 0.1 --max_edges 500000
```

### 3.2 convert_oulad_abc.py（按模块筛选）

仅保留指定模块（如 AAA、BBB、CCC），数据量更小，适合快速实验。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--modules` | str | AAA,BBB,CCC | 保留的模块，逗号分隔 |
| `--sample_ratio` | float | 1.0 | 边数据采样比例 (0.0–1.0) |
| `--max_edges` | int | None | 最大边数上限（可选） |

- 输出目录：`sample_ratio=1.0` → **`all_data/data_abc_1/`**；`0.1` → **`all_data/data_abc_0.1/`**；其它 → **`all_data/data_abc_<sample_ratio>/`**

```bash
cd code
python convert_oulad_abc.py
python convert_oulad_abc.py --sample_ratio 0.1
python convert_oulad_abc.py --modules AAA,BBB --sample_ratio 0.5
```

### 3.3 每个输出目录下的四个文件

| 文件 | 说明 |
|------|------|
| **ml_oulad.csv** | 边表：id, u, i, ts, label, idx（图模型主表） |
| **oulad.content** | 节点特征矩阵，行号对应节点 ID |
| **ml_oulad_edge_feature.csv** | 边特征，与边表行对齐 |
| **ml_oulad_pairs.csv** | 学生–课程对聚合表 |

**script/** 仅使用 **ml_oulad.csv** 和 **oulad.content**；后两个供边特征或下游表格模型使用。

---

## 四、基线模型（models/）

从 **`code/all_data/<data_dir>/`** 或 **`code/process_data/<data_dir>/`**（如增强数据 `data_abc_0.01_enhanced`）读取；通过 **`--data_dir`** 指定。目录内需包含 `ml_oulad.csv` 和 `oulad.content`。可用 **`--list_data`** 列出当前可用数据目录。

### 4.1 train_lstm.py（序列 LSTM）

- 按学生–课程对将边表转为时间序列，LSTM 编码后二分类（是否通过）。
- 输出：**`code/result/lstm_results_<data_dir>.csv`**（或通过 `--output_dir` 指定）。

常用参数示例：`--data_dir`、`--test_size`/`--val_size`、`--sample_ratio`、`--max_seq_len`、`--hidden_dim`、`--num_epochs`、`--device`。

```bash
cd code
python models/train_lstm.py --data_dir data_0.1
python models/train_lstm.py --data_dir data_abc_0.01_enhanced --max_seq_len 30 --num_epochs 30
python models/train_lstm.py --list_data
```

### 4.2 train_xgboost.py（边级 XGBoost）

- 每条边一个样本，特征为学生特征 + 课程特征 + 时间戳 + 统计特征；按 (u,i) 划分 train/val/test 避免泄露。
- 输出：**`code/result/xgboost_results_<data_dir>.csv`**。

```bash
cd code
python models/train_xgboost.py --data_dir data_0.1
python models/train_xgboost.py --data_dir data_abc_0.01_enhanced
python models/train_xgboost.py --list_data
```

---

## 五、整体流程建议

1. **准备数据**：将 OULAD 原始 CSV 放入 `OULAD-main/data/`。
2. **转换**：运行 `convert_oulad.py` 或 `convert_oulad_abc.py`，得到 `all_data/data_*` 或 `all_data/data_abc_*`。
3. **主模型（script）**：对某个 `data_dir` 运行 `script/run_enhance.py` 一键完成预训练、微调与增强表导出，或分步执行 `script/pretrain.py` → `script/main.py`。
4. **基线对比（可选）**：用 `models/train_lstm.py`、`models/train_xgboost.py` 在同一或增强数据上跑实验，结果在 `result/`，可与主模型评估对比。

更多参数见各脚本 `--help`。
