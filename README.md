# Code 使用说明

本仓库将 OULAD 开放教育数据整理成时序图（学生–课程边 + 节点特征），并完成：**图模型对课程通过/不通过的预测**（script：ContraTGT 风格预训练 + pass/fail 主训练，输出**课程级** AUC/AP/Acc）与 **基线分类**（models 中的 XGBoost / LSTM / GNN / Node2Vec+MLP）。除 `ContraTGT/` 子目录外，所有脚本与数据流均在本说明中统一描述。

---

## 一、目录结构

```
code/
├── README.md                 # 本说明（不含 ContraTGT 内部文档）
├── convert_oulad.py          # OULAD 全量 → ContraTGT 格式
├── convert_oulad_abc.py      # OULAD 按模块(AAA/BBB/CCC)筛选 → 小规模数据
├── data/
│   ├── OULAD-main/           # 原始 OULAD 数据（需自行放置）
│   │   └── data/             # studentVle.csv 等 7 个 CSV
│   └── all_data/             # 转换后的统一数据目录
│       ├── data_1/           # 全量或按采样比例，如 data_0.1、data_1
│       │   ├── ml_oulad.csv
│       │   ├── oulad.content
│       │   ├── ml_oulad_edge_feature.csv
│       │   └── ml_oulad_pairs.csv
│       └── data_abc_0.01/    # 仅 AAA/BBB/CCC 模块 + 采样
│           └── （同上）
├── script/                   # 图模型：预训练 + 链接预测
│   ├── main.py               # 主训练（加载预训练权重）
│   ├── pretrain.py           # 预训练
│   ├── utils.py              # 参数、数据加载、EarlyStopping 等
│   ├── sampling.py           # 时空邻居采样
│   ├── model.py              # SpatialTemporal 等
│   ├── graph_transformer.py
│   ├── 性能瓶颈与优化方案.md
│   ├── pretrain_model/       # 预训练权重（按 data_name 存）
│   ├── middle_model/
│   ├── saved_checkpoints/
│   └── saved_models/
├── models/                   # 基线模型
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── train_gnn.py          # GCN 链接预测
│   └── train_node2vec.py     # Node2Vec 嵌入 + MLP 链接预测
└── result/                   # 基线 + 图模型主训练结果，按数据目录分文件夹
    └── data_abc_0.01/
        ├── script_results_data_abc_0.01.csv   # 图模型 main.py 输出
        ├── xgboost_results_data_abc_0.01.csv
        ├── lstm_results_data_abc_0.01.csv
        ├── gnn_results_data_abc_0.01.csv
        └── node2vec_results_data_abc_0.01.csv
```

---

## 二、数据流程

1. **原始数据**：将 [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) 解压到 `code/data/OULAD-main/`，保证存在 `data/` 子目录及 7 个 CSV（如 `studentVle.csv`、`studentInfo.csv` 等）。
2. **转换**：
   - **全量或按比例采样**：`convert_oulad.py` → 输出到 `data/all_data/data_{比例}/`（如 `data_1`、`data_0.1`）。
   - **仅 AAA/BBB/CCC 模块**：`convert_oulad_abc.py` → 输出到 `data/all_data/data_abc_{比例}/`（如 `data_abc_0.01`）。
3. **统一格式**：每个子目录内含 `ml_oulad.csv`（边表：u, i, ts, label, idx 等）、`oulad.content`（节点特征，每行一个节点），供 script 与 models 共用。

---

## 三、运行方式

**建议在 `code` 目录下执行以下命令。**

### 1. 数据转换

```bash
# 全量 → data/all_data/data_1/
python convert_oulad.py

# 全量 10% 采样 → data/all_data/data_0.1/
python convert_oulad.py --sample_ratio 0.1

# 仅 AAA,BBB,CCC 模块，1% 采样 → data/all_data/data_abc_0.01/
python convert_oulad_abc.py --modules AAA,BBB,CCC --sample_ratio 0.01
```

### 2. 图模型（script）：预训练 → 主训练

数据来自 `code/data/all_data/<data_dir>/`，通过 `--data_dir` 指定子目录名（如 `data_abc_0.01`）。过程文件都落在 `script/` 下（`pretrain_model/`、`middle_model/`、`saved_checkpoints/`、`saved_models/`）。**主训练结束后**会在 `code/result/<data_dir>/` 下生成 `script_results_<data_dir>.csv`，与基线同目录、**同语义**：均为**课程级** Train/Val/Test AUC、AP、Acc、Loss（script 对同一 (u,i) 的多条边做时间加权聚合后再算指标）。

```bash
# 预训练（必须先跑，按时间划分的边做 link 对比学习）
python script/pretrain.py --data_dir data_abc_0.01

# 主训练（pass/fail 监督，课程级评估；依赖同名的预训练权重）
python script/main.py --data_dir data_abc_0.01

# 与基线可比：按 (u,i) 划分 train/val/test（默认按时间划分）
python script/main.py --data_dir data_abc_0.01 --split_by_ui
```

可选参数：`--ctx_sample`、`--tmp_sample`、`--n_epoch`、`--bs`、`--gpu`、`--drop_out`、`--split_by_ui`（按 (u,i) 划分）、`--seed` 等（见 `script/utils.py`）。

### 3. 基线模型（models）

数据同样从 `code/data/all_data/` 下读取，通过 `--data_dir` 指定子目录。结果写入 `code/result/<data_dir>/`。

```bash
# 列出当前可用的数据目录
python models/train_xgboost.py --list_data
python models/train_lstm.py --list_data
python models/train_gnn.py --list_data
python models/train_node2vec.py --list_data

# 指定数据目录并训练
python models/train_xgboost.py --data_dir data_abc_0.01
python models/train_lstm.py --data_dir data_abc_0.01
python models/train_gnn.py --data_dir data_abc_0.01
python models/train_node2vec.py --data_dir data_abc_0.01
```

各脚本支持 `--test_size`、`--val_size`、`--output_dir` 等，详见各文件内 `argparse`。

---

## 四、模块说明

| 模块 | 作用 |
|------|------|
| **convert_oulad.py** | 读 OULAD 全部 7 张表，建学生/课程节点与边（含时间、标签），生成节点特征（学生画像 + 课程统计等），输出 ContraTGT 所需边表 + `oulad.content`。 |
| **convert_oulad_abc.py** | 先按模块（如 AAA/BBB/CCC）过滤，再复用 convert_oulad 的建图与特征逻辑，得到更小的 `data_abc_*` 数据。 |
| **script/pretrain.py** | 在时序图上做自监督预训练（真边 vs 负样本边的对比学习），得到 `script/pretrain_model/<data_name>.pth`。 |
| **script/main.py** | 加载预训练权重，用**课程通过/不通过**标签做主训练（BCE + 正类权重），按**课程级**（时间加权聚合）算 Train/Val/Test/NN_Test，保存最佳模型并写入 `result/<data_dir>/script_results_<data_name>.csv`。 |
| **script/utils.py** | 命令行解析（`--data_dir`、`--split_by_ui` 等）、`get_data_paths`、Dataset（支持按时间或按 (u,i) 划分）、EarlyStopping。 |
| **script/sampling.py** | 时空邻居列表构建（邻接表、offset）、`get_neighbor_list`、`get_unique_node_sequence`（已优化：二分查找 + 边数组向量化切片，见性能瓶颈方案 1）等。 |
| **script/model.py** | SpatialTemporal（空间/时间编码器 + 链接预测头）、Top_k 等。 |
| **models/train_xgboost.py** | 使用边+节点特征，按 (u,i) 划分，训练 XGBoost，输出指标到 `result/<data_dir>/`。 |
| **models/train_lstm.py** | 将 (u,i) 序列与节点特征转为时序输入，训练 LSTM，输出指标到 `result/<data_dir>/`。 |
| **models/train_gnn.py** | 2 层 GCN 消息传递 + 链接解码 MLP，按 (u,i) 划分，输出 `gnn_results_*.csv`。 |
| **models/train_node2vec.py** | 图上随机游走 + Skip-Gram（小批量）得到节点嵌入，再 MLP(emb_u, emb_i) 做链接预测；支持 `--num_walks`、`--walk_length`、`--n2v_batch_size`，输出 `node2vec_results_*.csv`。 |

---

## 五、输入输出约定

- **数据输入**：`code/data/all_data/<data_dir>/` 下必须包含 `ml_oulad.csv`、`oulad.content`。script 通过 `--data_dir` 解析到 `code/data/all_data/<data_dir>`（在 `code` 下运行时）。
- **图模型输出**：均在 `code/script/` 下，不向 `code` 根目录新增文件夹；预训练与主训练权重按 `data_name`（即 `data_dir` 名）区分。
- **基线输出**：`code/result/<data_dir>/`，每个数据目录对应一个子文件夹，内含各基线及图模型主训练的 CSV 结果文件（如 `script_results_*.csv`、`xgboost_results_*.csv` 等）。

---

## 六、性能与调参

- **已做优化**：`script/sampling.py` 中 `get_unique_node_sequence` 已按《性能瓶颈与优化方案》方案 1 优化；Node2Vec 使用小批量 Skip-Gram，可通过 `--n2v_batch_size`、`--num_walks`、`--walk_length` 调节速度与质量。
- **Script 与基线可比性**：主训练加 `--split_by_ui` 后按 (u,i) 划分，与 XGBoost/LSTM/GNN/Node2Vec 的划分方式一致，便于公平对比课程级 AUC/AP。
- 若仍较慢，可参考 `script/性能瓶颈与优化方案.md`，或先用小数据（如 `data_abc_0.01`）验证；小数据可适当减小 `--n_epoch`、`--ctx_sample`、`--tmp_sample` 或增大 `--bs`。
