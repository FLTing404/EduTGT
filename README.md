# EduTGT 使用说明

本仓库将 OULAD 开放教育数据整理成时序图（学生–课程边 + 节点特征）。**首要目标为链路预测**（边存不存在，边级 AUC/AP/Acc；ContraTGT 预训练 + 链路主训练，并与 TGN/TGAT/GraphSAGE 等基线对比）。此外支持**课程通过/不通过预测**（ContraTGT + pass/fail 主训练，课程级指标）与**基线分类**（XGBoost / LSTM / GNN / Node2Vec+MLP）。数据与转换脚本在 `data/` 下，结果按任务分为 `result/link/` 与 `result/passfail/`。

---

## 一、目录结构

```
EduTGT/
├── README.md                   # 本说明
├── data/                       # 数据与生成脚本
│   ├── convert_oulad.py        # OULAD 全量 → 时序图格式
│   ├── convert_oulad_abc.py    # 按模块(AAA/BBB/CCC)筛选 → 小规模
│   ├── OULAD_rawdata/          # 原始 OULAD 数据（7 个 CSV，需自行放置）
│   └── all_data/               # 转换后的统一数据目录
│       ├── data_1/、data_0.1/、data_abc_0.01/ 等
│       │   ├── ml_oulad.csv
│       │   ├── oulad.content
│       │   └── ...
│       └── ...
├── script/                     # ContraTGT 预训练（供链路预测与下游复用）
│   ├── pretrain.py             # 预训练
│   ├── utils.py、sampling.py、model.py、graph_transformer.py
│   ├── pretrain_model/、saved_checkpoints/、saved_models/
│   └── 链路预测图模型方案.md
├── link_models/                # 【首要】链路预测：ContraTGT + TGN/TGAT/GraphSAGE 基线
│   ├── main_link.py            # ContraTGT 链路预测入口
│   ├── train_link_tgn.py       # TGN 基线
│   ├── train_link_tgat.py      # TGAT 基线
│   └── train_link_graphsage.py # GraphSAGE 基线
├── passfail_models/            # 通过/不通过：图模型 + 基线
│   ├── main_passfail.py        # ContraTGT 课程通过/不通过（课程级指标）
│   ├── train_xgboost.py、train_lstm.py、train_gnn.py、train_node2vec.py
│   └── ...
└── result/                     # 结果按任务分两类目录
    ├── link/                   # 链路预测（边存不存在）
    │   └── <data_dir>/
    │       ├── link_contratgt_*.csv、link_tgn_*.csv、link_tgat_*.csv、link_graphsage_*.csv
    └── passfail/               # 通过/不通过（课程级）
        └── <data_dir>/
            ├── script_*.csv    # main_passfail 输出
            ├── xgboost_*.csv、lstm_*.csv、gnn_*.csv、node2vec_*.csv
            └── ...
```

---

## 二、环境与依赖

在项目根目录（EduTGT）下一条命令安装所有依赖：

```bash
pip install -r requirements.txt
```

主要包含：`torch`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`tqdm`、`xgboost`。  
**可选**：链路预测基线 TGN / GraphSAGE 需单独安装 `torch-geometric`（`pip install torch-geometric`）。若遇代理或网络错误可暂不安装，ContraTGT、main_link、TGAT 及 passfail 脚本仍可运行。  
若使用 GPU，请先到 [PyTorch 官网](https://pytorch.org) 安装与 CUDA 对应的 `torch`，再执行 `pip install -r requirements.txt`。

---

## 三、数据流程

1. **原始数据**：将 [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) 解压后的 7 个 CSV 放到 `data/OULAD_rawdata/`（如 `studentVle.csv`、`studentInfo.csv`、`courses.csv`、`assessments.csv`、`studentAssessment.csv`、`studentRegistration.csv`、`vle.csv`）。
2. **转换**（在**项目根 EduTGT** 下执行）：
   - **全量或按比例采样**：`python data/convert_oulad.py` → 输出到 `data/all_data/data_{比例}/`（如 `data_1`、`data_0.1`）。
   - **仅 AAA/BBB/CCC 模块**：`python data/convert_oulad_abc.py --modules AAA,BBB,CCC --sample_ratio 0.01` → 输出到 `data/all_data/data_abc_0.01/`。
3. **统一格式**：每个子目录内含 `ml_oulad.csv`（边表：u, i, ts, label, idx 等）、`oulad.content`（节点特征），供 script、link_models、passfail_models 共用。

### 数据划分与预训练-下游对齐

为体现**预训练/数据增强对下游的增益**，默认采用与 ContraTGT 论文一致的**按边随机 1:1:8**（10% 训练 : 10% 验证 : 80% 测试），且**预训练、链路预测、通过/不通过**使用**相同划分与相同 `--seed`**：

- **预训练**（`script/pretrain.py`）：在 10% 训练边上做对比预训练，权重供链路与下游复用。
- **链路预测**（`link_models/main_link.py`）：默认按边 1:1:8，**不**按 (u,i) 划分；可与论文/基线公平对比。
- **通过/不通过**（`passfail_models/main_passfail.py`）：默认同样按边 1:1:8、相同 seed，**不**按时间划分，下游训练集与预训练使用的边一致。

若需按 (u,i) 或按时间划分，可传 `--split_by_ui` 或自行指定 `--train_ratio` 等（见各脚本帮助）。

---

## 四、运行方式

**请在项目根目录（EduTGT）下执行以下命令。**

### 1. 数据转换

```bash
# 全量 → data/all_data/data_1/
python data/convert_oulad.py

# 全量 10% 采样 → data/all_data/data_0.1/
python data/convert_oulad.py --sample_ratio 0.1

# 仅 AAA,BBB,CCC 模块，1% 采样 → data/all_data/data_abc_0.01/
python data/convert_oulad_abc.py --modules AAA,BBB,CCC --sample_ratio 0.01
```

### 2. 预训练（script）

数据来自 `data/all_data/<data_dir>/`，通过 `--data_dir` 指定子目录名（如 `data_abc_0.01`）。预训练权重保存在 `script/pretrain_model/`、`script/saved_checkpoints/`，供链路预测与通过/不通过复用。

```bash
python script/pretrain.py --data_dir data_abc_0.01
```

### 3. 链路预测（ContraTGT + 对比基线）【首要目标】

链路预测结果写入 `result/link/<data_dir>/`（边级 AUC/AP/Acc）。先做预训练，再跑主训练与基线。默认按边 1:1:8 划分，与论文一致。

```bash
# 预训练（若尚未运行）
python script/pretrain.py --data_dir data_abc_0.01

# ContraTGT 链路预测
python link_models/main_link.py --data_dir data_abc_0.01

# 对比基线：TGN、TGAT、GraphSAGE（需安装 PyTorch Geometric 以运行 TGN/GraphSAGE）
python link_models/train_link_tgn.py --data_dir data_abc_0.01
python link_models/train_link_tgat.py --data_dir data_abc_0.01
python link_models/train_link_graphsage.py --data_dir data_abc_0.01
```

详见 `link_models/README.md` 与 `链路预测对比组方案.md`。

**一键多 seed 评估（与论文 Section V-B 一致）**：运行一条命令即可跑 ContraTGT + 基线、多随机种子并生成汇总表 `paper_eval_summary_<data_dir>.csv`：

```bash
# 在项目根 EduTGT 下执行；需先完成预训练
python link_models/run_paper_eval.py --data_dir data_abc_0.01 --seeds 42,43,44,45,46

# 只跑部分方法
python link_models/run_paper_eval.py --data_dir data_abc_0.01 --methods contratgt,tgat

# 仅汇总已有 CSV，不重新跑
python link_models/run_paper_eval.py --data_dir data_abc_0.01 --no_run
```

### 4. 课程通过/不通过（passfail_models）

依赖上述预训练权重。结果写入 `result/passfail/<data_dir>/`（课程级 AUC/AP/Acc）。默认与预训练一致：按边 1:1:8、相同 seed，不按时间划分。

```bash
# 默认：按边 1:1:8，与预训练对齐
python passfail_models/main_passfail.py --data_dir data_abc_0.01
```

可选参数：`--ctx_sample`、`--tmp_sample`、`--n_epoch`、`--bs`、`--gpu`、`--drop_out`、`--split_by_ui`、`--seed` 等（见 `script/utils.py`）。

**一键多 seed 评估（方案 A：全程按边 1:1:8）**：运行一条命令即可跑 ContraTGT 通过/不通过、多随机种子并生成汇总表 `paper_eval_summary_<data_dir>.csv`：

```bash
# 在项目根 EduTGT 下执行
python passfail_models/run_paper_eval.py --data_dir data_abc_0.01 --seeds 42,43,44,45,46
# 仅汇总已有 CSV，不重新跑
python passfail_models/run_paper_eval.py --data_dir data_abc_0.01 --no_run
```

### 5. 通过/不通过：Logistic / LSTM 头与纯基线（passfail_models）

与 main_passfail（ContraTGT + MLP 头）同一 (u,i) 划分、课程级评估。方案见 `passfail_models/修改方案-头与纯基线.md`。

```bash
# 预训练 + Logistic 与 纯 Logistic
python passfail_models/train_logistic.py --data_dir data_abc_0.01 --mode pure
python passfail_models/train_logistic.py --data_dir data_abc_0.01 --mode contratgt

# 预训练 + LSTM 与 纯 LSTM
python passfail_models/train_lstm.py --data_dir data_abc_0.01 --mode pure
python passfail_models/train_lstm.py --data_dir data_abc_0.01 --mode contratgt

# 纯 MLP（与 main_passfail 的 ContraTGT+MLP 对照）
python passfail_models/train_mlp.py --data_dir data_abc_0.01
```

可选：预训练表示预先提取并保存（供多处复用）  
`python passfail_models/extract_embeddings.py --data_dir data_abc_0.01 --seed 42`  
结果写入 `result/passfail/<data_dir>/logistic_*.csv`、`lstm_*.csv`、`mlp_pure_*.csv`。

---

## 五、模块说明

| 模块 | 作用 |
|------|------|
| **data/convert_oulad.py** | 读 OULAD 全部 7 张表，建学生/课程节点与边（含时间、标签），生成节点特征，输出边表 + `oulad.content` 到 `data/all_data/`。 |
| **data/convert_oulad_abc.py** | 先按模块（如 AAA/BBB/CCC）过滤，再复用 convert_oulad 的建图与特征逻辑，输出到 `data/all_data/data_abc_*`。 |
| **script/pretrain.py** | 在时序图上做自监督预训练（真边 vs 负样本边），得到 `script/pretrain_model/<data_name>.pth`。 |
| **passfail_models/main_passfail.py** | 加载预训练权重，用**课程通过/不通过**标签做主训练（BCE + 正类权重），按**课程级**算 Train/Val/Test/NN_Test，结果写 `result/passfail/<data_dir>/script_*.csv`。 |
| **script/utils.py** | 命令行解析（`--data_dir`、`--split_by_ui` 等）、`get_data_paths`、Dataset（按时间或按 (u,i) 划分）、EarlyStopping。 |
| **script/sampling.py** | 时空邻居列表构建、`get_neighbor_list`、`get_unique_node_sequence` 等。 |
| **script/model.py** | SpatialTemporal（空间/时间编码器 + 链接预测头）、Top_k 等。 |
| **passfail_models/train_xgboost.py** | 按 (u,i) 划分，训练 XGBoost，输出到 `result/passfail/<data_dir>/`。 |
| **passfail_models/train_lstm.py** | 时序输入训练 LSTM，输出到 `result/passfail/<data_dir>/`。 |
| **passfail_models/train_gnn.py** | 2 层 GCN + MLP 预测通过/不通过，输出到 `result/passfail/<data_dir>/`。 |
| **passfail_models/train_node2vec.py** | 随机游走 + 嵌入 + MLP 预测通过/不通过，输出到 `result/passfail/<data_dir>/`。 |
| **link_models/main_link.py** | ContraTGT 链路预测（边存在性），结果写 `result/link/<data_dir>/link_contratgt_*.csv`。 |
| **link_models/train_link_tgn.py** | TGN 链路预测基线（PyG），写 `link_tgn_*.csv`。 |
| **link_models/train_link_tgat.py** | TGAT 链路预测基线，写 `link_tgat_*.csv`。 |
| **link_models/train_link_graphsage.py** | GraphSAGE 链路预测基线（PyG），写 `link_graphsage_*.csv`。 |

---

## 六、输入输出约定

- **数据输入**：`data/all_data/<data_dir>/` 下必须包含 `ml_oulad.csv`、`oulad.content`。各脚本通过 `--data_dir` 解析，**请在项目根（EduTGT）下运行**。
- **预训练权重**：保存在 `script/pretrain_model/`、`script/saved_checkpoints/`、`script/saved_models/`，按 `data_name` 区分；main_passfail 与 script 共用同一套权重。
- **结果输出**：**链路预测**（首要目标，边存不存在）→ `result/link/<data_dir>/`；**通过/不通过**（课程级）→ `result/passfail/<data_dir>/`，内含 script（ContraTGT）、XGBoost、LSTM、GNN、Node2Vec 等 CSV。

---

## 七、性能与调参

- **Script 与基线可比性**：main_passfail 加 `--split_by_ui` 后按 (u,i) 划分，与 XGBoost/LSTM/GNN/Node2Vec 的划分方式一致，便于公平对比课程级 AUC/AP。
- Node2Vec 使用小批量 Skip-Gram，可通过 `--n2v_batch_size`、`--num_walks`、`--walk_length` 调节。
- 若较慢，可先用小数据（如 `data_abc_0.01`）验证，或适当减小 `--n_epoch`、`--ctx_sample`、`--tmp_sample`、增大 `--bs`。
