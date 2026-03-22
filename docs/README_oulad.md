# OULAD → ContraTGT / EduTGT 数据说明

> 文档索引见 **[README.md](./README.md)**；算法与仓库对照见 **[方法与ContraTGT对比.md](./方法与ContraTGT对比.md)**。命令行步骤见 **[使用指南.md](./使用指南.md)**。

## 1. 映射关系

| OULAD | 本构建的图 / 文件 |
|--------|-------------------|
| `studentVle.csv`（按课程实例过滤） | 动态边：学生 ↔ **VLE 站点节点** `(code_module, code_presentation, id_site)` |
| `vle.csv`（`id_site` join） | 每条 VLE 记录补齐 `activity_type` |
| `studentInfo.csv` + `studentRegistration.csv` | 学生节点特征（不含 `final_result`） |
| `courses.csv` | 站点侧 `module_presentation_length` 与全表 module/presentation one-hot 空间 |
| 聚合键 `(id_student, code_module, code_presentation, id_site, date)` | 同一学生–站点–日一条边；`ts = pres_index * 1000 + date`（`date` 为相对日） |

节点原始 ID 见 `node_map.csv`：`student::<id_student>`、`site::<MODULE>_<PRESENTATION>_<id_site>`。

### 1.1 与 ContraTGT 官方数据需求是否一致（「一样即为合适」指什么）

若「合适」指 **能否像官方仓库一样被 `Dataset` / `pretrain.py` / `main.py` 读入并训练**，应对齐的是 **文件形态与维度约定**，而不是在 `ml` 里预先写满 `label=0` 的负边。

| 检查项 | ContraTGT（`ContraTGT/data`、`ContraTGT/node_feature`） | 本仓库 OULAD 导出 |
|--------|--------------------------------------------------------|-------------------|
| 边列表 CSV | 首行表头 `,u,i,ts,label,idx`，数据从第二行起；列语义与 `utils.Dataset` 一致 | 相同 |
| `label` 列 | 官方 `ml_slashdot.csv` 等 **几乎全为 1**；`ml_wiki.csv` 中偶见 `-1`，属数据本身 | OULAD 写 **全 1**，与 slashdot 类文件一致 |
| 训练时谁提供「负例」 | **`main.py` / `pretrain.py` 并不用文件里的 `label` 做 BCE 目标**：对当前 batch 仍设 `pos_label=1`、`neg_label=0`，负样本靠 **`RandEdgeSampler` 随机抽终点节点**（与 pretrain 中构造 fake 边一致） | 行为相同，**不要求**在 `ml` 里显式列出未观测负边 |
| 节点特征 `.content` | `node_feature/<name>.content`，无表头，一行一个节点；经 `normalize_features` 后与 `Dataset` 重映射后的节点下标对齐 | `data/processed/data_*/*.content` 同样无表头、一行一节点；节点 id 在边中连续出现时与重映射一致 |

**结论（格式 / 代码路径）：** 当前 OULAD 预处理输出 **满足 ContraTGT 系代码的数据接口**；与「是否和官方 `node_feature` 文件逐列同分布」无关，只要 **行数 = 图中节点数、列数为实值特征** 即可被同一套 `read_csv` + 归一化 + 可选 pad 逻辑消费。

**结论（任务语义）：** 官方数据在 **多个不同终点节点** 时，`RandEdgeSampler` 抽到的 `dst` 有多样性。当前预处理 **仅导出学生–VLE 站点二部图**，终点为各 presentation 下的 `id_site`，候选通常远多于「每 presentation 一个课程节点」的旧版构图。支持 **整模块**（如 `data/processed/data_AAA/`）与 **单课程实例**（如 `data/processed/data_AAA_2013J/`）。任务与切分层面的注意见第 3 节。

## 2. 边聚合逻辑

对同一 `(学生, 课程实例, id_site, date)`：

- 先按 `id_site` join `vle.csv` 得到该站点在元数据中的 `activity_type`（若无法匹配则丢弃该行并 `[WARN]`）；`sum_click <= 0` 的行丢弃。
- `total_clicks` = 该组内 `sum(sum_click)`。
- 写出 `ml_*.csv` 前，为每条边的 `activity_type` 构造 **one-hot** 列 `activity_type_*`（用于 `edge_attr.csv` 与后续特征侧一致）。

边按 `(ts, u, i)` 排序，保证 **ts 非递减**；`idx` 为 1…E 的事件序号（与 ContraTGT 中 `get_unique_node_sequence` 的假设一致）。

## 3. 当前代码与设置：与「真正的 link prediction」的差距

本节对应**学生–站点图**预处理与 `EduTGT` 训练脚本（`--data_dir data/processed/data_<MODULE>/` 等），说明与经典「链路预测」任务定义之间的差距。

### 3.1 图与标签（预处理输出）

- 图为 **学生节点 + 站点节点** `(presentation, id_site)`；边为 **学生 → 站点** 的日聚合交互。
- 边在 `ml_*.csv` 里 **`label` 列写 1**（表示该条为观测到的交互边）。这与 ContraTGT 里 **`ml_slashdot.csv` 等以 1 为主** 的写法一致；且 **`main.py` 训练阶段并不会用该列构造 BCE 的真值**，而是对「当前 batch 真实边」标为正、对「替换 `dst` 后的边」标为负——负例来源仍是 **代码内从训练中出现过的 `dst` 集合里随机采样**，**不是**依赖你在 CSV 里预先写 `label=0` 的负边。
- 因此：**就「满足 ContraTGT 训练脚本的数据需求」而言，全 `label=1` 是合适、且与官方用法不冲突。** 与教科书式「显式负边表」的差异，官方脚本同样存在。

→ 站点图下 **`dst` 候选通常较多**（大量 `id_site`），与「极少课程节点」的旧版构图相比更接近多候选设定；但与文献中严格的 **未观测负边池、按对切分** 等仍可能不对齐（见 3.2–3.4）。

### 3.2 负采样在代码里的实际行为（关键）

`main.py` / `pretrain.py` 沿用 ContraTGT 写法：

- `RandEdgeSampler(train_data['idx'][:, 1])`（及验证/测试时对全图 `dst` 的类似用法）从 **`dst` 在训练边中出现过的取值** 里均匀抽样「假终点」。
- **单课程实例**（如仅 `AAA_2013J`）时，站点节点仍来自该实例的 VLE，`dst` 多样性取决于该课下活跃站点数；**整模块**合并多 presentation 时，候选进一步增加。
- 若某极端子集上训练边仍只覆盖极少数站点 id，负采样仍可能退化——实践中站点图远好于「单 presentation 单课程节点」情形。

→ 与标准多候选 link prediction 的差距见 3.4（切分方式、负例语义等）。

### 3.3 评估与时间切分

- `utils.Dataset` 按 **时间戳分位数**（约前 10% / 10%–20% / 余下）划分 train/val/test，侧重 **时间外推**，而不是随机留出 **(u,i) 对**。
- 真正的 link prediction 常配合 **按边或按对留出** 的 test，并明确 **负样本分布**（如均匀、流行度偏置等）。当前设置 **未** 按该范式构造。

### 3.4 小结（对照「真正的 link prediction」）

| 维度 | 当前 EduTGT + OULAD（学生–站点图） | 典型 link prediction |
|------|-------------------------------------|----------------------|
| 目标节点多样性 | 大量站点节点 id，`dst` 通常较丰富 | 大量候选终点或显式负例池 |
| 负采样语义 | 从训练中出现过的站点 id 中换 `dst` | 负样本多为「未出现边」或难例 |
| 标签（文件内） | 观测边写 1（与 slashdot 类官方 ml 一致；BCE 真值由代码 pos/neg + 采样决定） | 论文设定里常单独讨论负例协议 |
| 切分 | 按时间分位 | 常按边/对随机切分 + 负采样协议 |

**结论：** 站点图在终点多样性上已显著优于旧版「每 presentation 一课程节点」构图；若要与文献级 link prediction 严格对齐，仍需关注 **切分与负例协议**（见第 4 节）。

**说明：** 此前约定 **暂不修改 `sampling.py` 的负采样逻辑**；若要做严肃 link prediction，见下文第 4 节改进方向。

## 4. 推荐改进

1. **构图（当前默认）**：`python data/scripts/preprocess_oulad_for_contratgt.py --module AAA` → `data/processed/data_<MODULE>/`。节点为 `(code_module, code_presentation, id_site)`；边按日聚合 `total_clicks`；`studentVle` 无法 join `vle` 的行会丢弃并在 stderr 打 `[WARN]`。站点特征：`activity_type` one-hot、module/presentation one-hot、`module_presentation_length` z-score、`week_from`/`week_to`（可解析则 z-score，否则按列中位数填）。**负采样**从训练边中出现过的**站点节点 id** 中抽 `dst`。
2. **Temporal link**：以「预测下一时间窗是否交互」等为标签，需重新定义标签与切分（与 ContraTGT 当前 `main.py` 的 BCE+采样目标可并行讨论，属任务 redesign）。

## 5. 切分方式（与 ContraTGT 一致）

`utils.Dataset` 使用 **时间戳分位数**：`val_time`、`test_time` 分别为全体边 `ts` 的 **10% 与 20% 分位数**（不是按条数 70%/15%/15%）。

- train：`ts <= val_time`
- val：`val_time < ts <= test_time`
- test：`ts > test_time`

脚本同时写出 `train_idx.txt`、`val_idx.txt`、`test_idx.txt`（及新节点相关的 `nn_*`），与 `Dataset` 逻辑一致，便于对照；**训练代码仍以 `Dataset()` 内部分割为准**。

## 6. `--module`（仅学生–站点图）

- **三位模块码** `AAA`、`BBB`、…、`GGG`：合并 `ALLOWED_COURSES` 中该模块的 **全部 presentation**，输出 `data/processed/data_<MODULE>/`（如 `data/processed/data_AAA/`），`ml` / `.content` 前缀为 `<MODULE>`。
- **单实例** `<MODULE>-<PRESENTATION>`（须出现在 `ALLOWED_COURSES` 中，如 `AAA-2013J`）：输出 `data/processed/data_<MODULE>_<PRESENTATION>/`，文件前缀 `AAA_2013J`。
- **`ALL`**：按模块首次出现顺序依次处理 **AAA → BBB → … → GGG**；每个模块只生成整模块目录 `data/processed/data_<MODULE>/`（**不包含**单实例子目录）。

各模块内含的 presentation 集合由脚本内 `ALLOWED_COURSES` 定义（例如 AAA 含 2013J、2014J；BBB 含四个学期等）。

```bash
python data/scripts/preprocess_oulad_for_contratgt.py --module AAA
python data/scripts/preprocess_oulad_for_contratgt.py --module AAA-2013J
python data/scripts/preprocess_oulad_for_contratgt.py --module BBB
python data/scripts/preprocess_oulad_for_contratgt.py --module ALL
```

`ts = pres_index * 1000 + date`（`date` 为各课相对日，块间不可比）。

## 7. 运行方式

**更完整的命令与参数说明**见 **`EduTGT/EduTGT/docs/使用指南.md`**（预处理 / 预训练 / 微调 / 消融 / 清空权重等）。

在含 `main.py` 的 `EduTGT` 目录下（一般为 `EduTGT/EduTGT`）：

```bash
# 预处理（默认读取 data/raw/OULAD）
python data/scripts/preprocess_oulad_for_contratgt.py --module AAA
python data/scripts/preprocess_oulad_for_contratgt.py --module AAA-2013J
python data/scripts/preprocess_oulad_for_contratgt.py --module ALL

# 预训练 / 微调（--data_dir 指向数据目录；data_name 为该文件夹名，用于权重文件名）
python pretrain.py --data_dir data/processed/data_AAA --n_epoch 50 --bs 800
python main.py --data_dir data/processed/data_AAA --n_epoch 50 --bs 800
python pretrain.py --data_dir data/processed/data_AAA_2013J --n_epoch 50 --bs 800
python main.py --data_dir data/processed/data_AAA_2013J --n_epoch 50 --bs 800
```

仍可使用原始 ContraTGT 风格：`python main.py -d slashdot`（从 `data/ml_*.csv` 与 `node_feature/*.content` 读取）。

## 8. 输出文件

每个 `data/processed/data_<MODULE>/`（整模块）包含：

- `ml_<MODULE>.csv`、`<MODULE>.content`

每个 `data/processed/data_<MODULE>_<PRESENTATION>/`（单实例）包含：

- `ml_<MODULE>_<PRESENTATION>.csv`、`<MODULE>_<PRESENTATION>.content`

`stats.json` 中 `graph` 字段为 `"vlesite"`，表示学生–站点二部图（与历史实现一致，仅目录名不再带 `_vlesite`）。

其余通用：

- `ml_*.csv`：`,u,i,ts,label,idx`（首列为行号，与官方示例一致）
- `*.content`：每行一个节点的特征（无表头）
- `edge_attr.csv`：与边对齐的聚合统计
- `node_map.csv`：`node_id` → 原始 ID
- `stats.json`：规模与时间范围等
- `train_idx.txt` / `val_idx.txt` / `test_idx.txt` / `nn_val_idx.txt` / `nn_test_idx.txt`

## 9. 训练侧注意

- **特征维度**：Transformer 多头 `n_heads=4` 要求输入维为 4 的倍数；`main.py` / `pretrain.py` 会在加载后 **右侧零填充** 至 4 的倍数（不改变磁盘上的 `.content`）。
- **无 GPU**：已自动回退 `cpu`，权重 `torch.load(..., map_location=device)`。
- **首跑**：需先有 `outputs/pretrain/<数据目录名>.pth`，否则 `main.py` 会报错。

## 10. 消融实验脚本（presentation 关系负采样 / 同学生时序一致性）

在 `EduTGT/EduTGT` 目录下，需 **先对同一 `data_dir` 跑完 `pretrain.py`**（与 `main.py` 共用 `outputs/pretrain/{data_name}.pth`）。

| 脚本 | 作用 | 输出权重 |
|------|------|----------|
| `main_ablation_pres_relation.py` | 微调训练阶段负样本 `dst` 按 **`pres_relation.json`**（课程实例站点 Jaccard 矩阵 W）与当前边 `ts` 所在 presentation **偏置采样**（与 `main.py` 同结构，无 edge 拼接） | `outputs/models/{data_name}_pres_rel.pth` |
| `main_ablation_student_temporal.py` | 训练时在 BCE 上增加 **batch 内同源节点 `u` 的源端 `getEmbed` 余弦一致性**：对所有同 `u` 索引对求平均 `1 - cos` | `outputs/models/{data_name}_stc.pth` |
| `run_ablation_experiments.py` | 依次执行上述两个脚本（可用 `--skip_pres` / `--skip_stc` 关掉某一项；`--skip_edge` 为 `--skip_pres` 的别名） | 同上 |

整模块图需先生成关系文件：

```bash
python data/scripts/build_pres_relation.py --data_dir data/processed/data_AAA
```

一键示例：

```bash
python run_ablation_experiments.py --data_dir data/processed/data_AAA --n_epoch 50 --bs 800
```

单独跑、并调节时序项权重：

```bash
python main_ablation_pres_relation.py --data_dir data/processed/data_AAA --n_epoch 50 --bs 800
python main_ablation_student_temporal.py --data_dir data/processed/data_AAA --lambda_tc 0.1 --n_epoch 50 --bs 800
```

**说明**：`pres_relation` 消融依赖数据目录下的 **`pres_relation.json`**；`EarlyStopping` 的检查点写在 `outputs/checkpoints/{data_name}_pres_rel.pth` 与 `outputs/checkpoints/{data_name}_stc.pth`，避免与原版 `main.py` 互相覆盖。

## 11. 训练结果自动记录（`outputs/logs/training_runs.jsonl`）

`main.py`、`main_ablation_pres_relation.py`、`main_ablation_student_temporal.py` 在**打印完 test / nn_test 指标后**，会向 **`outputs/logs/training_runs.jsonl` 追加一行 JSON**，并打印一行 `[RESULT] ...` 摘要。

字段含：`ended_at`、`run_type`（`baseline` / `ablation_pres_relation` / `ablation_student_temporal`）、`data_name`、`data_dir`、超参、`metrics.test` / `metrics.nn_test`、`training.early_stopped` 与 `best_epoch`。设计说明见 **`docs/training_result_log.md`**。
