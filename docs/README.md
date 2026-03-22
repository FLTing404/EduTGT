# EduTGT 文档索引

本目录收录 **EduTGT**（基于 ContraTGT 思想的 OULAD 教育时序链路实验工程）的说明文档。代码执行目录一律为仓库内 **`EduTGT/EduTGT/`**（与 `main.py` 同级）。

---

## 核心文档（建议阅读顺序）

| 文档 | 内容 |
|------|------|
| **[方法与ContraTGT对比.md](./方法与ContraTGT对比.md)** | 论文/算法思路：动态图链路预测、时空 Graph Transformer、预训练与微调；与 **ContraTGT（IJCAI）** 官方仓库的异同与本项目扩展。 |
| **[使用指南.md](./使用指南.md)** | 命令行手册：预处理、`pretrain` / `main`、消融、`baseline`、目录结构（`data/processed`、`outputs/`）等。 |

---

## 专题与附录

| 文档 | 内容 |
|------|------|
| **[run_batch_metrics_csv.md](./run_batch_metrics_csv.md)** | 批跑脚本 **`run_batch_metrics_csv.py`**：一键 AAA/BBB × main/消融/baseline × 多 seed，汇总 **`outputs/result/*.csv`**。 |
| **[README_oulad.md](./README_oulad.md)** | OULAD → 学生–VLE 站点图、`ml_*.csv` / `.content` 与 ContraTGT 数据接口对齐、时间切分、负采样语义、任务边界说明。 |
| **[MIGRATION_CD.md](./MIGRATION_CD.md)** | 目录重构（`data/processed`、`outputs/`）旧路径 → 新路径对照表。 |
| **[training_result_log.md](./training_result_log.md)** | `outputs/logs/training_runs.jsonl` 字段设计与写入约定。 |
| **[run_record_main_data_AAA_2013J.md](./run_record_main_data_AAA_2013J.md)** | 示例运行记录（单实例）。 |

---

## 仓库内其他说明

- 项目根 **`paths.py`**：`outputs/` 与 `data/` 分层路径定义。  
- **`baseline/README.md`**：TGN / TGAT / JODIE 风格基线与主流程公平对比说明。  
- 并列目录 **`ContraTGT/`**（若存在）：上游官方脚本副本，数据路径约定与 EduTGT 不同，勿与 OULAD 流程混用。
