# EduTGT

面向 OULAD 学生与学习资源交互的时序图学习实验，包括链路预测、消融、TGN/TGAT/JODIE 风格基线，以及课程通过预测下游任务。

## 目录

| 目录 | 内容 |
|---|---|
| [EduTGT/](EduTGT/) | 主模型、预训练、消融、基线和数据处理脚本 |
| [downstream_pass_prediction/](downstream_pass_prediction/) | 课程通过预测代码、配置、标签和学生划分 |
| [ContraTGT/](ContraTGT/) | 上游参考实现及原有公开图数据、权重；保留原 LICENSE |
| [results/](results/) | 汇总表、所有实验日志、JSON、ROC 图和模型权重 |
| [paper/](paper/) | 论文草稿、修订稿和参考材料 |
| [docs/](docs/) | 前期交接文档、目录迁移说明和本地数据清单 |
| [data/](data/) | OULAD 探索 notebook；本地另存的原始数据不上传 |

先看 [结果索引](results/README.md) 和 [前期工作交接](docs/2026-10-27_前期工作交接.md)。具体命令见 [主实验使用指南](EduTGT/docs/使用指南.md) 与 [下游任务说明](downstream_pass_prediction/README.md)。

## 安装与运行

以下命令从仓库根目录开始。依赖未锁定为原实验环境；原实验配置和已知问题见交接文档。

```bash
pip install -r requirements.txt

# 只检查代码语法、目录和结果清单，不训练模型
python scripts/check_layout.py

# 将 OULAD 七张原始 CSV 放入 EduTGT/data/raw/OULAD/ 后构图
python EduTGT/data/scripts/preprocess_oulad_for_contratgt.py --module ALL

# 主实验命令在主代码目录执行
cd EduTGT
python pretrain.py --data_dir data/processed/data_AAA --n_epoch 50 --bs 800
python methods/main_ablation_pres_relation.py --data_dir data/processed/data_AAA
```

下游脚本从仓库根目录执行，例如：

```bash
python downstream_pass_prediction/scripts/run_pass_batch_abc_seeds.py --modules data_AAA --strict-pretrain
```

主实验输出到 `results/link_prediction/`，下游输出到 `results/pass_prediction/`。原有历史文件原样保留，没有重新训练或重算历史分数。汇总表包含历史单次运行选取值，具体评测差异、消融标签含义及重复归档说明见交接文档。

## 数据与上传范围

代码、数据处理脚本、notebook、处理元数据、小型下游标签与划分、论文、结果及权重均纳入 Git。两套 OULAD 原始数据和处理后的大型图表/索引留在本地，由根 [.gitignore](.gitignore) 忽略；详细文件、大小与 SHA256 见 [本地数据清单](docs/local-data-manifest.csv)。从 GitHub 克隆后需补齐数据或运行预处理。

目录整理及旧路径对照见 [文件整理说明](docs/文件整理说明.md)。原 `FLTing404/EduTGT` 提交历史继续保留；本项目根目录现为统一的 Git 工作区。
