# 实验结果索引

| 目录 / 文件 | 用途 |
|---|---|
| [summary/对比试验.csv](summary/对比试验.csv) | 论文所用历史模型对比汇总 |
| [summary/消融实验.csv](summary/消融实验.csv) | 论文所用历史消融/变体汇总 |
| [link_prediction/](link_prediction/) | 主实验预训练、微调及早停权重、日志、带时间戳批跑表 |
| [link_prediction/batches/](link_prediction/batches/) | 原 `EduTGT/outputs/result/`，保留每次批跑 CSV 和 `_runs` |
| [link_prediction/baselines/](link_prediction/baselines/) | 原 `EduTGT/baseline/results/`，包括直接运行结果及历史 smoke 记录 |
| [pass_prediction/](pass_prediction/) | 下游批跑汇总、随机种子/配置、模块权重、运行日志和 ROC 图 |
| [archive/legacy/](archive/legacy/) | 原 `result/result_old/` |
| [archive/2026-03-26/](archive/2026-03-26/) | 原 `result/result_0326/` |
| [files.csv](files.csv) | 所有原有结果文件的路径、大小、SHA256 和类别 |

文件内容全部保留，未重跑实验。历史归档与主结果之间存在同内容副本，查看 `files.csv` 的 SHA256 可识别；它们不代表独立重复实验。汇总表是历史单次运行选取值，消融标签及评测差异需结合 [前期交接文档](../docs/2026-10-27_前期工作交接.md) 理解。

日志和运行记录内部仍保留当时写入的文件路径，迁移对照见 [文件整理说明](../docs/文件整理说明.md)。下载暂存文件 `.download` 和 `.digest` 原样保留，其内容未按正式模型验证。

新运行的主实验写入 `link_prediction/`；下游写入 `pass_prediction/`。历史归档目录仅用于留存旧结果。
