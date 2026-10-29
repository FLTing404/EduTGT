# 链路预测训练产物

本目录在整理后位于仓库根目录的 `results/link_prediction/`，由 `EduTGT/paths.py` 统一解析。主代码仍从 `EduTGT/` 运行。

| 子目录 | 内容 |
|---|---|
| `pretrain/` | `pretrain.py` 最终权重，微调及消融从此加载 |
| `checkpoints/` | `EarlyStopping` 最佳 checkpoint |
| `models/` | 微调结束写出的 `SpatialTemporal` 权重 |
| `middle/` | 预训练过程中的中间权重 |
| `logs/` | `training_runs.jsonl`，由 `result_logger.py` 追加 |
| `batches/` | 消融与基线批跑 CSV，以及 `_runs/` 中各基线的 JSON 和 checkpoint |
| `baselines/` | `baseline/run_all_baselines.py` 及单模型 runner 的默认结果 |

历史批次另存于 `results/archive/`，论文汇总表位于 `results/summary/`。整理只调整文件位置与输出路径，没有重新训练或改写历史指标。
