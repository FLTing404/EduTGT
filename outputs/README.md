# 训练产物（方案 D）

| 子目录 | 原路径 | 说明 |
|--------|--------|------|
| **`pretrain/`** | `pretrain_model/` | `pretrain.py` 最终权重；`main.py` / 消融从此加载 |
| **`checkpoints/`** | `saved_checkpoints/` | `EarlyStopping` 最佳 checkpoint |
| **`models/`** | `saved_models/` | 微调结束写出的 `SpatialTemporal` 权重 |
| **`middle/`** | `middle_model/` | 预训练过程中的中间权重 |
| **`logs/`** | （自 `results/` 迁入）`training_runs.jsonl` | 现为 **`outputs/logs/training_runs.jsonl`**，由 `result_logger` 追加 |

清空权重（保留日志）：`python clear_training_artifacts.py -y`  
连同日志一起清空：`python clear_training_artifacts.py -y --with-logs`
