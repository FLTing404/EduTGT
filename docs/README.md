# EduTGT 文档索引

EduTGT：基于 ContraTGT 时空图 Transformer 的 OULAD 教育交互链路预测。  
代码执行目录一律为 **`EduTGT/EduTGT/`**。

---

| 文档 | 内容 |
|------|------|
| **[方法与ContraTGT对比.md](./方法与ContraTGT对比.md)** | 算法思路：时空双路编码、预训练+微调、课程感知偏置负采样；与 ContraTGT（IJCAI）的异同与本项目创新点。 |
| **[使用指南.md](./使用指南.md)** | 命令行手册：预处理、预训练、消融批跑、对比基线、输出目录结构、OULAD 数据格式速查。 |

---

其他说明：
- **`baseline/README.md`**：TGN / TGAT / JODIE 基线实现说明。
- **`methods/`**：所有训练方法脚本（`main.py`、`main_ablation_*.py`）。
- **`outputs/logs/training_runs.jsonl`**：各方法训练结束后自动追加的 JSON 记录。
