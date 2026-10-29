# OULAD 数据与处理脚本

- `scripts/`：构图预处理和 presentation 关系生成脚本，纳入版本管理。
- `raw/OULAD/`：七张原始 CSV 的默认位置，数据留在本地。
- `processed/data_AAA/` 至 `processed/data_GGG/`：处理后的图数据；上传 `stats.json`、`pres_relation.json` 等小型元数据，大型表和索引留在本地。

从仓库根目录运行：

```bash
python EduTGT/data/scripts/preprocess_oulad_for_contratgt.py --module ALL
```

现有本地数据清单见 [local-data-manifest.csv](../../docs/local-data-manifest.csv)。下游标签与学生切分保存在仓库根的 `downstream_pass_prediction/data/`；训练结果统一到根 `results/`。
