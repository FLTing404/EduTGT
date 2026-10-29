# EduTGT：知识追踪与评估探索

当前处于方向调整后的初期探索阶段，重点是教育数据集适配、知识追踪基线、超图模块消融，以及错误识别与阈值选择。现有实验用于筛选问题和候选方案，尚未形成定型方法。

本分支 `research/kt-exploration-20261029` 从旧 EduTGT 历史继续，工作目录以当前新代码为主体。此前的 OULAD 时序图链路预测和课程通过预测保留在 [main 的固定版本](https://github.com/FLTing404/EduTGT/tree/dbeb7d42e656967c3eb3b0c4f69f9f64d73201d4)。两套任务、标签和数据接口不同；关系与目录说明见 [方向调整说明](docs/方向调整与目录说明.md)。

## 从哪里开始

| 路径 | 内容 |
| --- | --- |
| [reproduction/](reproduction/README.md) | 数据预处理、BKT/DKT/AKT/simpleKT/FC-HGNN 等模型、训练与分析入口 |
| [reproduction/configs/](reproduction/configs/) | 小规模检查、前期复现和新数据集候选筛选配置 |
| [reproduction/tests/](reproduction/tests/) | 数据划分、因果掩码、模型、阈值和学生聚类抽样测试 |
| [datasets/](datasets/README.md) | 六套数据的来源、许可、校验和、统计与下载审计脚本 |
| [汇报材料/项目综合汇报.md](汇报材料/项目综合汇报.md) | 已有实验观察、证据边界与待验证问题 |
| [汇报材料/新数据集双方向实验简报.md](汇报材料/新数据集双方向实验简报.md) | XES3G5M、ASSIST2017、NIPS Task 3/4 的初步筛选记录 |
| [reproduction/artifacts/](reproduction/artifacts/README.md) | 纳入 Git 的汇总指标、实验协议和报告图表 |

## 环境与检查

使用 Python 3.10 或更高版本，从项目根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r reproduction/requirements-dev.txt
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m pytest reproduction/tests -q
python reproduction/test_new_datasets.py --help
```

仅运行实验时，可安装根目录的 `requirements.txt`。AKT/simpleKT 的固定上游版本、许可证和本地补丁均保留在 [vendor/pykt](reproduction/src/fchgnn_repro/vendor/pykt/PATCHES.md)。

## 新数据集实验

先按各数据集 `metadata/README.md` 的来源说明补齐本地 `raw/` 文件。根目录下的 `datasets/` 与 `reproduction/` 应保持同级；配置中的路径相对于配置文件解析。

```bash
# 小规模技术检查，仍需对应原始数据
python reproduction/test_new_datasets.py all --config reproduction/configs/new_datasets_smoke.yaml

# 三数据、七变体、三种子的候选筛选
python reproduction/test_new_datasets.py all --config reproduction/configs/new_datasets_dual.yaml
```

完整流程和分析命令见 [实验说明](reproduction/README.md#新数据集双方向实验)。小规模配置用于检查流程；完整候选筛选仍是短预算实验。当前观察提示应继续检验阈值目标与错误识别的关系；相同输入的消融尚未支持稳定的超图传播收益。

## 文件范围

Git 包含代码、配置、测试、数据来源与许可、汇总实验记录。原始数据、逐事件预测、学生映射、预处理数据、权重、虚拟环境、临时文件和报告导出版留在本地。研究资料中的参考论文链接继续保留；本分支不包含论文初稿或投稿规划。
