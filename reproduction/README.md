# 知识追踪与评估实验

当前为初期探索。主入口是下方的 [新数据集双方向实验](#新数据集双方向实验)，用于比较超图候选与序列基线，以及阈值目标对错误识别的影响。前期 FC-HGNN 趋势复现保留为实验基础。安装步骤见 [根目录说明](../README.md#环境与检查)，需要 Python 3.10+。

前期工程在 ASSISTments 2012–2013、EdNet-KT1 与 OULAD 上建立统一的下一次作答正确性预测任务。目标是验证参考论文所描述的性能趋势，不宣称逐数值复刻。

新增 XES3G5M、ASSISTments 2017、NIPS2020 Task 3/4，分别检验超图增强候选与现代基线的表现，以及类别不平衡下的评估偏差和错误敏感决策。原始下载文件、来源和许可见 [数据目录](../datasets/README.md)。

## 模型

- BKT：全局参数网格选择的经典贝叶斯知识追踪。
- DKT：题目、知识点与作答嵌入输入 LSTM，并对下一题预测。
- AKT / simpleKT：使用固定版本的作者官方 pyKT 实现，适配统一的题目、代表知识点和下一题预测接口；源码许可、哈希和补丁见 [上游记录](src/fchgnn_repro/vendor/pykt/PROVENANCE.json) 与 [适配说明](src/fchgnn_repro/vendor/pykt/PATCHES.md)。
- SAINT-style：带因果掩码的 Transformer 序列基线。由于原论文没有提供 SAINT 版本和参数，这里明确标为风格化实现。
- FC-HGNN：题目为节点、知识点为超边；执行 1–3 层节点—超边—节点传播，并对各尺度加权融合；题目、知识点和仅由训练用户估计的经验难度通过 factorization-machine 二阶交叉融合，再输入 LSTM。训练集中未出现的题目使用中性难度 0.5。

## 数据定义

- ASSIST12：`skill_id` 非空；学习者至少 3 条此类记录；`correct` 必须是 0/1。该规则精确得到论文表 2 的 27,485 人、53,065 题、265 技能和 2,709,436 条交互。
- EdNet：固定种子抽取用户；用 `questions.csv` 的答案生成正确性，用 tags 构造超边。`full.yaml` 使用 5,000 人，但无法恢复原论文未公开的用户清单。
- OULAD：以 `studentAssessment` 为作答序列，分数不低于 40 记为正确；课程模块呈现作为知识点。

三套数据均按用户划分 70%/15%/15% 的训练、验证、测试集合，避免同一学习者跨集合泄漏。所有随机过程使用种子 `20260824`。

## 运行

```bash
cd reproduction
../.venv/bin/python run.py prepare --config configs/smoke.yaml
../.venv/bin/python run.py train --config configs/smoke.yaml
```

完整规模配置：

```bash
../.venv/bin/python run.py all --config configs/full.yaml
```

结果写入 `reproduction/artifacts/<profile>/results/<dataset>/`。每个模型保存独立 JSON，`summary.json` 汇总测试集 Accuracy、Balanced Accuracy、Precision、Recall、F1、MCC、AUC、PR-AUC、Brier 与 ECE。启用 evidential 头时还会记录整体、预测正确和预测错误样本的平均不确定性。

## 前沿增强与顺序消融

`configs/advanced.yaml` 为完整增强实验配置，依次包含时间/遗忘特征、多知识点聚合、题目—知识点对比目标、注意力超图与动态尺度融合、Beta evidential 不确定性头，以及可选的 guided response diffusion 插件。这些模块均可在 YAML 中独立开关。

先用验证集按顺序做累积消融，不在模型选择期间查看测试集：

```bash
../.venv/bin/python ablate.py --dataset assist12 --epochs 8
../.venv/bin/python ablate.py --dataset ednet --epochs 8
../.venv/bin/python ablate.py --dataset oulad --epochs 8
```

顺序为 `baseline -> time -> multi_concept -> contrastive -> attentive_hypergraph -> uncertainty -> guided_diffusion`。结果写入 `artifacts/ablations/<dataset>/`，只有在验证集上证明有效的阶段才应进入最终测试。

快速检查全部增强模块：

```bash
../.venv/bin/python run.py train --config configs/advanced_smoke.yaml
```

完整增强实验：

```bash
../.venv/bin/python run.py train --config configs/advanced.yaml
```

## 与原论文的主要偏差

- 原论文未公开代码、EdNet 抽样种子、数据划分、嵌入维度、隐藏维度、批量大小、训练轮数、超边阈值与正则系数。
- 原论文提到“课程—产业”和经济指标，但公开数据没有这些字段；本实现只验证知识追踪部分。
- `full.yaml` 保留论文给出的学习率 0.00025、三层超图和 Dropout 0.8；其余值是明确记录的工程假设。

## 新数据集双方向实验

从项目根目录运行：

```bash
.venv/bin/python reproduction/test_new_datasets.py all --config reproduction/configs/new_datasets_dual.yaml
.venv/bin/python reproduction/test_new_datasets.py sensitivity --config reproduction/configs/new_datasets_dual.yaml
.venv/bin/python reproduction/analyze_new_datasets.py --root reproduction/artifacts/new_datasets_dual --bootstrap-reps 200
.venv/bin/python reproduction/analyze_static_priors.py --root reproduction/artifacts/new_datasets_dual --bootstrap-reps 200
.venv/bin/python reproduction/analyze_prediction_slices.py --artifact-root reproduction/artifacts/new_datasets_dual
.venv/bin/python reproduction/analyze_tie_sensitivity.py --root reproduction/artifacts/new_datasets_dual
.venv/bin/python reproduction/plot_new_dataset_findings.py --root reproduction/artifacts/new_datasets_dual
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest reproduction/tests -q
```

主实验为三套数据 × 七个变体 × 三个训练种子，共 63 次训练。七个变体为 DKT、AKT、simpleKT、FC-HGNN、增强 FC-HGNN、增强版去超图传播和逆频率加权 FC-HGNN。增强版使用全部叶知识点、时间特征（NIPS 除外）、注意力传播、动态尺度与对比目标；去超图版保留相同输入与目标，仅将超图传播层数置零。其他模型使用第一叶知识点，所以基础与增强版之间的差异包含输入信息变化。

固定数据划分种子为 `20261004`，训练种子为 `20261004/20261005/20261006`。每次最多 8 轮，以验证 AUC 早停，统一 embedding 64、hidden 128、序列长度 100。XES 取 5,000 名学生，保留作者 test、fold 0 valid 和独立片段；其余两套使用所有至少三条有效记录的学生，先按学生划分再处理序列。ASSIST17 排除无具名技能、提示和非主问题，但仍保留重复作答；NIPS 仅使用官方训练表，保留叶知识点，同分钟按 AnswerId 固定排序。数据口径、实际规模和目标数写入各 `prepared/<dataset>/metadata.json`。

所有验证训练完成后先保存 `frozen_plan.json`，再加载最佳 checkpoint 进行测试。阈值从验证集选择，保存固定 0.5、正确类 F1、MCC、Balanced Accuracy、错误类 F1 和预设 5:1 漏错/误报成本阈值。AUC、两类 AP、Brier、ECE 不随阈值改变。加权 BCE 的权重仅取自训练目标频率。

每个种子保存验证和测试目标、概率、学生与事件标识。分析使用学生聚类配对 bootstrap，两个方法及三组固定训练种子共享学生抽样，每次计算种子内差值后取平均。区间条件于当前训练模型和固定数据划分，不包括重新训练、阈值重选或新平台的不确定性。200 次重复用于本轮筛选。

`sensitivity` 在 NIPS 相同分钟内随机重排，另训练 AKT 和基础 FC-HGNN 各一次；学生划分固定。它只能检查结果对排序约定的依赖，不能恢复真实作答时序。本轮本地预算和数据协议用于比较候选方案，不构成论文 SOTA 数值复现。

时间增强版和 noHG 均接收目标事件的 `next_time_gap`，推断时需要已知该事件时间。XES 的时间字段尚未核实是在题目呈现还是答案提交时记录，因此本轮属于事件时间已知条件下的离线预测，不能直接当作答题前实时干预性能。两者的同输入消融仍可比较传播模块，但跨模型差异需连同额外时间信息解读。

结果目录：

Git 仅保留汇总记录与报告图表；训练数据、模型和逐事件预测需在本地生成，详见 [产物范围说明](artifacts/README.md)。

- `artifacts/new_datasets_dual/protocol.json`：配置、源码哈希与范围。
- `screen/`、`checkpoints/`：验证结果与最佳模型。
- `final/`、`predictions/`：最终测试与逐事件概率。
- `analysis/双方向实验报告.md`、`analysis/summary.json`：三种子均值、标准差、配对区间及决策图。
- `analysis/题目先验对照.md`：训练题目经验正确率对照，单份确定预测，自身阈值仅由验证集选择。
- `analysis/切片结果.md`：目标在作者片段内的位置和训练题目曝光切片；保持各种子的原验证阈值，缺失的原始多叶概念列明确不作该切片。
- `analysis/排序敏感性.md`：同学生划分、单种子的 NIPS 并列排序比较。
- `tie_sensitivity/`：NIPS 排序敏感性结果。

运行中断后执行相同命令会复用已完成运行。技术烟雾检查使用 `configs/new_datasets_smoke.yaml`，其结果不能用作研究性能结论。

已完成结果与研究路线判断见 [新数据集双方向实验简报](../汇报材料/新数据集双方向实验简报.md)。完整主实验为 63 次训练、63 组测试，另有 2 次排序敏感性训练；最终 40 项测试通过。训练源码哈希、目标与事件对齐、阈值不变性检查见 [核验记录](artifacts/new_datasets_dual/verification.json)。
