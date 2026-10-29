# 教育数据集原始文件清单

检索与下载日期：2026-09-19（Asia/Shanghai）

最近新增与核验：2026-10-04（Asia/Shanghai）。选型结论见 [数据集选型建议](数据集选型建议_2026-10-04.md)。

本地工作区保存原始发布制品或可追溯镜像，不保存清洗后数据。Git 只上传来源说明、许可、校验和、统计及下载审计脚本，`raw/` 不上传；克隆后需按各数据集说明补齐原始文件。每个数据集采用相同结构：

- `raw/`：下载得到的不可变原始文件
- `metadata/README.md`：来源、许可、完整性与论文复刻说明
- `metadata/SHA256SUMS`：本地制品的 SHA-256 锁定值

当前数据：

- `ASSISTments2012_2013`：完整 2012–2013 原始 ZIP
- `EdNet`：完整 EdNet-KT1 ZIP，以及知识追踪所需的 `questions.csv`
- `OULAD`：Figshare v1 的完整 `anonymisedData.zip`
- `XES3G5M`：作者官方完整 `XES3G5M.tar.gz` 发布包，包含已处理的题目级/知识点级序列、中文题目与解析、知识点路线、图片与嵌入；新增主基准首选。
- `ASSISTments2017`：USTC/EduData 可追溯镜像的完整原始 ZIP；包含提示和支架行为，标签处理需明确。
- `NIPS2020_Task34`：Eedi 官方竞赛 ZIP 的全部 10 个 Task 3/4 CSV 成员及对应元数据子集，保持原始成员字节；未下载 Task 1/2、题图或完整竞赛 ZIP。

新增三套数据均已完成压缩包/成员完整性校验和 SHA-256 记录，并在各自 `metadata/` 中保存全量统计、来源与使用条款。三套数据已接入 `reproduction` 的统一训练入口；数据名称为 `xes3g5m`、`assist17`、`nips34`，双方向实验的协议和运行方式见 [实验说明](../reproduction/README.md#新数据集双方向实验)。清洗后的数据和实验结果单独保存于 `reproduction/artifacts/`。

注意：EdNet 官方还发布 KT2、KT3、KT4 和包含 4 张表的 Contents ZIP；当前实验属于知识追踪任务，所需层级是 KT1，且只依赖 Contents 中的题目元数据，因此未下载 KT2–KT4。
