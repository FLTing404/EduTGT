# 实验记录的上传范围

此目录在本地还包含训练数据、逐事件预测和模型权重。Git 仅保留不含学生映射或逐事件数据的 JSON 汇总、Markdown 报告和 PNG 图表；`prepared/`、`sensitivity_prepared/`、`preprocess_qa/` 及所有 `analysis/*.json` 均被排除。

分析 JSON 的部分文件嵌入完整学生映射，因此不发布。可阅读 [双方向实验报告](new_datasets_dual/analysis/双方向实验报告.md)、[题目先验对照](new_datasets_dual/analysis/题目先验对照.md)、[切片结果](new_datasets_dual/analysis/切片结果.md) 和 [排序敏感性](new_datasets_dual/analysis/排序敏感性.md)。指标 JSON、运行协议及验证/测试选择记录保留，便于追溯报告中的数字。

这些是既有运行的历史记录，整理时没有重新训练或改写数值。部分路径指向当时的本地 checkpoint、预测或预处理文件；这些路径不表示对应文件随仓库发布。`verification.json` 内的分析哈希针对当时完整的本地产物，未发布的分析 JSON 需按 [实验说明](../README.md#新数据集双方向实验) 重新运行生成后核验。

`*_smoke` 记录仅证明小规模流程曾运行，不能用于研究性能结论。重做分析需要先补齐原始数据、运行训练和测试，生成逐事件预测；仅克隆这些汇总记录不足以重算学生聚类区间。
