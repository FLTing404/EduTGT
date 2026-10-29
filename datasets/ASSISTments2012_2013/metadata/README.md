# ASSISTments 2012–2013 原始数据

## 文件

- `raw/2012-2013-data-with-predictions-4-final.zip`
- ZIP 内含一个 CSV：`2012-2013-data-with-predictions-4-final.csv`
- CSV 解压大小：3,009,494,391 字节

## 来源链

- 论文给出的发布页：https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect
- 发布页指向的 Google Drive 文件 ID：`1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY`
- 本地下载镜像：https://base.ustc.edu.cn/data/ASSISTment/2012-2013-data-with-predictions-4-final.zip
- 镜像由中国科学技术大学 EduData 项目公开维护，文件名与官方发布制品一致。

官方 Google Drive 在当前网络环境中连接超时，因此使用上述公开镜像。ZIP CRC 全量检查通过，并记录 SHA-256。

## 原始数据核验

- 原始记录：6,123,270
- 原始唯一学习者：46,674
- 原始唯一题目：179,999
- 唯一技能 ID：265
- 无技能 ID 的记录：3,411,457

论文表 2 的统计不是原始全集统计。通过本地核验可以精确匹配其来源统计：先保留 `skill_id` 非空记录；按这些记录统计每名学习者的交互数并保留至少 3 条交互的学习者；最后保留 `correct` 为 `0` 或 `1` 的记录。结果恰为 2,709,436 条交互、27,485 名学习者、53,065 道题和 265 个技能。该规则是根据原始数据与论文统计反推得到的，论文正文没有明示。

## 使用注意

CSV 含带换行的带引号字段，不能用简单的逐行文本方法可靠计数；清洗时应使用符合 RFC 4180 的 CSV 解析器。原始 ZIP 应保持不变，后续派生文件应写入独立的 `processed/` 目录。
