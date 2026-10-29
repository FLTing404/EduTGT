# EdNet KT1 原始数据

## 文件

- `raw/EdNet-KT1.zip`：784,309 名用户的 KT1 交互 CSV
- `raw/questions.csv`：13,169 道题的题目、答案、part 与技能标签元数据

## 来源链

- 官方仓库：https://github.com/riiid/ednet
- 官方 KT1 链接最终指向 Google Drive 文件 ID：`1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw`
- 官方 Contents 链接最终指向 Google Drive 文件 ID：`117aYJAWG3GU48suS66NPaB82HwFj6xWS`
- KT1 本地下载镜像：https://base.ustc.edu.cn/data/EdNet/EdNet-KT1.zip
- `questions.csv` 可追溯副本：https://gist.github.com/LeaveNhA/fbb8c6ce2a6422fab97f66ffdb7a7852

官方 Google Drive 在当前网络环境中连接超时，因此使用公开镜像。KT1 ZIP 共 784,310 个条目（其中一个为 `KT1/` 目录），压缩数据全量 CRC 检查通过，并记录 SHA-256。

## 许可

官方仓库声明 EdNet 按 Creative Commons Attribution-NonCommercial 4.0 International（CC BY-NC 4.0）发布，仅用于研究等非商业用途。

## 与论文的关系

论文表 2 并非使用全部 EdNet，而是使用 5,000 名学习者的随机子集。其 12,161 道题、189 个技能、676,974 条交互等统计与《Graph Knowledge Tracking Interaction Model》表 2 一致，但该来源没有公开随机种子或所选用户 ID。因此，本地保存完整 KT1，以便后续建立明确种子的可复现抽样。

完整 Contents ZIP 的 lectures、payments、coupons 三张表当前未下载；知识追踪复刻所需的 `questions.csv` 已取得。KT2、KT3、KT4 不属于本论文所述 KT1 评估范围。

