# ASSISTments 2017 原始数据与核验说明

检索、下载与核验日期：2026-10-04（Asia/Shanghai）。

## 为什么适合本项目

本项目研究错误敏感知识追踪及类别不平衡评估。此数据提供学生与题目标识、skill 标签，以及作答起止时间、耗时、提示和支架行为；规模适合作为 ASSISTments 2012–2013 与 EdNet 之外的中等规模基准。skill 没有空字符串，但存在表示未标注技能的 `noskill` 哨兵值，不能据此认为所有记录都有有效知识点；题目数较适合现有题目嵌入模型。2017 是发布/比赛年份，作答实际来自 2004–2007 年，不能当成 2017 年学生作答。

原始全集的错误占比 62.73%，可以检验模型在错误占多数时的行为；其中存在提示行为，必须在论文中区分“学生错误”和“请求提示”。本数据主要用于行为和错误序列分析；它没有随档提供完整题干和知识图谱，中文多知识点、题目语义的主要补充应使用 XES3G5M。

## 文件与完整性

- 原始 ZIP：`raw/anonymized_full_release_competition_dataset.zip`，68,263,324 字节（约 65.10 MiB）。
- ZIP 内唯一 CSV：`anonymized_full_release_competition_dataset.csv`，548,528,619 字节，82 列。
- 已读取全部 ZIP 成员并通过 CRC 校验；未修改、解压或清洗原始制品。
- ZIP SHA-256：`bd8ca6eb7d2c3bc3bc4236754678e516f09748f588377e7eb5b14080e474097d`。可从本数据集目录执行 `shasum -a 256 -c metadata/SHA256SUMS` 核验。
- 逐条 CSV 解析统计：[stats.json](stats.json)。来源与下载清单：[manifest.json](manifest.json)。

## 来源链

- [官方数据集发布页](https://sites.google.com/view/assistmentsdatamining/dataset)：官方常规获取方式是填写邮箱表单，官方没有在该页给出直接公开 ZIP 链接。
- [官方比赛说明](https://sites.google.com/view/assistmentsdatamining/data-mining-competition-2017)：原比赛目标是预测学生以后是否进入 STEM 职业，与知识追踪的逐题正确性目标不同。
- [EduData 官方下载模块](https://edudata.readthedocs.io/en/doc/_modules/EduData/DataSet/download_data/download_data.html)：其资源表将 `assistment-2017` 映射到 USTC 托管文件。
- 本地实际下载：[USTC/EduData 公共学术镜像](https://base.ustc.edu.cn/data/ASSISTment/anonymized_full_release_competition_dataset.zip)。该链接是镜像，不是 ASSISTments 官方无登录下载链接。
- [官方字段说明](https://docs.google.com/spreadsheets/d/1QVUStXiRerWbH1X0P11rJ5IsuU2Xutu60D1SjpmTMlk/edit)：已保存 CSV 快照 `OFFICIAL_COLUMN_DESCRIPTION.csv`；官方发布页/比赛页及可访问历史条款 HTML 也已保存，来源见 `sources.json`。

## 原始统计与标签

| 统计 | 原始全集 | 仅 original=1（主问题） |
|---|---:|---:|
| 交互数 | 942,816 | 249,105 |
| 学生数 | 1,709 | 1,709 |
| 题目数 | 3,162 | 1,183 |
| 非空 skill 标签数（含 noskill） | 102 | 86 |
| 有效具名技能标签数（不含 noskill） | 101 | 85 |
| correct=1 | 351,370 | 107,213 |
| correct=0 | 591,446 | 141,892 |
| 正确率 | 37.27% | 43.04% |
| 错误率 | 62.73% | 56.96% |

`noskill` 共出现 78,103 次（正确 33,696，错误 44,407），涉及 1,038 名学生与 1,071 道题。102 个非空标签中包含这 1 个未标注哨兵值，因此有效具名知识点为 101 个。题目在不同记录中可能有不同标注，不能从总体题目数直接减去上述 1,071 得到具名题目数。

下表是只读统计得到的具名技能视角，没有生成清洗后数据：

| 过滤视角（均排除 noskill） | 交互数 | 学生数 | 题目数 | 具名技能数 | 正确率 | 错误率 |
|---|---:|---:|---:|---:|---:|---:|
| 其余原始交互全部保留 | 864,713 | 1,709 | 2,210 | 101 | 36.74% | 63.26% |
| 再保留 hint=0 | 570,221 | 1,709 | 2,208 | 101 | 55.71% | 44.29% |
| 再保留 hint=0 且 original=1 | 183,973 | 1,708 | 755 | 85 | 50.93% | 49.07% |

不计 `hint=1` 的非提示交互共 630,720 条，正确率 55.71%、错误率 44.29%；同时保留主问题（`hint=0, original=1`）则共 207,728 条，正确率 51.61%、错误率 48.39%。这些仅是行为过滤视角，不是已经确认的严格首答数据。另有 18 条 `hint=1, correct=1` 的不一致组合，原始档保持不变，后续清洗需明确如何处理。

数据实际标签列为 `correct`，不存在 `score` 列。已逐值核验：942,816 条记录全部使用字符串 `0` 或 `1`，没有小数、其他取值或缺失；不能把学生平均正确率 `AveCorrect`、BKT 估计 `Ln` 等当作真实答题标签。学生、题目、correct、startTime、endTime、action_num 均无空值；skill 无空字符串，但有前述 `noskill` 未标注哨兵值。所有 action_num 唯一，无重复动作 ID；这不意味着学生不会多次作答同一题。原始学生序列长度中位数为 441，范围 2–3,057。

## 后续处理建议（未执行清洗或训练修改）

1. 字段映射：`studentId → user_raw`，`problemId → item_raw`，`skill → concepts_raw`，`correct → correct`，`startTime → timestamp`。官方说明使用 `ITEST_id`/`actionId`，镜像实际表头为 `studentId`/`action_num`；应按实际 CSV 表头读取。时间戳单位是 UNIX 秒，需统一到项目采用的时间单位。
2. 原始行序在同一学生内部有 2,554 次 startTime 倒序，另有 8,176 对相邻记录 startTime 相同。应按学生分组、数值 startTime 排序，同一时间以 action_num 数值和原始行序保持稳定；不要把原始文件行序直接视为严格时间序列。
3. `skill` 是知识点文字标签；排除 `noskill` 后的 101 个具名标签可作为概念 ID 和语义描述。不要未经核验把连字符或冒号当多技能分隔符。知识点实验应明确选择去除 `noskill` 记录，或将它保留为单独的未知概念；若保留，不能将该哨兵值计为有实际语义的知识点，需分别报告未知比例。原始 102 个非空标签的统计包含 `noskill`。
4. 这是动作/交互日志，包含 312,096 条 `hint=1` 记录和 693,711 条 `original=0` 支架问题记录。官方 `correct` 描述为“回答正确”，不是明确的“第一次提交正确”。若论文目标是严格首答预测，应先定义主问题与重复作答的聚合规则，用 assignmentId/assistmentId/problemId、动作和时间区分题目机会，再验证首答标签。简单过滤 `hint=0` 或 `original=1` 并不等于严格首答。原始全集与过滤版本应分别报告，不能将二者指标混用。
5. 预测下一次作答时可使用此前已完成交互的 correct、耗时和提示等信息；预测本次作答时不要使用本次结果/提示/耗时。不要直接把 `AveCorrect`、`AveKnow`、`Ln`、`timeGreater10SecAndNextActionRight`、`endsWithScaffolding` 等聚合、标签派生或未来相关字段输入模型；历史统计应仅从过去记录重算。
6. 按学生划分训练/验证/测试集，并保留时间顺序。题目难度、类别权重、采样规则等只用训练集估计。错误类建议设为正类，报告错误类 PR-AUC、召回率、F1、平衡准确率，并配合 ROC-AUC；记录具体提示/支架/首答处理方案和类别比例。
7. 下载核验时仅新增原始数据和核验材料；此后已以 `assist17` 接入统一训练入口，当前过滤口径与协议见 [实验说明](../../../reproduction/README.md#新数据集双方向实验)。

## 使用条款（不是 CC 许可）

2017 官方页说明，下载或使用即接受其 Terms of Use；其特定旧链接 `http://www.assistmentstestbed.org/the-data/termsofusefordata` 在本次检查中无法访问（HTTP 502，HTTPS TLS 握手失败），因此不能声称已经取得该版本完整文本，也不能假设属于 CC-BY、CC0 或 unrestricted license。

已保存可访问的[官方历史 TestBed 数据条款](https://sites.google.com/site/assistmentstestbed/4-analyze-data/terms-of-use-for-data)为 `OFFICIAL_LEGACY_TESTBED_DATA_TERMS.html`。该页面要求不尝试识别学生身份；发现可识别信息应删除并通知 ASSISTments；不向第三方再分发数据；不作商业或恶意用途。另保存[官方 ASSISTmentsData 使用条款](https://sites.google.com/site/assistmentsdata/termsofuseforusingdata)为 `OFFICIAL_ASSISTMENTSDATA_TERMS.html`，其还要求按声明目的使用、在发表时致谢，并公开所发表工作的算法/代码。这两份材料是同一官方体系中可访问的历史参考，并非已证明逐字等同于 2017 失效链接的存档。

本地制品仅用于本项目非商业研究，不应将 ZIP 或包含学生交互的派生表再上传公开仓库或转发第三方。发表时应引用 ASSISTments、2017 发布页和所用镜像/制品哈希，并核对相应研究条款。当前 E-TRIALS 的新条款不是已核验的 2017 版本，本说明没有将两者直接等同。
