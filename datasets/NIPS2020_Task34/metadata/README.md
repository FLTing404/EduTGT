# NIPS2020 / Eedi Task 3/4 原始 CSV 子集

下载日期：2026-10-04（Asia/Shanghai）。来源为 [Eedi 官方竞赛页](https://www.eedischool.com/projects/neurips-education-challenge) 链接的 [官方 ZIP](https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip)。

这是官方竞赛包的 **Task 3/4 CSV 子集**，不是完整竞赛包。共保存 10 个原始成员，包含训练、Task 3 题质 public/private 测试、Task 4 public/private 测试，以及对应元数据和共用知识点树；未下载 Task 1/2、题目图片、starter kit 或 macOS 辅助文件。`raw/data/` 后面的路径与官方 ZIP 成员路径一致，所有 CSV 字节未改动。

官方整包为 656,787,242 bytes（626.36 MiB）。本次按 HTTP Range 提取所选成员，压缩数据量为 33,191,571 bytes（31.65 MiB；不含请求头、ZIP 目录等少量开销），解压后原始 CSV 为 111,743,284 bytes（106.57 MiB）。每个成员通过官方 ZIP 中央目录 CRC32 校验，另有 SHA-256 清单。

## 全量核验结果

以下统计只使用 `raw/data/train_data/train_task_3_4.csv`，时间通过 `AnswerId` 与作答元数据连接。完整结果见 [stats.json](stats.json)。

| 项目 | 实测结果 |
| --- | --- |
| 训练作答数 | 1,382,727 |
| 训练学生数 | 4,918 |
| 训练题目数 | 948 |
| 正确 / 错误 | 742,983 / 639,744 |
| 正确率 / 错误率 | 53.7332% / 46.2668% |
| 每生作答数：最少 / 中位数 / 最多 | 50 / 239 / 827 |
| 训练作答日期 | 2019-09-06 14:05 至 2020-05-01 08:33；原字段未声明时区 |
| 作答时间缺失、题目元数据缺失、学生元数据缺失 | 均为 0 |
| 重复学生—题目记录 | 0 |
| `IsCorrect` 与 `AnswerValue == CorrectAnswer` 的矛盾数 | 0 |
| 同一学生同一分钟并列的额外记录 | 666,111，占训练记录约 48.17% |
| 题目元数据 / 学生元数据 / 作答元数据行数 | 948 / 6,148 / 1,508,917；包含官方测试学生及其元数据 |
| 共用知识点树节点 / 本组题目涉及原始主题数 | 388 / 86 |
| 本组题目涉及叶知识点数 | 57 |
| 多叶知识点题目 | 14 / 948；其余 934 题只有 1 个叶知识点 |

## 对本项目的适用性

相较于以课程活动、考核得分和结课结果为主的 OULAD，本数据直接给出学生—题目—正确性作答记录、作答时间和题目主题层级，映射到 FC-HGNN 的知识追踪输入更直接。题目主题树能支持层级图建模；`AnswerValue` 和 `CorrectAnswer` 还可用于研究错误选项与误概念相关行为。规模适合本地先完成全量对照实验。

本组正确/错误比例接近平衡，因此建议作为 **接近平衡对照和错误诊断补充**，与 ASSISTments、EdNet 或中文 XES3G5M 的结果比较，不应把它宣传为严重类别不平衡数据。它的原始 `SubjectId` 多标签多数表示知识点层级路径，不能直接宣称每道题考察多个独立叶知识点。中文和丰富多知识点覆盖可优先由 XES3G5M 提供。

下载核验时仅处理原始制品；此后已以 `nips34` 接入统一训练入口，当前同分钟排序约定与敏感性分析见 [实验说明](../../../reproduction/README.md#新数据集双方向实验)。

## 字段映射

| 知识追踪用途 | 原始文件与字段 | 注意事项 |
| --- | --- | --- |
| 学生 ID | 训练表 `UserId` | Task 1/2 与 Task 3/4 独立匿名化，不能跨组连接 |
| 题目 ID | 训练表 `QuestionId` | 与题目元数据按同名键连接 |
| 作答连接键 | 训练表 `AnswerId` | 连接作答元数据的唯一键 |
| 二元正确性标签 | 训练表 `IsCorrect` | 1 为正确，0 为错误；只有历史正确性可作为预测输入 |
| 历史选项 / 正确选项 | 训练表 `AnswerValue` / `CorrectAnswer` | 取值 1–4；不能把待预测作答的 `AnswerValue` 输入模型 |
| 作答时间 | 作答元数据 `DateAnswered` | 精度为分钟；不能仅依赖此字段恢复同分钟的真实先后顺序 |
| 题目知识点标签 | 题目元数据 `SubjectId` | 字符串形式的 ID 列表；需要解析，含祖先主题 |
| 知识点名称与层级图 | 共用 `subject_metadata.csv` 的 `SubjectId,Name,ParentId,Level` | 原文件含 UTF-8 BOM；读取时可用 `encoding="utf-8-sig"` |
| 可选作答上下文 | 作答元数据 `Confidence,GroupId,QuizId,SchemeOfWorkId` | 一些字段有缺失；回答后得到的信息不得泄漏到待预测目标 |
| 官方测试选择标记 | Task 4 测试表 `IsTarget` 或 `IsTarget_0` 至 `IsTarget_9` | 官方当前 ZIP 已公开正确性标签；按原任务遮罩评估，或明确另建本地切分 |

## 评估限制

作者的 [官方指南第 2 节](https://arxiv.org/pdf/2007.12061) 说明，重复回答同一题时只保留最后一次；此前的尝试不在本发布版本中。官方 Task 3/4 主要面向题质评价和主动选择题目，不是标准的连续下一次作答 KT 切分。新 KT 实验需要显式规定按学生或按时间划分，避免随机打散作答导致未来信息进入历史，并明确同分钟并列的处理方式。

`test_private_task_4.csv` 与 `test_private_task_4_more_splits.csv` 是同一批测试作答的不同遮罩版本，不能拼接为独立样本，也不能并入训练。本次按原字节保留所有这些文件。

## 文件、复现与许可

`download_manifest.json` 与 `selected_zip_members.json` 记录官方链接、所选成员、CRC32、压缩/原始大小和本地路径。`SHA256SUMS` 覆盖 10 个原始 CSV，校验命令应从本数据集根目录执行：

```bash
shasum -a 256 -c metadata/SHA256SUMS
python metadata/analyze_dataset.py
```

下载脚本为 `metadata/download_task34.py`（依赖 `requests`），统计脚本依赖 `pandas`。重新运行下载脚本时会复用已通过 CRC 校验的文件，不重新下载这些成员。

数据许可为 **CC BY-NC-ND 4.0**。需要署名，禁止商业使用，不得分发改编后的数据。原始数据只解压，不清洗改写；本地研究用的统计和后续预处理应继续遵守原许可。许可来源摘要与官方引用见 [LICENSE_SOURCE.md](LICENSE_SOURCE.md)，官方来源页快照见 `official_source_page.html`。题图另有训练用途及禁止打印、分享的限制，本次没有下载。
