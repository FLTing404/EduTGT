# XES3G5M 官方发布数据与核验说明

下载与核验日期：2026-10-04（Asia/Shanghai）。

## 本项目的适配判断

这是本轮最推荐的新增主基准。它直接提供中文数学逐题正确性、学生和题目 ID、多知识点、时间、题干、解析与知识点路线，能同时支持当前错误敏感评估和后续超图/语义分析。相较于 OULAD 的课程评测分数阈值与课程模块，它更接近标准细粒度知识追踪；相较于已有 ASSIST12、EdNet，它补充中文数学与知识点结构。

该判断基于任务和字段匹配，并不意味着 FC-HGNN 会获得更高指标。下载核验时尚未接入训练；此后已接入统一入口，当前配置与协议见 [实验说明](../../../reproduction/README.md#新数据集双方向实验)。

## 下载来源与许可

- [作者官方仓库](https://github.com/ai4ed/XES3G5M)。来源说明已固定到提交 `b5a35caf223c5c5d558da2fb836087fa6af2bef5`，本地保存为 `UPSTREAM_README.md`。
- [作者官方 Google Drive 文件](https://drive.google.com/file/d/1eFiIYyh5O2V90RA0brammGH6EpHvPDQe/view)，文件名为 `XES3G5M.tar.gz`。本次直接下载该发布包，未使用第三方加工版本。
- [NeurIPS 2023 数据集论文](https://papers.neurips.cc/paper_files/paper/2023/hash/67fc628f17c2ad53621fb961c6bafcaf-Abstract-Datasets_and_Benchmarks.html)。论文统计为 18,066 名学生、7,652 道题、865 个叶知识点与 5,549,635 条原始交互。
- 数据许可为 MIT；论文第 3.1 节及[官方补充材料](https://papers.nips.cc/paper_files/paper/2023/file/67fc628f17c2ad53621fb961c6bafcaf-Supplemental-Datasets_and_Benchmarks.pdf)均明确将 MIT 用于数据集。已保存作者 `LICENSE`，使用时保留版权和许可，并引用数据论文。

## 制品完整性

- 文件：`raw/XES3G5M.tar.gz`。
- 压缩大小：373,235,093 字节，约 355.94 MiB。
- SHA-256：`62d145bd995248f78726a0b6ab69612cf418e3118d0fe01bfe6f4c61b5072f73`。
- 已读取整个 gzip 流至尾部并通过 gzip CRC/长度检查，同时遍历 TAR 的全部 3,886 个成员。
- 全部普通文件解压后的内容合计 8,642,670,992 字节，约 8.05 GiB。滑动窗口测试文件含大量重复历史，体积很大；当前保留完整压缩包，没有全量解压。
- 初次传输尾部提前结束，大小核验发现后通过 HTTP 206 Range 补齐缺失的 16,482,401 字节；最终大小及完整性检查全部通过。清单见 `download_manifest.json`，成员列表见 `archive_members.json`。

## 本地只读统计

以下统计针对官方包内两个题目级文件；已排除 `questions=-1` 的填充 token，要求真实题目的响应为 0/1。具体值见 [stats.json](stats.json)。

| 项目 | train_valid_sequences_quelevel.csv | test_quelevel.csv |
| --- | ---: | ---: |
| 文件行数/序列片段数 | 30,965 | 3,613 |
| 不同学生 | 14,453 | 3,613 |
| 非填充二值作答 token | 4,446,374 | 1,102,810 |
| 正确 / 错误 token | 3,532,679 / 913,695 | 877,369 / 225,441 |
| 正确率 | 79.45% | 79.56% |
| 文件涉及题目 | 7,618 | 7,052 |
| 文件涉及叶知识点 | 865 | 828 |
| 多叶知识点作答 token 比例 | 12.52% | 12.49% |

两个文件合计 18,066 名学生、5,549,184 个非填充作答 token、7,651 个序列涉及题目；训练/测试学生交集为 0。合计 token 正确率为 79.47%，错误率 20.53%，约 3.87:1。没有非法响应或真实题目对应空概念的记录。所有序列题目 ID 和叶知识点 ID 均能匹配题目/知识点元数据。

时间专项核验发现：测试文件有 1 名学生的最前 7 个 token 时间落在 2001-01-06，其余训练/测试作答时间不早于 2018 年；这些早期值确实来自原档，后续时间间隔实验需单独定义异常处理。测试序列内相邻时间倒序为 0，但同时间相邻对有 569,744 个，涉及全部 3,613 名测试学生。应保留作者给出的序列顺序，不要用题目 ID 重新排列同时间作答；UNIX 毫秒编码不代表事件都能以毫秒精度区分。

题目元数据实际有 7,652 条：6,142 道填空题、1,510 道单选题，包含 `content, kc_routes, answer, options, analysis, type`。知识点路线映射有 1,175 个节点项，序列涉及其中 865 个叶知识点。

**这些是所审作者处理文件的统计，不能当成未处理原始日志统计。** 训练数据被切分、填充，过短尾段等处理会改变可统计的事件；测试文件格式也不同。7,651 个序列题目与元数据 7,652 个题目的差异、5,549,184 个 token 与论文 5,549,635 条原交互的差异，均不能据此判断下载不完整。作者的 [Clarification](https://github.com/ai4ed/XES3G5M#clarification) 已说明：知识点展开、不同截断方式与缺少唯一事件 ID，使完整恢复原始交互困难。

## 文件与后续使用

| 用途 | 包内路径或字段 | 注意事项 |
| --- | --- | --- |
| 优先训练/验证输入 | `XES3G5M/question_level/train_valid_sequences_quelevel.csv` | `uid` 是学生，同一学生有多个片段；不能按 CSV 行随机分割学生 |
| 原测试序列 | `XES3G5M/question_level/test_quelevel.csv` | 不含 selectmasks；实际下一次作答目标数由后续评估协议决定，不能把所有 token 自动视为预测目标 |
| 题目 ID | `questions` | 逗号分隔的序列，-1 为填充 |
| 知识点 | `concepts` | 逗号分隔的序列；同题多知识点用下划线分隔，适配现有代码时须转换为约定格式 |
| 正确性 | `responses` | 1 正确、0 错误；排除填充，预测目标不可进入历史 |
| 时间 | `timestamps` | UNIX 毫秒；做时间/遗忘实验前需核验异常年份、逆序及并列记录 |
| 训练掩码 | `selectmasks` | 忽略 -1 位置，保留真实事件和作者协议 |
| 中文题干/解析/题型 | `XES3G5M/metadata/questions.json` | 题型实际使用中文字符串“填空”“单选”，并非 README 示例中的数值编码 |
| 知识点路线 | `XES3G5M/metadata/kc_routes_map.json` 和 questions.json 中 kc_routes | 区分层级祖先与多个独立叶知识点 |
| 可选语义与视觉信息 | `metadata/embeddings/`、`metadata/images/` | 已完整包含在官方压缩包中；当前模型尚未使用 |

`kc_level/` 数据把同题多知识点展开为连续行，使用它时必须处理 `is_repeat` 和题目级融合，避免同一次作答的标签泄漏。`*_window_*` 文件有重复历史，不能与原测试序列拼成更多独立样本。建议先用题目级序列与官方按学生的测试集，再明确本项目的训练/验证划分、下一题预测及掩码规则。

从本数据集根目录可执行 `shasum -a 256 -c metadata/SHA256SUMS` 复核下载文件。`metadata/audit_archive.py` 可重新做只读全流和题目级统计；运行目录为项目根目录。该脚本不生成清洗后数据，也不修改原发布包。
