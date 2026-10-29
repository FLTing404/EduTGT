# Eedi 数据许可来源

核验日期：2026-10-04（Asia/Shanghai）。

官方来源：[NeurIPS 2020 Education Challenge](https://www.eedischool.com/projects/neurips-education-challenge)。该页的「Download Eedi Dataset」指向本次下载所用的 [Azure 官方 ZIP](https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip)。

官方数据许可为 [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/)（CC BY-NC-ND 4.0）。官方页面要求署名、链接许可并说明变更；禁止商业用途；对数据进行改编后，不得分发改编版本。官方还单独规定题目图片仅用于模型训练，不能打印或分享。本次没有下载题图。

本目录保存的是官方包中 Task 3/4 的原始 CSV 成员：解压只还原原始字节，没有清洗、重编码、去重、合并或修改这些文件。每个成员均与官方 ZIP 中央目录 CRC32 核验，并提供 SHA-256。`stats.json` 是本地只读统计结果，训练数据尚未接入项目。后续如生成清洗后的训练集，应保留在本地研究环境并遵守原许可，不能把修改后的数据集当成可公开发布制品。

`official_source_page.html` 保存下载时的官方来源页；`download_manifest.json` 记录来源、包大小、下载方法与选取范围。

引用信息（官方建议）：

```bibtex
@article{wang2020diagnostic,
  title={Diagnostic questions: The neurips 2020 education challenge},
  author={Wang, Zichao and Lamb, Angus and Saveliev, Evgeny and Cameron, Pashmina and Zaykov, Yordan and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Turner, Richard E and Baraniuk, Richard G and Barton, Craig and Jones, Simon Peyton and Woodhead, Simon and Zhang, Cheng},
  journal={arXiv preprint arXiv:2007.12061},
  year={2020}
}
```

字段与采集方式参考作者的 [官方竞赛指南](https://arxiv.org/pdf/2007.12061)，尤其是第 2 节。
