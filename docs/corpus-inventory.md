# CCLUE 古文标点数据核验

日期：2026-10-15（Asia/Shanghai）。上游 [Ethan-yt/CCLUE](https://github.com/Ethan-yt/CCLUE)，固定提交 `ed481c453c1c36397f86905a2bda625ecc5607e1`。下载八个官方文件，严格 UTF-8 解码，原始字节不改变，不执行上游代码。

## 格式与原生标签

train/dev/test 为五列 CoNLL：character、seg_tags、punc_tags、quote_tags、book_tags。空行分隔记录，DOCSTART 分隔匿名源文档。官方构建 notebook 直接将标点写入前一个字符的 punc 标签，且 seg=B 当且仅当 punc 非 O。

| ID | 原生 punc 标签 | 输出 |
| --- | --- | --- |
| 0 | O | 不插入 |
| 1 | B-, | ， |
| 2 | B-. | 。 |
| 3 | B-? | ？ |
| 4 | B-! | ！ |
| 5 | B-\ | 、 |
| 6 | B-: | ： |
| 7 | B-; | ； |

读取器逐行检查五列、单字符、已知标签及 seg/punc 一致性，遇到错误立即失败，不静默忽略。quote/book 标签保留但不用于本轮训练。训练正文不做规范化，不重新推断标点，不将 DOCSTART 或空记录当作正文。

## 数量及重复审计

官方 README 标示训练/开发/测试为 26,935/4,075/3,992；固定提交中解析出的非空记录为 26,932/4,074/3,992。差异如实记录，不补造样本。

| 集合 | 官方文件非空记录 | 实际使用记录 | 字符数 | 集合内保留重复正文 |
| --- | ---: | ---: | ---: | ---: |
| train | 26,932 | 26,890 | 2,278,777 | 0 |
| validation | 4,074 | 4,074 | 341,563 | 2 |
| test | 3,992 | 3,992 | 417,786 | 2 |

训练文件原有 31 条重复正文记录；它与开发、测试分别有 6、5 个共同正文哈希。先排除与任一留出集相同正文的训练记录，再去除其余训练内部重复，最终移除 16+26=42 条。两项数量有交叠，不可直接把原始重复统计相加。开发和测试不互相重复，内部各有 2 条重复，保留原有记录和顺序。

只使用留出集正文哈希做泄漏检查，不根据留出标签调参。上游 notebook 按类别中的文件进行划分，但发布文件没有原始书名，DOCSTART 只给出匿名组。因此保留官方划分不代表已独立证明全部书籍版本无重叠；当前也没有执行近重复段落检索。GuwenBERT 预训练数据与 CCLUE 有来源关联，报告须说明预训练重叠风险。

## SHA-256

```text
ec5c18082475ec786e05eb0e168092ac767e3192ba984409af82ec97a33b3913  data/punctuation/train.txt
16d12a47b4de7685bfa756e035fe31132715730becb7a221e3f2d9d9edb9c282  data/punctuation/dev.txt
b364aee5a5ee03817aad25f0553f024b2b416b8cfdf16cc8af394efd50652ab1  data/punctuation/test.txt
1238b7a6b9cc800352abff936e869c56aae63d49f6ee2ad80385c5366de9975c  data/punctuation/punctuation.py
d7fbb5d07ec644451222f043481d9e2d87a35eb61446aca5040fd16efb575a65  data/punctuation.ipynb
539c7fde8fd5b0a80bd62523c36d4d381943c5e156409df23283beb8b6758a4b  README.md
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  LICENSE
0504a0c605e202b3f4010f3f2efafe17c8465f057e40c05b981ad2fae06357fb  run_sequence_labeling.py
```

完整路径、大小及下载链接见 [data/manifest.json](../data/manifest.json)。派生数据版本 `cclue-punc-v2`，划分校验值和移除记录清单位于本地 data/processed、data/splits；精简统计纳入实验报告。原始仓库与基座模型许可和本项目 MIT 分开说明，数据本体不提交 Git。
