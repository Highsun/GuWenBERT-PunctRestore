# 任务一 v2：通用 BERT 的古文断句标点微调

主实验使用 `google-bert/bert-base-chinese`，对照使用 `ethanyt/guwenbert-base`。两者都从预训练编码器重新微调，不使用现成标点模型；共享同一 CCLUE 划分、标签、训练超参数和 seed。当前代码版本不再支持旧语料解析。最终结果见 [实验总报告](reports/experiment.md)。

## 独立环境与代码

Python 3.12，uv/venv；本任务自己的 `pyproject.toml`、`uv.lock`、`.venv/`，不修改任务二依赖。uv 缓存放 `.cache/uv/`，Hugging Face 缓存放 `.cache/huggingface/`。源码位于 `src/guwenbert_task1/`，这个内部包名表示项目任务一，不限定所用基座。

```bash
cd /home/highsun/code/GuWenBERT-PunctRestore/code/task1
uv sync --locked
uv run --locked pytest -q
```

PyTorch 使用官方 CUDA 12.8 wheel。GPU 命令需要可访问 WSL GPU 的终端，CUDA 不可用时明确失败，不以 CPU 或虚假日志代替训练。下载需要网络；缓存齐全后可用 `--offline`。

## 数据与主实验

```bash
uv run --locked python scripts/run.py fetch-data
uv run --locked python scripts/run.py prepare
uv run --locked python scripts/run.py download
uv run --locked python scripts/run.py train
uv run --locked python scripts/run.py verify
uv run --locked python scripts/run.py evaluate
uv run --locked python scripts/run.py report
uv run --locked python scripts/run.py predict --text '学而时习之不亦说乎有朋自远方来不亦乐乎'
```

默认配置 `configs/cclue_bert.toml`，固定数据和模型提交，主实验标识 `cclue-bert-base-v2`。`fetch-data` 下载八个固定上游文件并记录 SHA-256；`prepare` 验证来源、严格解析原生五列 CoNLL，保留官方开发与测试集，清理训练重复正文。没有随机重新划分，也不对训练正文进行标点删除或标签重构。

实际训练/开发/测试为 26,890/4,074/3,992 条。原始非空数量与上游 README 稍有不同，见 [数据核验](../docs/corpus-inventory.md)。匿名 DOCSTART 用于追踪源文档组；记录标识字段 `article_id` 是兼容内部编码器的唯一键，表示上游片段，不代表完整文章或已知书名。

## 古文基座对照

```bash
uv run --locked python scripts/run.py download --config configs/cclue_guwenbert.toml
uv run --locked python scripts/run.py train --config configs/cclue_guwenbert.toml
uv run --locked python scripts/run.py verify --config configs/cclue_guwenbert.toml
uv run --locked python scripts/run.py evaluate --config configs/cclue_guwenbert.toml
uv run --locked python scripts/run.py report --config configs/cclue_guwenbert.toml
```

对照标识 `cclue-guwenbert-v2`，单独保存模型、来源收据、训练记录和测试结果。两个 `report` 完成后自动生成总报告，校验数据和训练设置一致。只用开发集选择各自最佳 epoch；测试集不参与选 checkpoint 或后续调参。单种子比较不能证明统计显著性，词表和预训练过程的差异也须保留解释空间。

## 标签与训练

八类标签顺序遵循官方 punc_tags：`O`、`，`、`。`、`？`、`！`、`、`、`：`、`；`。标点附着在字符后；quote_tags/book_tags 保留供审计，不训练引号和书名号。训练输入保持原生 tokens，不做旧版的规范化、连续标点合并或随机文章划分。

每个字符独立经 tokenizer 编码，监督第一个 subtoken；后续 subtoken、CLS/SEP、padding 标签为 -100。空编码用 UNK 占位。按实际 token 数切窗，最大 256 token（含两个特殊 token），不重叠、不静默截断，保留窗口尾的原生标签。BERT 的有效位置上限为 512，当前 RoBERTa 的 514 配置包含偏移，有效上限同样为 512。

两个实验均为 batch=32、评测 batch=64、BF16、AdamW、学习率 3e-5、weight decay=0.01、warmup=0.1、线性衰减、梯度裁剪 1.0、3 epochs、seed=42。无类别加权或重采样，未承诺 CUDA 逐位确定性。

标点 micro/macro-F1 排除 O 作为正类；macro 固定七类等权，零分母计 0。`segmentation_any_punctuation` 将全部七类合并，与 CCLUE seg_tags 一致；`sentence_boundary` 仅合并 `。？！`。含 O 准确率及全 O 基线单列，不混称各种边界指标。

## 产物与冻结

路径相对于项目根 `code/`：

| 产物 | 路径 |
| --- | --- |
| 原始数据 | `data/raw/cclue/<commit>/` |
| 派生审计与划分 | `data/processed/task1/cclue-punc-v2/`、`data/splits/task1/cclue-punc-v2/` |
| 主模型 / 对照模型 | `models/task1/cclue-bert-base-v2/`、`models/task1/cclue-guwenbert-v2/` |
| 运行结果 | `outputs/task1/<experiment>/` |
| 分实验来源、报告、精简指标与验证 | `task1/reports/<experiment>/` |
| 总报告与冻结清单 | `task1/reports/experiment.md`、`task1/reports/freeze-manifest.json` |

`verify` 生成参考预测，并启动不导入任务一源码的独立进程验证模型导出，同时检查空输入、未知字符及长输入拒绝。`evaluate` 保存每个字符的预测和位置索引；`report` 重算测试 micro-F1、检查原生标签和完整字符覆盖。

两个实验和报告均完成后运行 `uv run --locked python scripts/run.py freeze`，提交文档和清单，再创建新标签 `task1-v2.0`。历史 `task1-v1.0` 不移动、不覆盖，但其本地旧数据和产物已按用户要求删除。需要复现实验时使用新的 `experiment`；数据规则改变另设版本，不覆盖冻结成果。

演示仅接收单窗口，去掉七类标点、中文引号、书名号、空白和控制字符后重新预测，其他字符保留。长文本滑窗与 batch 解码属于任务二。标准 Transformers 导出契约见 [交接说明](reports/handoff.md)，任务二默认使用 v2 主模型，不导入任务一源码。
