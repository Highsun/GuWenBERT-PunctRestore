# GuWenBERT-PunctRestore

任务一：用通用中文预训练 BERT 在 CCLUE 古文数据上微调断句标点模型；用古文预训练 GuwenBERT 做同数据对照。任务二：使用冻结主模型，研究长文本串行滑窗与批量滑窗的推理质量和效率。

当前任务一 v2 已完成两个模型的三轮微调、独立测试与导出验证，并保持冻结。详细设置见 [任务一实验总报告](task1/reports/experiment.md)。任务二初版已实现独立的串行句号推进、固定重叠窗口与批量推理，运行方式和实测结果见 [任务二说明](task2/README.md)。旧版代码和报告保留在历史 Git 标签中；旧语料、派生数据、微调权重和运行输出已按用户要求移除。

任务二后续已完成句长指导的上下文预算实验和 batch 扩展、测试集验证，见 [实验记录](task2/reports/experiment-register.md)。当前句长中位数方案在重建测试长文上的标点 micro-F1 为 68.509%，相对对应串行参考加速 10.88 倍；质量差值为 −0.124 个百分点，配对区间包含零。任务二长文推理结果与下表任务一原始测试片段结果分别评价。

本轮扩充工作归纳于[任务二阶段总结](task2/reports/stage-summary.md)，下一阶段[跨窗口概率融合计划](task2/docs/next-stage-plan.md)待实施。

同一 CCLUE 测试集上的结果：

| 实验 | 标点 micro-F1 | 标点 macro-F1 | 任意标点断句 F1 | 句末边界 F1 |
| --- | ---: | ---: | ---: | ---: |
| 通用中文 BERT（主实验） | 70.76% | 55.32% | 85.46% | 72.06% |
| GuwenBERT（古文基座对照） | 79.86% | 66.11% | 92.65% | 81.84% |

14 项单元测试通过，两个模型分别通过独立导出加载核对。对照的 micro-F1 高 9.09 个百分点，但只有单种子、单配置，且存在词表差异和潜在预训练重叠，不能单独归因为领域预训练。测试文本中的 UNK 字符比例分别约 0.85%/0.14%，详见 [分词审计](task1/reports/tokenization-audit.json)。

## 任务定位

- 主实验：`google-bert/bert-base-chinese` → CCLUE 原生七类标点微调 → 官方测试记录上的独立评测。
- 对照实验：`ethanyt/guwenbert-base` → 同一数据和设置微调 → 同一测试集；不直接使用现成标点模型。
- 任务二：滑窗解决 token 上限，batch 提高 GPU 利用率，重叠区合并保证正文完整。核心是推理加速；重叠本身会增加计算量，不保证更快。

## 目录与 Git

工作目录为 `/home/highsun/code/GuWenBERT-PunctRestore`，项目和唯一仓库根为其下的 `code/`。所有代码、缓存、模型和输出均保存在项目内。远程为 [Highsun/GuWenBERT-PunctRestore](https://github.com/Highsun/GuWenBERT-PunctRestore)，分支 `main`；线上仓库根对应本地 `code/`。

```text
code/
├── README.md、LICENSE、AGENTS.md
├── pyproject.toml、uv.lock      # 根目录初始化环境
├── data/
│   ├── README.md、manifest.json
│   ├── raw/cclue/<commit>/     # 原始 CoNLL、来源与许可，不提交
│   ├── processed/task1/cclue-punc-v2/
│   └── splits/task1/cclue-punc-v2/
├── docs/
│   ├── corpus-inventory.md     # 实际数据核验与标签定义
│   └── task-boundaries.md      # 两项任务边界与冻结约定
├── task1/
│   ├── pyproject.toml、uv.lock、.python-version
│   ├── src/guwenbert_task1/    # 独立数据、训练、评测与报告实现
│   ├── configs/               # 通用 BERT 主实验与 GuwenBERT 对照
│   ├── scripts/、tests/
│   └── reports/               # 分实验报告、比较、来源与冻结清单
├── task2/                     # 独立长文本推理、测试、配置与评测报告
├── models/task1/              # 两个微调模型和预训练权重，不提交
├── outputs/task1/             # 分实验运行记录与详细预测，不提交
└── .cache/                    # 根目录初始化缓存，不提交
```

`task1/` 与 `task2/` 各有自己的依赖声明、锁文件、虚拟环境、配置、脚本、测试和报告，不构成共享锁文件的 uv workspace，也不跨任务导入源码。任务二只读冻结模型、标签、tokenizer 和数据版本，运行结果另存 `outputs/task2/`。

## 数据与质量边界

数据采用 [CCLUE 官方仓库](https://github.com/Ethan-yt/CCLUE) `data/punctuation` 的固定提交，UTF-8 严格读取。直接使用逐字符 punc_tags，保持官方开发与测试记录。训练集移除与留出集完全重复的正文，以及训练内部完全重复正文；不进行另一次随机划分。

实际非空训练/开发/测试记录数为 26,932/4,074/3,992；训练清理后为 26,890 条。开发、测试内部各有 2 条重复正文，保留以维持官方测试记录。官方 README 的训练/开发数略有不同，详见 [数据核验](docs/corpus-inventory.md)。

本次只做七类标点任务，标签为 `O` 与 `，。？！、：；`；quote_tags 和 book_tags 保留供审计，不训练引号及书名号恢复。任意标点断句 F1 与仅 `。？！` 的句末边界 F1 分别报告。古籍版本和基座预训练语料可能重叠，不宣称完全无污染，也不将本次分数直接与其他数据集比较。

## 环境和运行

WSL、Python 3.12、uv/venv；任务一锁定 CUDA 12.8 的 PyTorch 2.9.1、Transformers 等依赖。设备为 RTX 5070 12GB，GPU 操作须在允许 WSL 设备访问的终端执行。缓存均限定在项目内，不使用系统 Python 安装依赖。

```bash
cd /home/highsun/code/GuWenBERT-PunctRestore/code/task1
uv sync --locked
uv run --locked python scripts/run.py fetch-data
uv run --locked python scripts/run.py prepare
uv run --locked python scripts/run.py download
uv run --locked python scripts/run.py train
uv run --locked python scripts/run.py verify
uv run --locked python scripts/run.py evaluate
uv run --locked python scripts/run.py report
```

对照实验在 `download`、`train`、`verify`、`evaluate`、`report` 命令后添加 `--config configs/cclue_guwenbert.toml`；两个实验都完成后执行 `freeze`，再提交报告并创建新的 `task1-v2.0` 标签；本地已按此流程完成冻结。完整说明见 [任务一 README](task1/README.md)。

数据下载与模型下载需要网络；依赖和权重齐全后的训练/测试可离线运行。训练以开发集选择最佳 epoch，独立测试不参与调参。已完成的实验不能被同名训练覆盖，重新实验须更换 `experiment`，数据规则改变须更换 `dataset_version`。

## 冻结与许可证

任务一 v2 使用新标签 `task1-v2.0`，不覆盖历史 `task1-v1.0`。冻结清单记录代码、配置、锁文件、数据、模型和结果校验值；Git 标签不自动备份权重及语料。任务二后续以 v2 通用 BERT 主模型为默认输入。

项目新增代码与文档使用 [MIT License](LICENSE)。CCLUE 仓库及两个基座标示 Apache 2.0，原始数据与权重不随项目 MIT 重新授权，也不默认上传 GitHub。
