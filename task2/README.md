# 任务二：长文本滑窗与批量推理

任务二使用任务一 `task1-v2.0` 的通用中文 BERT 主模型，独立实现长文本断句标点推理。源码、uv 环境、配置、测试和运行结果均与任务一分开，不重新训练模型。

滑窗初版、句长指导的上下文预算和 batch 扩展三阶段结果见[阶段总结](reports/stage-summary.md)。跨窗口概率融合已完成实现、开发集参数选择及测试，详见下方章节和[实验报告](reports/cclue-window-fusion-v0.4-optimized/experiment.md)。

初版已完成 19 项工程测试、独立模型验证、短片段质量回归及两种窗长的全量性能对照，见 [初版实验结果汇总](reports/initial-summary.md)。256-token、batch=16 相对相同窗口的 batch=1 加速 8.92 倍，相对句号推进串行方法加速 9.57 倍；长文标点 micro-F1 为 68.48%，比句号串行方案低约 0.16 个百分点。

## 推理方案

| 方案 | 窗口与输出规则 | 对照目的 |
| --- | --- | --- |
| `adaptive-b1` | 每窗预测后，在输出预算内最后一个预测句号处结束，无句号时按预算推进 | 题目中的串行方法 |
| `fixed-b1` | 预先确定固定核心及左右上下文，每个字符由其负责窗口输出 | 分离窗口策略变化的影响 |
| `fixed-bN` | 与 fixed-b1 相同的窗口，按长度排序组 batch，按原位置拼接 | 测量 batch 推理收益 |

默认输入上限 256 token（含两个特殊 token）、核心预算 192、左右上下文各最多 31 token。`window400.toml` 使用总预算 400、核心预算 300、左右上下文各最多 49 token。全文字符编码一次，各窗口复用编码结果；预算依据实际 token 数计算，保持多 subtoken 字符完整，空编码使用 UNK，只输出首个 subtoken 标签。短文本能够装入一窗时整段处理。

固定方案的核心输出区间互不重叠，重叠部分只提供上下文，每个字符计分一次。窗口位置与金标无关；串行方案只依据预测句号推进。batch 内动态 padding 到 8 的倍数，推理使用 BF16 和 inference mode，整数预测每 batch 传回一次。

## 环境与命令

Linux、NVIDIA GPU/CUDA、uv 和 Python 3.12。任务二有独立的 `pyproject.toml`、`uv.lock`、`.venv/` 和 `.cache/`，不导入任务一源码。

```bash
cd /home/highsun/code/GuWenBERT-PunctRestore/code/task2
uv sync --locked
uv run --locked --offline pytest -q
uv run --locked --offline python scripts/run.py --help
```

以下命令均在上述 `task2/` 目录执行。依赖、数据和模型已在本地时可使用 `--offline`。

## 数据与独立加载验证

```bash
uv run --locked --offline python scripts/run.py prepare
uv run --locked --offline python scripts/run.py verify
```

按官方测试数据的匿名 DOCSTART 组连接原始片段，保留全部字符、标签、源区间及连接位置，不插入分隔符。得到 180 个重建文档组、417,786 字符，最长 71,551 字符，99 组超过 512 字符；源片段共 3,992 条。这些文档组是工程评测单位，匿名信息无法完整恢复书籍和原始排版。

`verify` 独立加载标准 Transformers 模型，核对四个导出参考案例、3,200 字长输入完整覆盖，以及任务一源码未被导入。每次命令执行前后均核对任务一冻结文件。

## 文本预测

```bash
uv run --locked --offline python scripts/run.py predict --text '太宗谓侍臣曰为君之道必须先存百姓'
uv run --locked --offline python scripts/run.py predict --input /path/to/long-text.txt --output outputs/task2/demo/prediction.json
uv run --locked --offline python scripts/run.py predict --input /path/to/long-text.txt --strategy adaptive --batch-size 1
```

`--input` 是 UTF-8 文件。`--output` 相对于 `code/`，限于 `outputs/task2/`，已有文件拒绝覆盖；未指定时打印 JSON。演示输入去掉七类中文标点、中文引号、书名号、空白和 Cc 控制字符，其他字符保留。正式评价直接使用原生字符和标签。

## 质量回归与性能对照

```bash
uv run --locked --offline python scripts/run.py evaluate --dataset fragments
uv run --locked --offline python scripts/run.py evaluate --dataset documents
uv run --locked --offline python scripts/run.py benchmark
uv run --locked --offline python scripts/run.py benchmark --config configs/window400.toml
```

默认比较串行方法及固定窗口 batch=1/4/8/16。每组预热一次，完整测量三次，报告中位数、范围及标准差。端到端耗时包含编码、构窗、排序、padding、传输、前向、解码和文本恢复，排除数据/模型加载、质量计分及文件写出。模型加载及 GPU 前向与 argmax 时间另计；所有组使用预热后的模型和字符编码缓存。

评价记录标点 micro/macro-F1、任意标点断句 F1 和句末边界 F1，另统计核心交界左右各 8 字符及重建片段连接附近的质量。计数包括窗口数、前向次数、原文字符吞吐、重复计算系数、padding 系数和 allocated/reserved 显存峰值。固定单窗与批量使用相同窗口及排序；BF16 在不同 batch 形状下可能出现少量标签差异，逐字符记录。

已完成的基准拒绝覆盖，重复实验需复制配置并修改 `experiment`。`--limit N` 可检查前 N 个文档组，使用独立实验 ID，其结果与完整集合分开报告。400 token 改变了任务一 256 token 的训练上下文条件，质量单独评价。

初版使用固定顺序测量，后续可扩展更大 batch、随机化测量顺序和更多独立长文本。

## 句长指导的上下文预算实验

在训练集统计句长，在开发集重建长文上比较左右各 H=0/8/14/18/23/31/34/42/49/66/96 token。包含固定总窗长 256、400 和固定核心区间 192 三组实验；固定核心组保持交界位置一致，观察添加上下文的影响。每个条件预热一次，按随机顺序测量五次，并计算文档级配对 bootstrap 差值区间。

```bash
cd /home/highsun/code/GuWenBERT-PunctRestore/code/task2
uv sync --locked
uv run --locked --offline python scripts/context_study.py run
```

完整实验已存在时拒绝覆盖。结果及报告用图片位于 `task2/reports/cclue-context-budget-v0.2/`，包括训练句长分布、质量变化、质量与吞吐、交界附近效果和资源成本；每张图提供 PNG、SVG 和 PDF。完整预测、窗口索引及五次原始计时保存在 `outputs/task2/cclue-context-budget-v0.2/`。

实验设计见 [上下文预算对照设计](reports/cclue-context-budget-v0.2/design.md)。初版 v0.1 结果保留，其源码和依赖快照位于 `outputs/task2/source-snapshots/v0.1/`，校验清单见 `reports/v0.1-source-snapshot.json`。

本轮 33 个条件已完成全部 165 次正式测量，28 项工程测试通过。固定核心组中，H=14 相对零上下文使总体 micro-F1 提高 1.552 个百分点，交界区域提高 20.018 个百分点；相对原预算 H=31 的总体增益仅 0.093 个百分点，配对区间包含零。H=8～66 的总体差异较小，H=96 出现下降。详见[完整实验报告](reports/cclue-context-budget-v0.2/experiment.md)、[结果分析](reports/cclue-context-budget-v0.2/conclusions.md)及 [CSV 汇总](reports/cclue-context-budget-v0.2/conditions.csv)。本轮分数来自开发集，初版默认配置保持不变。

`scripts/context_study.py plot` 从保存结果重新生成五张图及自动生成的 `experiment.md`，会覆盖这些生成文件；独立撰写的 `conclusions.md` 保留。需要修改报告文字时，优先编辑后者。

## Batch 扩展与测试集验证

上下文预算实验已获用户验收，文件校验值保存于 [验收记录](reports/checkpoints/context-v0.2-accepted.json)。后续实验使用新的编号 `cclue-batch-scaling-v0.3`，比较原配置 256/192/31、句长中位数配置 222/192/14、质量配置 256/156/49 和题目示例配置 400/300/49，分别测量 batch=1/4/8/16/32/64，并保留两种句号推进串行参考。

开发集选定 batch 后锁定参数，再运行测试集评价。每种方案同时保留相同窗口的 batch=1，分别报告 batch 加速、相对串行加速和质量差异。详细规则见 [实验设计](reports/cclue-batch-scaling-v0.3/design.md)，阶段状态见 [实验记录](reports/experiment-register.md)。

```bash
cd /home/highsun/code/GuWenBERT-PunctRestore/code/task2
uv run --locked --offline python scripts/batch_study.py development
uv run --locked --offline python scripts/batch_study.py test
uv run --locked --offline python analysis/batch_report.py
uv run --locked --offline python analysis/audit_batch_study.py
```

已存在的实验阶段、图表报告和校验记录拒绝覆盖。重新测量需复制配置并修改 `experiment`，两个阶段均通过 `--config` 指定新配置；生成报告时用 `--experiment` 指定对应编号。图表与校验脚本放在 `analysis/`，仅处理已保存结果，不运行模型。

本轮已完成 26 个开发集条件、10 个测试集条件，共 180 次正式测量。开发集选定 batch 为 original256=16、median222=32、quality256=16、teacher400=16；batch=64 虽均可运行，但未取得最佳整体耗时。median222-b32 的测试 micro-F1 为 68.509%，中位耗时 2.142 秒，相对对应串行参考加速 10.88 倍。H=49 的开发集优势未在测试集得到明确确认。详见[完整实验报告](reports/cclue-batch-scaling-v0.3/experiment.md)、[结果分析](reports/cclue-batch-scaling-v0.3/conclusions.md)、[测试集 CSV](reports/cclue-batch-scaling-v0.3/test.csv)和[独立核验](reports/cclue-batch-scaling-v0.3/validation.json)。34 项工程测试通过，任务一与已验收的预算实验保持一致。

## 跨窗口概率融合

实验设计见 [v0.4 方案](reports/cclue-window-fusion-v0.4-optimized/design.md)。对 median222-b32 与 original256-b16 比较整数核心输出、保留概率的核心控制、等权概率融合和中心加权融合。开发集比较线性、平方中心权重，选定后再评价测试集。各方案保持窗口位置、输入和 batch 相同，融合仅改变概率聚合。

初始实现的 11 条件开发集结果保留于 `cclue-window-fusion-v0.4`，其[优化记录](reports/cclue-window-fusion-v0.4/pilot-note.md)说明每窗处理全文索引的额外成本。最终实现只处理窗口内索引，使用独立编号 `cclue-window-fusion-v0.4-optimized`；两轮开发集所有预测和概率指纹一致。

```bash
cd /home/highsun/code/GuWenBERT-PunctRestore/code/task2
uv run --locked --offline python scripts/fusion_study.py development
uv run --locked --offline python scripts/fusion_study.py test
uv run --locked --offline python analysis/fusion_report.py
uv run --locked --offline python analysis/audit_fusion_study.py
```

默认配置为 `configs/fusion-study-optimized.toml`。已完成阶段拒绝覆盖；新实验复制配置并修改 `experiment`，测量和核验用 `--config`，绘图用 `--experiment` 指定新编号。独立核验从保存的概率缓存复算融合标签、覆盖次数、权重和及质量指标，不需要 GPU。原概率、详细预测和聚合缓存位于 `outputs/task2/<实验编号>/`，不纳入 Git。

最终实验完成 11 个开发集条件、9 个测试集条件，共 100 次正式测量，另保留初始实现的 55 次开发集测量。原 256/192/31、batch=16 的线性中心加权使测试 micro-F1 从 68.476% 提高到 68.652%，增益区间为 [0.087, 0.243] 个百分点；耗时增加 5.13%，相对串行加速 9.10 倍，达到预设工程条件。等权融合在两种窗口下均降低总体质量。完整结果见[实验报告](reports/cclue-window-fusion-v0.4-optimized/experiment.md)、[结果分析](reports/cclue-window-fusion-v0.4-optimized/conclusions.md)和[独立核验](reports/cclue-window-fusion-v0.4-optimized/validation.json)。46 项工程测试通过，六张图均提供 PNG、SVG、PDF。

## 选择性局部复核

在上一阶段 256/192/31、batch=16 的线性融合后，以交界不确定性或跨窗预测分歧筛选候选，增加一轮居中窗口批量推理，只替换交界左右各 8 字符。开发集比较名义 10%、25%、50% 预算，并设置逐文档同预算随机复核、全量复核、仅评分和串行对照。算法不读取金标准，完整设计见 [v0.5 方案](reports/cclue-selective-review-v0.5/design.md)。

```bash
cd /home/highsun/code/GuWenBERT-PunctRestore/code/task2
uv run --locked --offline python scripts/review_study.py development
uv run --locked --offline python scripts/review_study.py test
uv run --locked --offline python analysis/audit_review_study.py
uv run --locked --offline python analysis/review_report.py
```

默认配置为 `configs/review-study.toml`。先锁定开发集选择，再测试、独立核验及绘图。已完成阶段与报告拒绝覆盖；新实验复制配置并修改 `experiment`，运行和核验用 `--config`，绘图用 `--experiment` 指定新编号。独立核验重放候选评分、逐文档预算、复核归属及最终预测，原概率与详细预测保存在 `outputs/task2/cclue-selective-review-v0.5/`。

本轮完成开发集 13 个条件、测试集 6 个条件，共 95 次正式测量。开发集锁定 disagreement-10，测试集实际复核 162/2,077 个交界，micro-F1 为 68.650%，相对无复核 68.652% 的差值区间为 [−0.0186, +0.0157] 个百分点，耗时增加 10.16%。相对同预算随机复核也未确认总体 F1 优势，因此保留为实验方法，不改动既有默认配置。分歧评分选中了基线错误较集中的区域，但第二遍修正错误的同时新增错误，抵消了收益。

完整结果见[实验报告](reports/cclue-selective-review-v0.5/experiment.md)、[结果分析](reports/cclue-selective-review-v0.5/conclusions.md)、[测试 CSV](reports/cclue-selective-review-v0.5/test.csv)和[独立核验](reports/cclue-selective-review-v0.5/validation.json)。56 项工程测试通过，六幅图均提供 PNG、SVG、PDF。任务一和前期实验保护文件保持一致。

方法演进与设计依据见[研究路线](docs/research-evolution.md)，本轮扩充工作见[v0.4～v0.5 阶段总结](reports/stage-summary-v0.5.md)。各阶段原设计、结果分析与源码快照分别保留。

## 文件与结果

| 内容 | 相对 code/ 的位置 |
| --- | --- |
| 源码与入口 | `task2/src/guwenbert_task2/`、`task2/scripts/run.py` |
| 窗口配置 | `task2/configs/initial.toml`、`task2/configs/window400.toml` |
| 上下文预算实验入口与配置 | `task2/scripts/context_study.py`、`task2/configs/context-study.toml` |
| Batch 扩展实验入口与配置 | `task2/scripts/batch_study.py`、`task2/configs/batch-study.toml` |
| Batch 实验图表与核验脚本 | `task2/analysis/batch_report.py`、`task2/analysis/audit_batch_study.py` |
| 概率融合实现与实验入口 | `task2/src/guwenbert_task2/fusion.py`、`task2/scripts/fusion_study.py` |
| 概率融合图表与核验 | `task2/analysis/fusion_report.py`、`task2/analysis/audit_fusion_study.py` |
| 局部复核实现、入口与配置 | `task2/src/guwenbert_task2/review.py`、`task2/scripts/review_study.py`、`task2/configs/review-study.toml` |
| 局部复核图表与核验 | `task2/analysis/review_report.py`、`task2/analysis/audit_review_study.py` |
| 重建长文与片段回归集 | `data/splits/task2/cclue-long-v0.1/` |
| 上下文预算开发集 | `data/splits/task2/cclue-dev-long-v0.2/` |
| 数据制备摘要 | `data/processed/task2/cclue-long-v0.1/summary.json` |
| 模型来源与验证 | `task2/reports/task1-inputs.json`、`task2/reports/verification.json` |
| 质量及性能报告 | `task2/reports/<实验 ID>/` |
| 详细预测、窗口和计时 | `outputs/task2/<实验 ID>/` |
| 测试记录 | `outputs/task2/validation-v0.1/`～`validation-v0.5/`、`outputs/task2/validation-v0.4-optimized/` |

派生长文、详细预测、缓存和环境不纳入 Git。模型只读复用任务一权重，实际性能结论见生成的分实验报告。
