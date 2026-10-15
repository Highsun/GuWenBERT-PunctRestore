# 任务一：GuwenBERT 标点恢复微调

本任务独立实现数据制备、预训练模型下载、微调、独立测试、短文本演示和实验报告。训练仅使用人民日报 9801（`data/raw/199801.txt`），2—6 月未纳入本次实验；任务二保持空骨架。

已完成三轮真实微调和留出测试：标点 micro-F1 **68.95%**，macro-F1 **48.98%**，句末边界 F1 **73.09%**。完整结果与局限见 [实验报告](reports/experiment.md)，13 项测试及独立导出加载验证记录见 [验证记录](reports/validation.json)。任务一以本地标签 `task1-v1.0` 冻结，后续修改须形成新版本。

## 环境与完整流程

源码位于 `src/guwenbert_task1/`，配置位于 `configs/pd9801.toml`，统一入口为 `scripts/run.py`，测试位于 `tests/`，报告及精简元数据位于 `reports/`。

使用本任务自己的 Python 3.12 `.venv/`、`pyproject.toml` 和 `uv.lock`。uv 缓存设置在 `pyproject.toml` 中，存入本任务 `.cache/uv/`；Hugging Face 缓存由入口设置到本任务 `.cache/huggingface/`。PyTorch 使用官方 CUDA 12.8 wheel 源，GPU 操作须在允许访问 WSL GPU 的终端执行。

```bash
cd /home/highsun/code/GuWenBERT-PunctRestore/code/task1
uv sync --locked
uv run --locked pytest -q
uv run --locked python scripts/run.py prepare
uv run --locked python scripts/run.py download
uv run --locked python scripts/run.py train
uv run --locked python scripts/run.py evaluate
uv run --locked python scripts/run.py predict --text '学而时习之不亦说乎有朋自远方来不亦乐乎'
uv run --locked python scripts/run.py report
```

首次安装及模型下载需要网络；包已缓存后可加 `--offline`。训练和测试仅加载本地模型。默认配置为 `configs/pd9801.toml`，通过 `--config` 指定其他配置。文件路径从脚本位置解析，不依赖启动目录；uv 的环境选择仍需在本任务目录执行。

## 数据、训练与评测规则

`prepare` 对照原始清单检查 SHA-256，以 UTF-8 严格读取并解析行号、词性和组合词包装，连接文章内各行并记录规范化审计。先去除完全相同的去标点文章，再按文章以 seed=42 划分约 80%/10%/10%；切窗在划分之后进行，确保文章及完全相同正文不跨集合。不执行近重复文章或段落去重。

标签为 `O` 与 `，。？！；：、`，每个字符预测其后的一个标点。ASCII 对应标点归一化，数值内部小数点、分数斜线及百分号保留；引号、括号等非目标 Unicode 标点删除并计数。连续目标标点合并为一个标签，句末标点优先，否则保留最后一个。控制字符只在派生文本中删除并审计，原始文件不修改。该任务不会恢复原文全部标点形式。

每个 Unicode 字符独立经 tokenizer 编码，监督第一个 subtoken，其余 subtoken、特殊 token 和 padding 标签为 -100；无编码字符使用 UNK。训练窗口按实际 token 数装箱，最大 256 token，包含两个特殊 token，没有重叠或静默截断。窗口尾字符保留原始标签，不人为补句号。

训练配置为 batch=16、BF16、AdamW、学习率 3e-5、3 epochs。根据每轮验证集标点 micro-F1 选择最佳模型，训练结束后再加载导出模型做独立测试。若 CUDA 不可用则明确报错，不静默切换到 CPU。完整实验耗时与显存以真实运行日志为准。

标点 micro precision/recall/F1 汇总七类标点，macro-F1 等权平均七类；单独报告含 O 准确率及全部预测 O 的基线。句末边界 F1 将 `。？！` 合并为二元目标。特殊 token、padding 和后续 subtoken 不计分。现代新闻留出测试结果不能当作古文评测结果；本次没有独立古文标注测试集。

`predict` 仅用于一个 token 窗口以内的演示；超长输入明确报错。它会去除输入已有标点后重新恢复，不是保留原有排版的接口。长文本滑窗和批量推理属于任务二。

## 输出、复现与交接

以下路径相对于项目根 `code/`：

| 产物 | 路径 |
| --- | --- |
| 派生文章与规范化审计 | `data/processed/task1/pd9801-v1/` |
| 训练/验证/测试 JSONL 与校验清单 | `data/splits/task1/pd9801-v1/` |
| 固定提交的预训练权重 | `models/task1/pretrained/ethanyt--guwenbert-base/<revision>/` |
| 最佳模型、tokenizer、标签及推理契约 | `models/task1/pd9801-guwenbert-v1/` |
| 配置、环境、训练日志、验证历史、测试指标与预测 | `outputs/task1/pd9801-guwenbert-v1/` |
| 报告、精简结果、上游版本及冻结清单 | `task1/reports/` |

`results.json` 和 `experiment.md` 由 `report` 根据真实训练、测试文件生成。`freeze-manifest.json` 记录源文件、数据划分、模型与运行输出的校验值。详细预测及其文章/字符区间索引不上传。

已完成的训练不会被同名实验覆盖。复现时复制 TOML，使用新的 `experiment` 标识；改变数据处理规则或划分时另设 `dataset_version`，保留已冻结版本。原始数据不随 Git 分发，需自行提供并匹配 `data/manifest.json`。

标准 Transformers 导出模型可独立加载，标签保存在模型配置及 `labels.json`；字符对齐、窗口预算与规范化规则见 `inference_contract.json` 和 [交接说明](reports/handoff.md)。任务二不导入任务一源码。

完成真实训练、测试与报告后再按 [冻结流程](../docs/task-boundaries.md)创建标签。任务二不修改任务一代码、配置、依赖、报告和冻结产物；模型和数据不由 Git 标签备份，需另行保存。
