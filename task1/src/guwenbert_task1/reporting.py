from __future__ import annotations

import json

from .common import LABELS, ROOT, TASK, sha256, write_json


def report(config: dict) -> dict:
    output = ROOT / "outputs/task1" / config["experiment"]
    training = json.loads((output / "training.json").read_text())
    test = json.loads((output / "test.json").read_text())
    model = ROOT / training["model_path"]
    for relative, digest in training["source_files_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Source changed after training: {relative}")
    for name, digest in test["model_files"].items():
        if sha256(model / name) != digest:
            raise ValueError(f"Exported model changed after test: {name}")
    dataset = training["dataset"]
    env = training["environment"]
    results = {"training": training, "test": test}
    write_json(TASK / "reports/results.json", results)
    rows = []
    for label in LABELS[1:]:
        item = test["per_class"][label]
        rows.append(f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']:,} |")
    split_rows = [f"| {name} | {item['articles']:,} | {item['characters']:,} |" for name, item in dataset["splits"].items()]
    epoch_rows = [f"| {item['epoch']} | {item['train_loss']:.5f} | {item['validation']['punctuation_micro']['f1']:.4f} | {item['validation']['punctuation_macro_f1']:.4f} |" for item in training["history"]]
    micro = test["punctuation_micro"]
    boundary = test["sentence_boundary"]
    document = f"""# 任务一实验报告：人民日报 9801 上的 GuwenBERT 微调

本报告来自完整真实运行，日期为 2026-10-15（Asia/Shanghai）。仅研究七类常用标点恢复，未开展任务二的滑窗效率实验。

## 数据与划分

使用 `data/raw/199801.txt`；2—6 月文件未参与本次训练或测试。共有 {dataset['source_articles']:,} 篇源文章，按去标点正文 SHA-256 去除 {dataset['duplicate_articles_removed']} 篇重复文章，保留 {dataset['deduplicated_articles']:,} 篇；重复文章中标签冲突数为 {dataset['duplicate_label_conflicts']}。种子 {config['seed']}，按文章随机分组后以约 80%/10%/10% 划分，随后再切分为互不重叠的 token 窗口。验证集用于选模型，测试集只在训练完成后用于最终评测。

| 集合 | 文章数 | 字符数 |
| --- | ---: | ---: |
{chr(10).join(split_rows)}

先去掉行号、词性标签和组合词包装，连接文章内各行，再构建“字符后标点”标签。标签为 `O` 与 `，。？！；：、`。ASCII 对应标点映射为中文标点；数值内部的中英文小数点、分数斜线和百分号保留。引号、括号、书名号、破折号等不在输出标签范围内，删除并计数；连续目标标点合并为一个标签，句末标点优先，否则取最后一个。源数据检查记录 {dataset['audit'].get('collapsed_target_sequences', 0)} 次合并事件。完整审计与校验值见 `results.json` 及本地数据清单。

原始文件未重写。3 月已知的 U+0016 不在本次 1 月语料中；解析器会记录并删除派生文本中的控制字符，不修复或覆盖原始文件。文章分组和整篇正文去重避免了窗口或完全相同文章跨集合；没有执行近重复检索，不能排除部分重复段落或转载片段。

## 模型与训练

采用 [ethanyt/guwenbert-base](https://huggingface.co/ethanyt/guwenbert-base) 的预训练编码器，新增八类 token 分类头并微调全部参数。其模型卡声明 Apache 2.0 许可，并说明其在古文上继续预训练；本次未使用作者已训练的标点恢复模型。固定上游提交为 `{training['pretrained_source']['resolved_revision']}`。

每个 Unicode 字符独立经 tokenizer 编码，监督其第一个 subtoken，其余 subtoken、特殊 token 与 padding 使用 -100 忽略；无编码字符以 UNK 占位。窗口按实际 token 数装箱，不按字符数直接截断，长度上限 {config['max_length']}，包含 CLS/SEP；所有字符恰好监督一次。

训练 {config['epochs']} 轮，batch={config['batch_size']}，评测 batch={config['eval_batch_size']}，AdamW 学习率 {config['learning_rate']}、weight decay={config['weight_decay']}，warmup 比例 {config['warmup_fraction']}，线性衰减，梯度裁剪 1.0，精度 {config['precision']}。以验证集标点 micro-F1 最高的 epoch 保存模型，不用测试集选 checkpoint。没有重采样或类别权重。

硬件：{env['gpu']}，设备显存 {env['gpu_total_bytes'] / 1024 ** 3:.2f} GiB；Python {env['python']}，PyTorch {env['torch']}，Transformers {env['transformers']}，CUDA runtime {env['cuda_runtime']}。训练窗数 {training['train_windows']:,}，验证窗数 {training['validation_windows']:,}，优化步数 {training['total_steps']:,}；耗时 {training['training_seconds_including_validation_and_export']:.2f} 秒（含逐轮验证及最佳模型导出），PyTorch 峰值已分配显存 {training['peak_allocated_bytes'] / 1024 ** 3:.2f} GiB、峰值保留显存 {training['peak_reserved_bytes'] / 1024 ** 3:.2f} GiB。时间边界已做 GPU 同步，不含模型下载及数据预处理。显存指标不等同于整机或驱动的总占用。

训练时 Git 提交 `{env['git_head_at_run']}`，工作区有未提交修改：`{env['git_dirty_at_run']}`。配置、依赖锁文件和源文件校验值保存在 `results.json`。固定种子有助于复现，未承诺 CUDA 运算逐位一致。

| Epoch | 训练 loss | 验证标点 micro-F1 | 验证标点 macro-F1 |
| --- | ---: | ---: | ---: |
{chr(10).join(epoch_rows)}

## 独立测试结果

加载导出的最佳 checkpoint（epoch {training['best_epoch']}）后，对 {test['test_articles']:,} 篇测试文章、{test['test_windows']:,} 个互不重叠窗口、{test['characters']:,} 个字符计分。测试窗末尾保留原始字符标签，不人为补句号。测试流程耗时 {test['evaluation_seconds_including_prediction_write']:.2f} 秒，包含预测保存，不作为任务二的吞吐量基准。

- 标点 micro precision / recall / F1：{micro['precision']:.4f} / {micro['recall']:.4f} / **{micro['f1']:.4f}**。
- 七类标点 macro-F1：**{test['punctuation_macro_f1']:.4f}**，按固定七类等权平均，零分母计 0。
- 句末边界 precision / recall / F1：{boundary['precision']:.4f} / {boundary['recall']:.4f} / **{boundary['f1']:.4f}**。
- 含 O 的字符准确率：{test['accuracy_including_O']:.4f}；全部预测 O 的基线准确率为 {test['all_O_baseline']['accuracy_including_O']:.4f}，标点 F1 为 0。

| 标点 | Precision | Recall | F1 | 测试支持数 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

标点 micro 汇总七类的 TP、FP、FN，排除 O 作为目标类；错标类别同时产生对应类的 FP 和 FN。句末边界将 `。？！` 合并为一个二元目标，句末标点之间的错分在此指标中视作边界正确，但仍计入标点分类错误。padding、特殊 token 和后续 subtoken 不计分。完整混淆矩阵见 `results.json`；逐窗预测、文章 ID 与字符区间分别保存在本地 `test-predictions.jsonl` 和 `test-window-index.jsonl`。

## 适用范围与局限

该结果是现代新闻语料同月留出测试，不能替代古文标注集上的评测。本次没有可靠的独立古文测试集，因此不报告古文质量指标。单次种子、单次配置实验不能用于说明所有配置或数据上的最优性；稀有标点的支持数较少，须结合各类指标阅读结果。

只恢复七类标点，不能还原原始引号、括号或连续标点的完整形式。数据规范化会删去非目标标点，因此这里衡量的是规范化后的目标任务。字符独立分词保证位置可追溯，但与预训练时的整句分词不同，尤其可能影响拉丁文本和未知字符。

任务一演示仅允许一个 token 窗口；多千字文本的滑窗、批量推理与效率对照留给任务二。标准 `save_pretrained` 模型、tokenizer、`labels.json` 和 `inference_contract.json` 已导出到 `{training['model_path']}`，可通过 Transformers 独立加载；字符对齐和规范化契约需一并遵守。

## 复现与产物

命令见 [任务一 README](../README.md)。训练输出在 `outputs/task1/{config['experiment']}/`；派生数据在 `data/processed/task1/{config['dataset_version']}/`，划分在 `data/splits/task1/{config['dataset_version']}/`。权重、语料和详细输出不纳入 Git；本报告、精简指标、上游模型版本与冻结清单纳入 Git。

模型来自 [GuwenBERT 模型卡](https://huggingface.co/ethanyt/guwenbert-base)，CUDA wheel 安装源依据 [PyTorch 官方版本安装说明](https://pytorch.org/get-started/previous-versions/)，忽略特殊 token 和后续 subtoken 的对齐原则可参考 [Transformers token classification 文档](https://huggingface.co/docs/transformers/tasks/token_classification)。
"""
    (TASK / "reports/experiment.md").write_text(document, encoding="utf-8")
    freeze = {"schema_version": 1, "status": "validated; tag created after report commit", "intended_tag": "task1-v1.0", "experiment": config["experiment"], "training_source_commit": env["git_head_at_run"], "source_files_sha256": training["source_files_sha256"], "model_path": training["model_path"], "model_files_sha256": test["model_files"], "dataset": dataset, "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(output.iterdir()) if path.is_file()}, "reports": {str(path.relative_to(ROOT)): sha256(path) for path in sorted((TASK / "reports").iterdir()) if path.is_file() and path.name != "freeze-manifest.json"}}
    write_json(TASK / "reports/freeze-manifest.json", freeze)
    return {"report": str((TASK / "reports/experiment.md").relative_to(ROOT)), "test_micro_f1": micro["f1"], "freeze_manifest": "task1/reports/freeze-manifest.json"}
