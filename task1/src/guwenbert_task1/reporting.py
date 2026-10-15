from __future__ import annotations

import json
from pathlib import Path

from .common import LABELS, ROOT, TASK, read_jsonl, sha256, write_json

EXPERIMENTS = ("cclue-bert-base-v2", "cclue-guwenbert-v2")


def verify_artifacts(training: dict, test: dict) -> None:
    for relative, digest in training["source_files_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Code/config differs from training snapshot: {relative}")
    model = ROOT / training["model_path"]
    for name, digest in test["model_files"].items():
        if sha256(model / name) != digest:
            raise ValueError(f"Model differs from evaluated checkpoint: {name}")


def audit_predictions(config: dict) -> dict:
    output = ROOT / "outputs/task1" / config["experiment"]
    rows = {row["article_id"]: row for row in read_jsonl(ROOT / "data/splits/task1" / config["dataset_version"] / "test.jsonl")}
    predictions = read_jsonl(output / "test-predictions.jsonl")
    indices = read_jsonl(output / "test-window-index.jsonl")
    if len(predictions) != len(indices):
        raise ValueError("Prediction/index length mismatch")
    last_end = {identity: 0 for identity in rows}
    tp = predicted_positive = gold_positive = count = 0
    for prediction, index in zip(predictions, indices):
        identity = index["article_id"]
        start, end = index["char_start"], index["char_end"]
        if last_end[identity] != start or prediction["gold"] != rows[identity]["labels"][start:end]:
            raise ValueError("Prediction positions or gold labels changed")
        if len(prediction["predicted"]) != end - start:
            raise ValueError("Lost or duplicated prediction characters")
        last_end[identity] = end
        for gold, predicted in zip(prediction["gold"], prediction["predicted"]):
            tp += int(gold != 0 and gold == predicted)
            predicted_positive += int(predicted != 0)
            gold_positive += int(gold != 0)
            count += 1
    if any(last_end[key] != len(row["text"]) for key, row in rows.items()):
        raise ValueError("Incomplete test coverage")
    test = json.loads((output / "test.json").read_text())
    independent_f1 = 2 * tp / (predicted_positive + gold_positive) if predicted_positive + gold_positive else 0.0
    if count != test["characters"] or abs(independent_f1 - test["punctuation_micro"]["f1"]) > 1e-12:
        raise ValueError("Independent test metric recomputation mismatch")
    export = json.loads((output / "export-validation.json").read_text())
    if not export["passed"] or export["task1_source_imported"]:
        raise ValueError("Independent model export verification failed")
    return {"passed": True, "test_records": len(rows), "test_windows": len(indices), "characters_exactly_once": count, "independent_micro_f1": independent_f1, "export_validation": export}


def report(config: dict) -> dict:
    output = ROOT / "outputs/task1" / config["experiment"]
    training = json.loads((output / "training.json").read_text())
    test = json.loads((output / "test.json").read_text())
    verify_artifacts(training, test)
    validation = audit_predictions(config)
    target = TASK / "reports" / config["experiment"]
    write_json(target / "results.json", {"training": training, "test": test})
    write_json(target / "validation.json", validation)
    dataset, env = training["dataset"], training["environment"]
    split_rows = [f"| {name} | {item['records']:,} | {item['documents']} | {item['characters']:,} |" for name, item in dataset["splits"].items()]
    class_rows = [f"| {label} | {test['per_class'][label]['precision']:.4f} | {test['per_class'][label]['recall']:.4f} | {test['per_class'][label]['f1']:.4f} | {test['per_class'][label]['support']:,} |" for label in LABELS[1:]]
    epoch_rows = [f"| {row['epoch']} | {row['train_loss']:.5f} | {row['validation']['punctuation_micro']['f1']:.4f} | {row['validation']['punctuation_macro_f1']:.4f} |" for row in training["history"]]
    doc = f"""# CCLUE 古文标点微调：{config['model_id']}

实验角色：`{config['role']}`。本报告由真实运行文件生成，日期 {config['report_date']}（Asia/Shanghai），实验标识 `{config['experiment']}`。

## 数据核验

使用 [CCLUE 官方仓库](https://github.com/Ethan-yt/CCLUE) `data/punctuation`，固定提交 `{dataset['revision']}`。原始 UTF-8 CoNLL 五列依次为字符、seg_tags、punc_tags、quote_tags、book_tags。直接使用提供的 punc_tags，不从文本重新生成标签，不改写原始 token；B-\\ 对应顿号，标点附着在前一个字符后。quote/book 标签保留在派生记录中，但不训练这两个任务。

实际非空记录数：{dataset['official_nonempty_records']}；README 数量 {dataset['official_readme_records']}。空行与 DOCSTART 仅用于边界，不制造空样本。官方构建脚本按类别中的源文件划分训练/开发/测试；发布文件只保留匿名 DOCSTART，不提供书名，不能仅凭本文件重新验证全部书籍的归属。

保留官方开发和测试记录及其顺序。训练集移除项：{dataset['removed_training']}。移除操作只用正文哈希，不用开发/测试标签选择模型或调整超参数。没有另做随机划分；开发、测试内部重复记录保留并计数。不执行近重复段落去重，因此不能保证不存在同书不同版本、转载片段或预训练重叠。

| 集合 | 记录数 | 匿名文档数 | 字符数 |
| --- | ---: | ---: | ---: |
{chr(10).join(split_rows)}

原始文件、官方加载器、构建 notebook、README、LICENSE 和评测脚本的 SHA-256 保存在数据清单中。仓库声明 Apache 2.0；本项目默认不上传原始数据和权重，来源许可与项目 MIT 授权分开说明。

## 微调设置

从 `{config['model_id']}` 的预训练编码器开始，新增八类 token 分类头，更新全部参数；上游模型提交 `{training['pretrained_source']['resolved_revision']}`。通用中文 BERT 是主实验，古文 GuwenBERT 是对照，二者都在同一 CCLUE 划分上重新微调，没有直接使用现成标点模型。

每个 Unicode 字符独立编码，监督第一个 subtoken；后续 subtoken、特殊 token 和 padding 用 -100 忽略，空编码以 UNK 占位。窗口按实际 token 数装箱，最大 {config['max_length']}（含 CLS/SEP），不重叠、不静默截断、不人为补句末。该规则便于将模型作为独立文件交给任务二，但与预训练整句分词不同，可能影响拉丁字符等输入。

seed={config['seed']}，训练 {config['epochs']} 轮，batch={config['batch_size']}，评测 batch={config['eval_batch_size']}，BF16；AdamW 学习率 {config['learning_rate']}，weight decay={config['weight_decay']}，warmup={config['warmup_fraction']}，线性衰减、梯度裁剪 1.0，无类别加权或重采样。只依据开发集标点 micro-F1 保存最佳 checkpoint（epoch {training['best_epoch']}），训练完成后才运行测试集。未依据测试结果继续调参。

硬件 {env['gpu']}；Python {env['python']}，PyTorch {env['torch']}，Transformers {env['transformers']}，CUDA {env['cuda_runtime']}。训练 {training['train_windows']:,} 窗，开发 {training['validation_windows']:,} 窗，共 {training['total_steps']:,} 优化步。训练耗时 {training['training_seconds_including_validation_and_export']:.2f} 秒（含逐轮开发评测及导出，不含下载和预处理），峰值已分配/保留显存 {training['peak_allocated_bytes']/1024**3:.2f}/{training['peak_reserved_bytes']/1024**3:.2f} GiB。时间边界已同步 GPU；PyTorch 显存指标不代表驱动总占用。

训练源码提交 `{env['git_head_at_run']}`，开始时工作区有修改：`{env['git_dirty_at_run']}`。源代码、配置、锁文件校验值见 `results.json`。固定种子不保证 CUDA 逐位复现。

| Epoch | 训练 loss | 开发标点 micro-F1 | 开发标点 macro-F1 |
| --- | ---: | ---: | ---: |
{chr(10).join(epoch_rows)}

## 独立测试

测试保留官方 {test['test_records']:,} 条非空记录，{test['characters']:,} 字符，{test['test_windows']:,} 窗；重新加载导出的最佳模型计分。

- 七类标点 micro precision/recall/F1：{test['punctuation_micro']['precision']:.4f} / {test['punctuation_micro']['recall']:.4f} / **{test['punctuation_micro']['f1']:.4f}**。
- 七类标点 macro-F1：**{test['punctuation_macro_f1']:.4f}**。
- 任意目标标点的断句位置 F1（CCLUE seg_tags 口径）：**{test['segmentation_any_punctuation']['f1']:.4f}**。
- 仅句末 `。？！` 的边界 F1：**{test['sentence_boundary']['f1']:.4f}**。
- 含 O 准确率：{test['accuracy_including_O']:.4f}；全部预测 O 的基线准确率 {test['all_O_baseline']['accuracy_including_O']:.4f}、标点 F1 为 0。

| 标点 | Precision | Recall | F1 | 支持数 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(class_rows)}

micro 汇总七类 TP/FP/FN，O 不作为正类；错标类型同时产生 FP/FN。macro 固定七类等权，零分母计 0。断句位置将七类都作为边界，句末边界只合并 `。？！`，两者不能混称。指标按字符及原生 punc_tags 计算，没有运行上游完整 quote/book 任务，因此不宣称取得官方综合排行榜成绩。

逐字符测试覆盖、独立 micro-F1 重算和不导入任务一源码的模型加载核对均通过，见 `validation.json`。详细预测及字符区间位于 `outputs/task1/{config['experiment']}/`，模型在 `{training['model_path']}`。演示文本无人工评分，只用于导出一致性核验。

## 局限和交接

这是特定古籍集合上的单种子基线，不代表所有体裁上的质量。标点与引号/书名号为分开的上游任务，本次只训练七类标点，不恢复引号或原始排版。GuwenBERT 预训练使用殆知阁，而 CCLUE 是其衍生数据，存在预训练重叠风险；通用中文模型也不能保证完全没见过这些经典。两个模型的词表及预训练过程不同，差异不能仅归因为一个因素。

标准 Transformers 模型、tokenizer、labels.json、inference_contract.json 和参考预测可由任务二独立加载。任务一只有单窗演示；长文本滑窗、批量解码和性能对照属于任务二，当前未实施。任务二应优先使用主实验的模型；对照模型另存，不覆盖主模型。
"""
    (target / "experiment.md").write_text(doc, encoding="utf-8")
    available = all((TASK / "reports" / name / "results.json").exists() for name in EXPERIMENTS)
    if available:
        comparison()
    return {"report": str((target / "experiment.md").relative_to(ROOT)), "micro_f1": test["punctuation_micro"]["f1"], "comparison_ready": available}


def comparison() -> dict:
    experiments = {name: json.loads((TASK / "reports" / name / "results.json").read_text()) for name in EXPERIMENTS}
    first, second = experiments.values()
    for key in ("dataset_version", "dataset_revision", "seed", "max_length", "batch_size", "eval_batch_size", "epochs", "learning_rate", "weight_decay", "warmup_fraction", "precision"):
        if first["training"]["config"][key] != second["training"]["config"][key]:
            raise ValueError(f"Comparison settings differ: {key}")
    if first["test"]["test_data_sha256"] != second["test"]["test_data_sha256"]:
        raise ValueError("Comparison test inputs differ")
    rows = []
    for name, result in experiments.items():
        test = result["test"]
        role = "通用 BERT 主实验" if name == EXPERIMENTS[0] else "GuwenBERT 古文基座对照"
        rows.append(f"| {role} | {test['punctuation_micro']['f1']:.4f} | {test['punctuation_macro_f1']:.4f} | {test['segmentation_any_punctuation']['f1']:.4f} | {test['sentence_boundary']['f1']:.4f} |")
    difference = second["test"]["punctuation_micro"]["f1"] - first["test"]["punctuation_micro"]["f1"]
    doc = f"""# 任务一 v2：通用中文 BERT 在 CCLUE 古文上微调

主实验为 `google-bert/bert-base-chinese`，对照为 `ethanyt/guwenbert-base`。两者从预训练编码器重新微调，使用相同的 CCLUE 数据、标签、超参数及种子，分别以开发集选最佳模型，再运行相同的官方测试记录。任务二尚未实施。

| 实验 | 标点 micro-F1 | 标点 macro-F1 | 任意标点断句 F1 | 句末边界 F1 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

GuwenBERT 相对通用 BERT 的标点 micro-F1 差值为 {difference*100:+.2f} 个百分点。该差值来自单种子、固定配置实验，不证明统计显著性；词表、预训练数据和模型实现也不同，不能把差异全部归因为古文预训练。

## 数据核验与实验解释

原始数据固定为 CCLUE 提交 `{first['training']['dataset']['revision']}`。训练数据直接读取原生 punc_tags，保留官方开发与测试集，移除训练正文中与留出集完全重复的记录以及训练内部重复记录。当前实际用于训练/开发/测试的记录数分别为 {first['training']['dataset']['splits']['train']['records']:,}/{first['training']['dataset']['splits']['validation']['records']:,}/{first['training']['dataset']['splits']['test']['records']:,}。

每个字符只计一次，padding、特殊 token 与后续 subtoken 不计分。标点分类和断句位置是两个指标，句末边界是另一种较窄的定义。数据来自既有古籍，存在同书不同版本及基座预训练重叠风险。没有新增专家标注，不宣称这是完全无污染的外部评测，也不与不同数据集上的旧分数直接比较。

## 报告与复现

- [通用 BERT 主实验报告](cclue-bert-base-v2/experiment.md)
- [GuwenBERT 对照报告](cclue-guwenbert-v2/experiment.md)
- [数据核验与来源](../../docs/corpus-inventory.md)
- [任务一命令与依赖](../README.md)
- [任务二文件交接](handoff.md)

主模型位于 `models/task1/cclue-bert-base-v2/`，对照位于 `models/task1/cclue-guwenbert-v2/`。原始语料、派生数据、权重和详细运行结果不提交 Git；精简报告、来源版本、验证记录和冻结清单纳入 Git。当前版本使用新的 `task1-v2.0` 标签冻结，保留历史标签，不覆盖旧版本。
"""
    (TASK / "reports/experiment.md").write_text(doc, encoding="utf-8")
    summary = {name: {"model_id": result["training"]["config"]["model_id"], "role": result["training"]["config"]["role"], "test": result["test"]} for name, result in experiments.items()}
    write_json(TASK / "reports/comparison.json", summary)
    return summary


def freeze(config: dict) -> dict:
    comparison()
    experiments = {}
    for name in EXPERIMENTS:
        result = json.loads((TASK / "reports" / name / "results.json").read_text())
        training, test = result["training"], result["test"]
        verify_artifacts(training, test)
        audit_predictions(training["config"])
        for item in training["dataset"]["sources"]:
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("CCLUE raw source changed")
        for item in training["dataset"]["splits"].values():
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Processed split changed")
        output = ROOT / "outputs/task1" / name
        experiments[name] = {"training_source_commit": training["environment"]["git_head_at_run"], "source_files_sha256": training["source_files_sha256"], "model_path": training["model_path"], "model_files_sha256": test["model_files"], "dataset": training["dataset"], "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(output.iterdir()) if path.is_file()}}
    manifest = {"schema_version": 2, "intended_tag": config["freeze_tag"], "primary_experiment": EXPERIMENTS[0], "experiments": experiments, "reports": {str(path.relative_to(ROOT)): sha256(path) for path in sorted((TASK / "reports").rglob("*")) if path.is_file() and path.name != "freeze-manifest.json"}}
    write_json(TASK / "reports/freeze-manifest.json", manifest)
    return {"manifest": "task1/reports/freeze-manifest.json", "primary_model": experiments[EXPERIMENTS[0]]["model_path"], "tag": config["freeze_tag"]}
