from __future__ import annotations

from functools import lru_cache
import json
import platform
import statistics
import time
import unicodedata

from .common import (LABELS, ROOT, TASK, assert_protected, digest, read_json,
                     read_rows, setup_cache, source_hashes, write_json, write_rows)
from .data import load_dataset
from .engine import infer
from .metrics import quality


class TorchBackend:
    def __init__(self, config: dict, receipt: dict):
        setup_cache()
        import torch
        import transformers
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use a Linux terminal with NVIDIA GPU access")
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unavailable")
        torch.set_num_threads(config["num_threads"])
        directory = ROOT / receipt["model_path"]
        contract = read_json(directory / "inference_contract.json")
        if contract["labels"] != LABELS or contract["special_tokens"] != 2:
            raise ValueError("Unsupported task-one inference contract")
        if config["max_length"] > contract["effective_model_token_limit"]:
            raise ValueError("Window exceeds model position limit")
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(directory, local_files_only=True,
                                                                    attn_implementation="sdpa").cuda().eval()
        if self.tokenizer.num_special_tokens_to_add(pair=False) != 2:
            raise ValueError("Expected two special tokens")
        effective = self.model.config.max_position_embeddings
        if self.model.config.model_type == "roberta":
            effective -= self.model.config.pad_token_id + 1
        if effective != contract["effective_model_token_limit"]:
            raise ValueError("Model and contract disagree about position limit")
        if self.tokenizer.pad_token_id != self.model.config.pad_token_id:
            raise ValueError("Padding token mismatch")
        if [self.model.config.id2label[i] for i in range(8)] != LABELS:
            raise ValueError("Model labels differ from contract")
        self.precision = config["precision"]
        self.max_length = config["max_length"]
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.environment = {"python": platform.python_version(), "platform": platform.platform(),
                            "torch": torch.__version__, "transformers": transformers.__version__,
                            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                            "precision": self.precision, "threads": config["num_threads"],
                            "attention": "sdpa", "training_max_length": contract["max_length_including_special_tokens"],
                            "inference_max_length": self.max_length}

    @lru_cache(maxsize=65536)
    def encode_character(self, character: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(character, add_special_tokens=False) or [self.tokenizer.unk_token_id])

    def begin_run(self):
        self.events = []
        self.padded_tokens = 0
        self.input_tokens = 0
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def predict_batch(self, encoded, windows):
        torch = self.torch
        inputs, positions = [], []
        for window in windows:
            item = encoded[window.document]
            origin = item.prefix[window.start]
            ids = [self.tokenizer.cls_token_id, *item.token_ids[origin:item.prefix[window.end]],
                   self.tokenizer.sep_token_id]
            inputs.append(ids)
            positions.append([1 + item.prefix[i] - origin for i in range(window.start, window.end)])
        length = (max(map(len, inputs)) + 7) // 8 * 8
        if length > self.max_length:
            raise ValueError("Padded batch exceeds window limit")
        mask = [[1] * len(ids) + [0] * (length - len(ids)) for ids in inputs]
        self.input_tokens += sum(map(len, inputs))
        self.padded_tokens += len(inputs) * length
        padded = [ids + [self.tokenizer.pad_token_id] * (length - len(ids)) for ids in inputs]
        ids = torch.tensor(padded, device="cuda")
        attention_mask = torch.tensor(mask, device="cuda")
        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before.record()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.precision == "bf16"):
            predicted = self.model(input_ids=ids, attention_mask=attention_mask).logits.argmax(-1)
        after.record()
        self.events.append((before, after))
        # Transfer integer labels once per batch, not logits or individual characters.
        values = predicted.cpu().tolist()
        return [[row[position] for position in offsets] for row, offsets in zip(values, positions)]

    def finish_run(self):
        self.torch.cuda.synchronize()
        return {"gpu_forward_argmax_seconds": sum(a.elapsed_time(b) for a, b in self.events) / 1000,
                "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(),
                "padded_tokens": self.padded_tokens, "input_tokens_including_special": self.input_tokens}


def run(config: dict, backend, rows: list[dict], strategy="fixed", batch_size=None) -> dict:
    return infer([row["text"] for row in rows], backend, config["max_length"], config["core_tokens"],
                 strategy, batch_size or config["batch_size"])


def normalize(text: str) -> str:
    removals = set("，。？！、：；“”‘’《》〈〉")
    return "".join(c for c in text if c not in removals and not c.isspace() and unicodedata.category(c) != "Cc")


def verify(config: dict, receipt: dict) -> dict:
    backend = TorchBackend(config, receipt)
    references = read_json(ROOT / receipt["model_path"] / "reference_predictions.json")
    # Reference cases fit a full window; the task-two code encodes them independently.
    output = run(config, backend, [{"text": item["normalized"]} for item in references], batch_size=1)
    for i, item in enumerate(references):
        if output["labels"][i] != item["labels"] or output["restored"][i] != item["output"]:
            raise AssertionError(f"Export contract mismatch on reference case {i}")
    long_text = "太宗谓侍臣曰为君之道必须先存百姓" * 200
    long_result = run(config, backend, [{"text": long_text}])
    if len(long_result["labels"][0]) != len(long_text) or len(long_result["windows"]) < 2:
        raise AssertionError("Long text not covered")
    import sys
    if any(name.startswith("guwenbert_task1") for name in sys.modules):
        raise AssertionError("Task-one source was imported")
    result = {"passed": True, "reference_cases": len(references), "task1_source_imported": False,
              "long_text_characters": len(long_text), "long_text_windows": len(long_result["windows"]),
              "long_text_characters_exactly_once": len(long_result["labels"][0]),
              "environment": backend.environment, "model_load_seconds": backend.load_seconds}
    assert_protected(receipt)
    write_json(TASK / "reports/verification.json", result)
    write_json(TASK / "reports/task1-inputs.json", receipt)
    return result


def save_predictions(path, rows, result):
    write_rows(path, [{"document_id": row.get("document_id", row.get("article_id")),
                       "gold": row["labels"], "predicted": predictions}
                      for row, predictions in zip(rows, result["labels"])])


def evaluate(config, receipt, kind="documents", limit=None, strategy="fixed", batch_size=None):
    rows, manifest = load_dataset(config, kind, limit)
    backend = TorchBackend(config, receipt)
    result = run(config, backend, rows, strategy, batch_size)
    metrics = quality(rows, result, config["boundary_margin"])
    directory = ROOT / "outputs/task2" / config["experiment"] / f"evaluation-{kind}-{strategy}-b{batch_size or config['batch_size']}"
    if (directory / "metrics.json").exists():
        raise FileExistsError("Evaluation exists; choose a new experiment ID")
    comparison = None
    if kind == "fragments" and config["max_length"] == 256:
        previous = read_rows(ROOT / "outputs/task1" / config["model_experiment"] / "test-predictions.jsonl")
        differences = sum(a != b for row, old in zip(result["labels"], previous[:len(rows)])
                          for a, b in zip(row, old["predicted"]))
        comparison = {"task1_different_characters": differences,
                      "task1_micro_f1": read_json(ROOT / "outputs/task1" / config["model_experiment"] / "test.json")["punctuation_micro"]["f1"]}
    output = {"config": config, "kind": kind, "limit": limit, "records": len(rows),
              "strategy": strategy, "batch_size": batch_size or config["batch_size"],
              "quality": metrics, "task1_short_fragment_comparison": comparison,
              "seconds": result["seconds"], "profile": result["profile"],
              "environment": backend.environment, "dataset": manifest, "source_files_sha256": source_hashes()}
    assert_protected(receipt)
    write_json(directory / "metrics.json", output)
    save_predictions(directory / "predictions.jsonl", rows, result)
    write_rows(directory / "windows.jsonl", result["windows"])
    write_json(TASK / "reports" / config["experiment"] / f"evaluation-{kind}.json", output)
    return output


def benchmark(config, receipt, limit=None):
    rows, dataset = load_dataset(config, "documents", limit)
    target = ROOT / "outputs/task2" / config["experiment"]
    if (target / "benchmark.json").exists():
        raise FileExistsError("Completed benchmark exists; choose a new experiment ID")
    initial_source = source_hashes()
    backend = TorchBackend(config, receipt)
    variants = [("adaptive-b1", "adaptive", 1)] + [(f"fixed-b{b}", "fixed", b) for b in sorted(set([1, *config["batch_sizes"]]))]
    records, fixed_reference = {}, None
    for name, strategy, batch in variants:
        print(json.dumps({"event": "benchmark_start", "variant": name, "records": len(rows)}, ensure_ascii=False), flush=True)
        for _ in range(config["warmup_runs"]):
            run(config, backend, rows, strategy, batch)
        times, gpu_times, allocated, reserved = [], [], [], []
        first_labels, stable = None, True
        for repetition in range(config["repetitions"]):
            result = run(config, backend, rows, strategy, batch)
            if first_labels is None:
                first_labels = result["labels"]
            stable &= first_labels == result["labels"]
            times.append(result["seconds"])
            gpu_times.append(result["profile"]["gpu_forward_argmax_seconds"])
            allocated.append(result["profile"]["peak_allocated_bytes"])
            reserved.append(result["profile"]["peak_reserved_bytes"])
            print(json.dumps({"event": "benchmark_repetition", "variant": name, "repetition": repetition + 1,
                              "seconds": round(times[-1], 4)}, ensure_ascii=False), flush=True)
        if name == "fixed-b1":
            fixed_reference = result
        different = None
        if strategy == "fixed":
            if result["windows"] != fixed_reference["windows"]:
                raise AssertionError("Fixed windows depend on batch size")
            different = sum(a != b for one, other in zip(result["labels"], fixed_reference["labels"])
                            for a, b in zip(one, other))
        median = statistics.median(times)
        entry = {"strategy": strategy, "batch_size": batch, "repetitions": len(times),
                 "seconds": times, "median_seconds": median, "min_seconds": min(times), "max_seconds": max(times),
                 "stdev_seconds": statistics.stdev(times), "gpu_forward_argmax_seconds": gpu_times,
                 "median_gpu_forward_argmax_seconds": statistics.median(gpu_times),
                 "characters_per_second": result["characters"] / median,
                 "peak_allocated_bytes": max(allocated), "peak_reserved_bytes": max(reserved),
                 "characters": result["characters"], "windows": len(result["windows"]),
                 "forward_calls": result["forward_calls"], "content_tokens": result["content_tokens"],
                 "processed_content_tokens": result["processed_content_tokens"],
                 "overlap_work_factor": result["processed_content_tokens"] / result["content_tokens"],
                 "padding_factor": result["profile"]["padded_tokens"] / result["profile"]["input_tokens_including_special"],
                 "different_characters_vs_fixed_b1": different, "repeat_predictions_identical": stable,
                 "quality": quality(rows, result, config["boundary_margin"])}
        records[name] = entry
        save_predictions(target / name / "predictions.jsonl", rows, result)
        write_rows(target / name / "windows.jsonl", result["windows"])
    for entry in records.values():
        entry["speedup_vs_adaptive"] = records["adaptive-b1"]["median_seconds"] / entry["median_seconds"]
        entry["speedup_vs_fixed_b1"] = records["fixed-b1"]["median_seconds"] / entry["median_seconds"]
    assert_protected(receipt)
    if source_hashes() != initial_source:
        raise RuntimeError("Task-two source changed during benchmark")
    output = {"schema_version": 1, "config": config, "limit": limit, "records": len(rows),
              "dataset": dataset, "environment": backend.environment, "model_load_seconds": backend.load_seconds,
              "timing": "warm model and tokenizer cache; encode, construct, sort, pad, transfer, forward, decode and render; exclude model/data loading, file writes and quality scoring",
              "gpu_timing": "CUDA events around forward and argmax only; exclude host transfer and text processing",
              "measurement_order": list(records), "variants": records, "source_files_sha256": initial_source,
              "task1_tag": receipt["task1_tag"], "task1_commit": receipt["task1_commit"],
              "model_files_sha256": receipt["model_files_sha256"], "task1_unchanged": True}
    write_json(target / "benchmark.json", output)
    report_benchmark(output)
    return {"experiment": config["experiment"], "records": len(rows),
            "characters": records["fixed-b1"]["characters"], "report": str((TASK / "reports" / config["experiment"] / "experiment.md").relative_to(ROOT)),
            "variants": {name: {k: r[k] for k in ("median_seconds", "speedup_vs_fixed_b1", "speedup_vs_adaptive", "different_characters_vs_fixed_b1")} for name, r in records.items()}}


def report_benchmark(result: dict):
    cfg = result["config"]
    lines = ["# 任务二初版：长文本滑窗与批量推理实验", "",
             f"实验：`{cfg['experiment']}`。基座为任务一 `{result['task1_tag']}` 的通用中文 BERT，使用 BF16。",
             "", "## 数据与方法", "",
             f"按 CCLUE 匿名 DOCSTART 文档组连接测试片段，评测 {result['records']} 组、{next(iter(result['variants'].values()))['characters']:,} 字符。原始片段边界另行保留。",
             f"总输入上限 {cfg['max_length']} token（含两个特殊 token），固定输出核心预算 {cfg['core_tokens']} token。剩余预算平均分配给左右上下文。短文本能装入一窗时整段输出。",
             "串行方法在输出预算内选最后一个预测句号推进，无句号时按预算推进；固定方法预先确定全部窗口，按核心区间交付字符预测。各方法均保持正文字符完整。",
             "", "## 测试结果", "",
             "| 方法 | 窗口数 | 前向次数 | 中位耗时/s | 字符/s | 相对固定单窗加速 | 相对句号串行加速 | 标点 micro-F1/% | macro-F1/% | 峰值 allocated/GiB | 与固定单窗不同字符 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, entry in result["variants"].items():
        q = entry["quality"]["overall"]
        diff = entry["different_characters_vs_fixed_b1"]
        lines.append(f"| {name} | {entry['windows']} | {entry['forward_calls']} | {entry['median_seconds']:.4f} | {entry['characters_per_second']:.0f} | {entry['speedup_vs_fixed_b1']:.2f}× | {entry['speedup_vs_adaptive']:.2f}× | {q['punctuation_micro']['f1']*100:.2f} | {q['punctuation_macro_f1']*100:.2f} | {entry['peak_allocated_bytes']/2**30:.2f} | {diff if diff is not None else '—'} |")
    lines += ["", "## 边界质量与重复计算", "",
              "| 方法 | 任意标点 F1/% | 句末 F1/% | 窗口交界附近 micro-F1/% | 交界字符数 | 片段连接附近 micro-F1/% | 重复计算系数 | padding 系数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, entry in result["variants"].items():
        q = entry["quality"]
        lines.append(f"| {name} | {q['overall']['segmentation_any_punctuation']['f1']*100:.2f} | {q['overall']['sentence_boundary']['f1']*100:.2f} | {q['window_edges']['punctuation_micro']['f1']*100:.2f} | {q['window_edges']['characters']} | {q['reconstruction_joins']['punctuation_micro']['f1']*100:.2f} | {entry['overlap_work_factor']:.3f} | {entry['padding_factor']:.3f} |")
    lines += ["", "## 测量条件", "",
              f"GPU：{result['environment']['gpu']}；PyTorch：{result['environment']['torch']}；每组预热 {cfg['warmup_runs']} 次，正式测量 {cfg['repetitions']} 次，报告中位数。模型加载耗时 {result['model_load_seconds']:.3f} 秒，单独记录。",
              "端到端计时包含字符编码、构窗、排序、padding、传输、模型前向、解码和文本恢复，排除模型/数据加载、质量计分及文件写出。各组使用预热后的模型与字符编码缓存。CUDA event 另记录前向和 argmax 时间。",
              "固定单窗与固定批量使用相同窗口及排序；标签差异逐字符统计，BF16 和不同 batch 形状可能带来数值差异。reserved 显存受缓存分配器之前运行影响，主要比较 allocated 峰值。",
              f"边界附近定义为输出交界左右各 {cfg['boundary_margin']} 字符的并集；句号推进与固定方案的交界位置不同。重建片段连接附近另用统一位置集合计分。",
              "", "## 实验范围", "",
              "文档由匿名源组中的片段连接重建，连接处新增上下文与原始排版有所区别。结果描述当前模型、数据和硬件的初版表现，最大 batch 与更广泛长文本分布仍需后续实验。400 token 配置改变了任务一 256 token 的上下文条件，单独报告。",
              "任务一文件在运行前后通过校验。本任务独立加载标准模型文件，不导入任务一源码。完整计时重复值、指标、混淆矩阵和校验值见同目录 `results.json`，逐字符预测及窗口位置见 `outputs/task2/<实验>/`。", ""]
    directory = TASK / "reports" / cfg["experiment"]
    write_json(directory / "results.json", result)
    (directory / "experiment.md").write_text("\n".join(lines), encoding="utf-8")
