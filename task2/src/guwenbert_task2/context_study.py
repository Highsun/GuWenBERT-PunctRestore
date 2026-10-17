from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .common import (ROOT, TASK, assert_protected, digest, read_json, read_rows,
                     source_hashes, write_json, write_rows)
from .data import reconstruct
from .engine import infer
from .metrics import calculate, quality
from .windows import EncodedText, Window


def sentence_statistics(rows, encode_character):
    lengths, character_lengths = [], []
    previous, tokens, chars, bounded = None, 0, 0, False
    runs = discarded_starts = discarded_endings = 0
    for row in rows:
        contiguous = (previous is not None and row["document_index"] == previous["document_index"]
                      and row["record_index"] == previous["record_index"] + 1)
        if not contiguous:
            discarded_endings += int(bounded and chars > 0)
            tokens = chars = 0
            bounded = False
            runs += 1
        if len(row["text"]) != len(row["labels"]):
            raise ValueError("Sentence statistics: character/label mismatch")
        for char, label in zip(row["text"], row["labels"]):
            pieces = encode_character(char)
            if not pieces:
                raise ValueError("Empty character encoding")
            tokens += len(pieces)
            chars += 1
            if label in (2, 3, 4):
                if bounded:
                    lengths.append(tokens)
                    character_lengths.append(chars)
                else:
                    discarded_starts += 1
                tokens = chars = 0
                bounded = True
        previous = row
    discarded_endings += int(bounded and chars > 0)
    if not lengths:
        raise ValueError("No bounded sentence spans")
    ordered = sorted(lengths)
    quantile = lambda q: ordered[math.floor((len(ordered) - 1) * q)]
    stats = {"split": "cleaned training only", "unit": "model content tokens excluding restored punctuation",
             "sentence_end_labels": ["。", "？", "！"], "count": len(lengths),
             "continuous_runs": runs, "discarded_initial_spans": discarded_starts,
             "discarded_incomplete_endings": discarded_endings,
             "mean": statistics.mean(lengths), "median": statistics.median(lengths),
             "p75": quantile(.75), "p90": quantile(.9), "p95": quantile(.95), "p99": quantile(.99),
             "maximum": max(lengths), "all_character_and_token_lengths_equal": lengths == character_lengths,
             "histogram": {str(k): v for k, v in sorted(Counter(lengths).items())},
             "policy": "join consecutive original record indices within same document; exclude each run's first unbounded span and final incomplete span"}
    return stats, lengths


def conditions_from_stats(config, stats):
    origins = {}
    for name in ("median", "mean", "p75", "p90", "p95", "p99"):
        context = int(math.floor(stats[name] + .5))
        origins.setdefault(context, []).append(name)
    for context in config["extra_contexts"]:
        if context < 0:
            raise ValueError("Context lengths must be nonnegative")
        origins.setdefault(context, []).append("control/reference")
    contexts = sorted(origins)
    conditions = []
    for length in config["window_lengths"]:
        if length % 8 or not 2 < length <= 512:
            raise ValueError("Fixed windows must be multiples of eight within 512")
        for context in contexts:
            core = length - 2 - 2 * context
            if core <= 0:
                continue
            conditions.append({"id": f"w{length}-h{context}", "family": f"fixed-window-{length}",
                               "max_length": length, "padded_limit": length, "core_tokens": core,
                               "context_tokens": context, "origins": origins[context]})
    core = config["anchor_core_tokens"]
    for context in contexts:
        logical = core + 2 * context + 2
        padded = (logical + 7) // 8 * 8
        if core <= 0 or padded > 512:
            raise ValueError("Anchored core/context exceeds model limit")
        conditions.append({"id": f"c{core}-h{context}", "family": f"fixed-core-{core}",
                           "max_length": logical, "padded_limit": padded, "core_tokens": core,
                           "context_tokens": context, "origins": origins[context]})
    return conditions


def anchored_windows(encoded, document, max_length, core_tokens):
    remaining = max_length - 2 - core_tokens
    if remaining < 0 or remaining % 2:
        raise ValueError("Anchored windows need symmetric integer context budgets")
    context = remaining // 2
    windows, start = [], 0
    while start < len(encoded.text):
        end = encoded.end_within(start, core_tokens)
        if end <= start:
            raise ValueError("A character exceeds anchored core budget")
        left = bisect_left(encoded.prefix, encoded.prefix[start] - context)
        right = encoded.end_within(end, context)
        window = Window(document, left, right, start, end)
        if window.input_tokens(encoded) > max_length:
            raise AssertionError("Anchored window exceeds input budget")
        windows.append(window)
        start = end
    return windows


def document_counts(gold_rows, predicted_rows):
    counts = []
    for gold, pred in zip(gold_rows, predicted_rows, strict=True):
        if len(gold) != len(pred):
            raise ValueError("Unaligned document predictions")
        counts.append([sum(g == p and g != 0 for g, p in zip(gold, pred)),
                       sum(p != 0 for p in pred), sum(g != 0 for g in gold)])
    return np.asarray(counts, dtype=np.int64)


def bootstrap_scores(counts, draws):
    totals = counts[draws].sum(axis=1)
    denominator = totals[:, 1] + totals[:, 2]
    return np.divide(2.0 * totals[:, 0], denominator,
                     out=np.zeros(len(totals), dtype=float), where=denominator > 0)


def prepare_development(config, receipt, backend):
    freeze = read_json(ROOT / "task1/reports/freeze-manifest.json")
    source = freeze["experiments"][config["model_experiment"]]["dataset"]["splits"]
    training_path = ROOT / source["train"]["path"]
    dev_path = ROOT / source["validation"]["path"]
    for name, path in (("train", training_path), ("validation", dev_path)):
        if digest(path) != source[name]["sha256"]:
            raise ValueError(f"Frozen {name} split differs")
    stats, lengths = sentence_statistics(read_rows(training_path), backend.encode_character)
    stats["source_path"] = str(training_path.relative_to(ROOT))
    stats["source_sha256"] = digest(training_path)
    fragments = read_rows(dev_path)
    documents = reconstruct(fragments)
    for doc in documents:
        doc["document_id"] = doc["document_id"].replace("test/", "validation/", 1)
    directory = ROOT / "data/splits/task2" / config["dataset_version"]
    write_rows(directory / "documents.jsonl", documents, immutable=True)
    manifest = {"dataset_version": config["dataset_version"], "split": "validation",
                "source_path": str(dev_path.relative_to(ROOT)), "source_sha256": digest(dev_path),
                "documents": len(documents), "fragments": len(fragments),
                "characters": sum(len(doc["text"]) for doc in documents),
                "maximum_characters": max(len(doc["text"]) for doc in documents),
                "longer_than_512_characters": sum(len(doc["text"]) > 512 for doc in documents),
                "joins": sum(len(doc["joins"]) for doc in documents),
                "documents_sha256": digest(directory / "documents.jsonl"),
                "construction": "concatenate native validation fragments by anonymous document; retain all supplied characters/labels and join positions"}
    write_json(directory / "manifest.json", manifest)
    target = ROOT / "outputs/task2" / config["experiment"]
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "training-sentence-lengths.npy", np.asarray(lengths, dtype=np.int32), allow_pickle=False)
    write_json(target / "training-sentence-statistics.json", stats)
    return documents, manifest, stats


def probe_masks(documents, backend, core_tokens, margin):
    masks = []
    for document, row in enumerate(documents):
        encoded = EncodedText.build(row["text"], backend.encode_character)
        windows = anchored_windows(encoded, document, core_tokens + 2, core_tokens)
        positions = set()
        for window in windows[:-1]:
            end = window.core_end
            positions.update(range(max(0, end - margin), min(len(row["text"]), end + margin)))
        masks.append(positions)
    return masks


def run_study(config, receipt):
    from .runtime import TorchBackend
    directory = ROOT / "outputs/task2" / config["experiment"]
    if (directory / "results.json").exists():
        raise FileExistsError("Completed context study exists; choose a new experiment ID")
    initial_source = source_hashes()
    backend = TorchBackend(config, receipt)
    documents, dataset, stats = prepare_development(config, receipt, backend)
    conditions = conditions_from_stats(config, stats)
    write_json(directory / "plan.json", {"config": config, "statistics": stats, "conditions": conditions,
                                        "dataset": dataset, "source_files_sha256": initial_source})
    texts = [doc["text"] for doc in documents]
    gold = [doc["labels"] for doc in documents]
    probes = probe_masks(documents, backend, config["anchor_core_tokens"], config["boundary_margin"])
    timings = {c["id"]: [] for c in conditions}
    first_hashes, stable, records, counts_by_condition = {}, {}, {}, {}
    rng = np.random.default_rng(config["seed"])
    orders = []

    def once(condition):
        backend.max_length = condition["padded_limit"]
        builder = anchored_windows if condition["family"].startswith("fixed-core") else None
        return infer(texts, backend, condition["max_length"], condition["core_tokens"],
                     batch_size=config["batch_size"], window_builder=builder)

    def prediction_digest(predictions):
        import hashlib
        h = hashlib.sha256()
        for row in predictions:
            h.update(len(row).to_bytes(8, "little"))
            h.update(bytes(row))
        return h.hexdigest()

    with (directory / "progress.jsonl").open("w", encoding="utf-8", buffering=1) as progress:
        def log(event):
            line = json.dumps(event, ensure_ascii=False)
            progress.write(line + "\n")
            print(line, flush=True)

        log({"event": "plan", "conditions": len(conditions), "documents": len(documents),
             "characters": dataset["characters"], "contexts": sorted({c['context_tokens'] for c in conditions})})
        warmup_order = rng.permutation(len(conditions)).tolist()
        for index in warmup_order:
            condition = conditions[index]
            for _ in range(config["warmup_runs"]):
                once(condition)
            log({"event": "warmup", "condition": condition["id"]})
        for repetition in range(config["repetitions"]):
            order = rng.permutation(len(conditions)).tolist()
            orders.append([conditions[i]["id"] for i in order])
            for index in order:
                condition = conditions[index]
                name = condition["id"]
                result = once(condition)
                fingerprint = prediction_digest(result["labels"])
                first_hashes.setdefault(name, fingerprint)
                stable[name] = stable.get(name, True) and first_hashes[name] == fingerprint
                timings[name].append({"seconds": result["seconds"], **result["profile"]})
                log({"event": "measurement", "round": repetition + 1, "condition": name,
                     "seconds": round(result["seconds"], 5), "prediction_sha256": fingerprint})
                if repetition != config["repetitions"] - 1:
                    continue
                scores = quality(documents, result, config["boundary_margin"])
                shared_scores = calculate(gold, result["labels"], probes)
                if scores["overall"]["characters"] != dataset["characters"]:
                    raise AssertionError("Lost or duplicated characters")
                if condition["family"].startswith("fixed-core"):
                    if scores["window_edges"] != shared_scores:
                        raise AssertionError("Anchored core boundaries changed")
                counts_by_condition[name] = document_counts(gold, result["labels"])
                metrics_file = directory / name / "metrics.json"
                write_rows(directory / name / "predictions.jsonl",
                           [{"document_id": row["document_id"], "gold": row["labels"], "predicted": pred}
                            for row, pred in zip(documents, result["labels"], strict=True)])
                write_rows(directory / name / "windows.jsonl", result["windows"])
                records[name] = {"condition": condition, "quality": scores,
                                 "shared_probe_quality": shared_scores, "prediction_sha256": fingerprint,
                                 "windows": len(result["windows"]), "forward_calls": result["forward_calls"],
                                 "processed_content_tokens": result["processed_content_tokens"],
                                 "content_tokens": result["content_tokens"],
                                 "overlap_work_factor": result["processed_content_tokens"] / result["content_tokens"],
                                 "padding_factor": result["profile"]["padded_tokens"] / result["profile"]["input_tokens_including_special"],
                                 "document_counts": counts_by_condition[name].tolist()}
                write_json(metrics_file, records[name])
    draws = np.random.default_rng(config["seed"] + 1).integers(
        0, len(documents), size=(config["bootstrap_samples"], len(documents)))
    boot = {name: bootstrap_scores(counts, draws) for name, counts in counts_by_condition.items()}
    for condition in conditions:
        name = condition["id"]
        record = records[name]
        samples = timings[name]
        seconds = [item["seconds"] for item in samples]
        record.update({"measurements": samples, "median_seconds": statistics.median(seconds),
                       "stdev_seconds": statistics.stdev(seconds), "min_seconds": min(seconds), "max_seconds": max(seconds),
                       "characters_per_second": dataset["characters"] / statistics.median(seconds),
                       "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in samples),
                       "peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in samples),
                       "repeat_predictions_identical": stable[name],
                       "micro_f1_bootstrap_95ci": np.quantile(boot[name], [.025, .975]).tolist()})
        baseline = next(c["id"] for c in conditions if c["family"] == condition["family"] and c["context_tokens"] == 0)
        baseline_score = records[baseline]["quality"]["overall"]["punctuation_micro"]["f1"]
        record["baseline_condition"] = baseline
        record["micro_f1_delta_vs_no_context_pp"] = 100 * (record["quality"]["overall"]["punctuation_micro"]["f1"] - baseline_score)
        record["paired_delta_bootstrap_95ci_pp"] = (100 * np.quantile(boot[name] - boot[baseline], [.025, .975])).tolist()
    assert_protected(receipt)
    if source_hashes() != initial_source:
        raise RuntimeError("Source changed during context study")
    result = {"schema_version": 1, "date": "2026-10-17", "config": config,
              "training_sentence_statistics": stats, "dataset": dataset, "environment": backend.environment,
              "model_load_seconds": backend.load_seconds, "conditions": [records[c["id"]] for c in conditions],
              "warmup_order": [conditions[i]["id"] for i in warmup_order], "measurement_orders": orders,
              "bootstrap": {"unit": "anonymous development document group", "samples": len(draws),
                            "seed": config["seed"] + 1, "paired": True, "multiple_comparison_adjustment": False},
              "timing": "warmed model and tokenizer; encode/window/sort/pad/transfer/forward/argmax/decode/render; exclude model/data load, provenance validation, scoring and result I/O",
              "source_files_sha256": initial_source, "task1_tag": receipt["task1_tag"],
              "task1_commit": receipt["task1_commit"], "model_files_sha256": receipt["model_files_sha256"],
              "task1_unchanged": True, "test_split_used_for_parameter_selection": False}
    write_json(directory / "results.json", result)
    report_dir = TASK / "reports" / config["experiment"]
    write_json(report_dir / "results.json", result)
    write_json(report_dir / "training-sentence-statistics.json", stats)
    export_table(result, report_dir / "conditions.csv")
    from .context_figures import make_figures, write_report
    make_figures(result, report_dir / "figures")
    write_report(result, report_dir)
    return {"experiment": config["experiment"], "conditions": len(conditions),
            "report": str((report_dir / "experiment.md").relative_to(ROOT))}


def export_table(result, path):
    import csv
    fields = ["family", "context_tokens", "max_length", "padded_limit", "core_tokens", "micro_f1_percent",
              "macro_f1_percent", "micro_delta_pp", "paired_delta_ci_low_pp", "paired_delta_ci_high_pp",
              "shared_probe_micro_f1_percent", "median_seconds", "stdev_seconds", "characters_per_second",
              "windows", "forward_calls", "overlap_work_factor", "peak_allocated_gib"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in result["conditions"]:
            row = {key: record["condition"][key] for key in fields[:5]}
            q = record["quality"]["overall"]
            row.update({"micro_f1_percent": q["punctuation_micro"]["f1"] * 100,
                        "macro_f1_percent": q["punctuation_macro_f1"] * 100,
                        "micro_delta_pp": record["micro_f1_delta_vs_no_context_pp"],
                        "paired_delta_ci_low_pp": record["paired_delta_bootstrap_95ci_pp"][0],
                        "paired_delta_ci_high_pp": record["paired_delta_bootstrap_95ci_pp"][1],
                        "shared_probe_micro_f1_percent": record["shared_probe_quality"]["punctuation_micro"]["f1"] * 100,
                        "median_seconds": record["median_seconds"], "stdev_seconds": record["stdev_seconds"],
                        "characters_per_second": record["characters_per_second"], "windows": record["windows"],
                        "forward_calls": record["forward_calls"], "overlap_work_factor": record["overlap_work_factor"],
                        "peak_allocated_gib": record["peak_allocated_bytes"] / 2**30})
            writer.writerow(row)
