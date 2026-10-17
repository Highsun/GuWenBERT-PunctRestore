from __future__ import annotations

import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

from .common import (ROOT, TASK, assert_protected, digest, read_json, read_rows,
                     source_hashes, write_json, write_rows)
from .context_study import anchored_windows, bootstrap_scores, document_counts
from .data import load_dataset
from .engine import infer
from .metrics import calculate, quality


def variants(config, selected=None):
    conditions = []
    for window in config["windows"]:
        batches = config["batch_sizes"] if selected is None else sorted({1, selected[window["id"]]["batch_size"]})
        for batch in batches:
            conditions.append({**window, "window_id": window["id"], "id": f"{window['id']}-b{batch}",
                               "strategy": "fixed", "batch_size": batch,
                               "padded_limit": (window["max_length"] + 7) // 8 * 8})
    for length, core in ((256, 192), (400, 300)):
        conditions.append({"id": f"serial{length}-b1", "window_id": f"serial{length}",
                           "label": f"Sentence-driven W{length}/C{core}", "strategy": "adaptive",
                           "max_length": length, "padded_limit": length, "core_tokens": core,
                           "anchored": False, "batch_size": 1})
    return conditions


def select_batches(records, config):
    selected = {}
    for window in config["windows"]:
        candidates = [r for r in records if r["condition"]["window_id"] == window["id"]]
        reference = next(r for r in candidates if r["condition"]["batch_size"] == 16)
        reference_f1 = reference["quality"]["overall"]["punctuation_micro"]["f1"]
        eligible = [r for r in candidates if r["repeat_predictions_identical"] and
                    abs(100 * (r["quality"]["overall"]["punctuation_micro"]["f1"] - reference_f1))
                    <= config["quality_drift_limit_pp"]]
        if not eligible:
            raise ValueError(f"No stable, quality-preserving batch for {window['id']}")
        fastest = min(r["median_seconds"] for r in eligible)
        near = [r for r in eligible if r["median_seconds"] <= fastest * config["near_fastest_ratio"]]
        choice = min(near, key=lambda r: r["condition"]["batch_size"])
        selected[window["id"]] = {"condition_id": choice["condition"]["id"],
                                  "batch_size": choice["condition"]["batch_size"],
                                  "median_seconds": choice["median_seconds"],
                                  "development_micro_f1": choice["quality"]["overall"]["punctuation_micro"]["f1"],
                                  "quality_delta_vs_batch16_pp": 100 * (choice["quality"]["overall"]["punctuation_micro"]["f1"] - reference_f1),
                                  "eligible_batches": [r["condition"]["batch_size"] for r in eligible],
                                  "near_fastest_batches": [r["condition"]["batch_size"] for r in near]}
    return selected


def prediction_digest(labels):
    h = hashlib.sha256()
    for row in labels:
        h.update(len(row).to_bytes(8, "little"))
        h.update(bytes(row))
    return h.hexdigest()


def checked_dataset(config, phase):
    if phase == "test":
        return load_dataset({**config, "dataset_version": config["test_dataset_version"]})
    directory = ROOT / "data/splits/task2" / config["dataset_version"]
    manifest = read_json(directory / "manifest.json")
    if manifest["split"] != "validation" or manifest["dataset_version"] != config["dataset_version"]:
        raise ValueError("Development dataset mismatch")
    if digest(directory / "documents.jsonl") != manifest["documents_sha256"]:
        raise ValueError("Development documents changed")
    if digest(ROOT / manifest["source_path"]) != manifest["source_sha256"]:
        raise ValueError("Frozen development source changed")
    rows = read_rows(directory / "documents.jsonl")
    if len(rows) != manifest["documents"] or sum(len(r["text"]) for r in rows) != manifest["characters"]:
        raise ValueError("Development dataset counts differ")
    return rows, manifest


def check_checkpoint(config):
    path = ROOT / config["accepted_checkpoint"]
    checkpoint = read_json(path)
    if checkpoint["status"] != "accepted_by_user":
        raise ValueError("Previous experiment is not accepted")
    for name, sha in checkpoint["files_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Accepted experiment changed: {name}")
    return digest(path)


def run_phase(config, receipt, phase):
    from .runtime import TorchBackend

    if phase not in ("development", "test"):
        raise ValueError("Unsupported experiment phase")
    target = ROOT / "outputs/task2" / config["experiment"] / phase
    report = TASK / "reports" / config["experiment"]
    if target.exists():
        raise FileExistsError("Experiment phase already exists; use a new experiment ID")
    checkpoint_sha = check_checkpoint(config)
    initial_source = source_hashes()
    selection = None
    if phase == "test":
        selection_path = report / "selection.json"
        selection = read_json(selection_path)
        if selection["source_files_sha256"] != initial_source or selection["config"] != config:
            raise ValueError("Selection source or configuration changed before test")
        if digest(report / "development.json") != selection["development_results_sha256"]:
            raise ValueError("Development results changed after selection")
        selection_sha = digest(selection_path)
    conditions = variants(config, None if selection is None else selection["selected"])
    documents, dataset = checked_dataset(config, phase)
    backend = TorchBackend(config, receipt)
    texts, gold = [d["text"] for d in documents], [d["labels"] for d in documents]
    target.mkdir(parents=True)
    write_json(target / "plan.json", {"phase": phase, "config": config, "conditions": conditions,
                                      "dataset": dataset, "source_files_sha256": initial_source,
                                      "accepted_checkpoint_sha256": checkpoint_sha,
                                      "selection_sha256": None if selection is None else selection_sha})
    timings, fingerprints, saved, failed = {}, {}, {}, {}
    rng = np.random.default_rng(config["seed"] + (phase == "test"))
    warm_order = rng.permutation(len(conditions)).tolist()
    orders = []

    def once(c):
        backend.max_length = c["padded_limit"]
        return infer(texts, backend, c["max_length"], c["core_tokens"], c["strategy"], c["batch_size"],
                     window_builder=anchored_windows if c["anchored"] else None)

    with (target / "progress.jsonl").open("x", encoding="utf-8", buffering=1) as progress:
        def log(event):
            line = json.dumps(event, ensure_ascii=False)
            progress.write(line + "\n")
            print(line, flush=True)

        def attempt(c, event):
            try:
                return once(c)
            except backend.torch.cuda.OutOfMemoryError:
                failed[c["id"]] = {"condition": c, "reason": "CUDA out of memory", "event": event}
                gc.collect()
                backend.torch.cuda.empty_cache()
                log({"event": "out_of_memory", "condition": c["id"], "at": event})
                return None

        log({"event": "plan", "phase": phase, "conditions": len(conditions), "documents": len(documents),
             "characters": sum(map(len, texts))})
        for index in warm_order:
            c = conditions[index]
            for _ in range(config["warmup_runs"]):
                if attempt(c, "warmup") is None:
                    break
            if c["id"] not in failed:
                log({"event": "warmup", "condition": c["id"]})
        for repetition in range(config["repetitions"]):
            order = rng.permutation(len(conditions)).tolist()
            orders.append([conditions[i]["id"] for i in order])
            for index in order:
                c = conditions[index]
                name = c["id"]
                if name in failed:
                    continue
                output = attempt(c, f"round-{repetition + 1}")
                if output is None:
                    continue
                fingerprint = prediction_digest(output["labels"])
                timings.setdefault(name, []).append({"seconds": output["seconds"], **output["profile"],
                                                    "prediction_sha256": fingerprint})
                fingerprints.setdefault(name, []).append(fingerprint)
                log({"event": "measurement", "round": repetition + 1, "condition": name,
                     "seconds": round(output["seconds"], 5), "prediction_sha256": fingerprint})
                if repetition != config["repetitions"] - 1:
                    continue
                directory = target / name
                write_rows(directory / "predictions.jsonl",
                           [{"document_id": doc["document_id"], "gold": doc["labels"], "predicted": pred}
                            for doc, pred in zip(documents, output["labels"], strict=True)])
                write_rows(directory / "windows.jsonl", output["windows"])
                lengths = [("le512", 0, 512), ("513to4096", 513, 4096), ("gt4096", 4097, float("inf"))]
                by_length = {}
                for label, low, high in lengths:
                    indices = [i for i, text in enumerate(texts) if low <= len(text) <= high]
                    by_length[label] = {"documents": len(indices), "quality": calculate(
                        [gold[i] for i in indices], [output["labels"][i] for i in indices])}
                saved[name] = {"condition": c, "quality": quality(documents, output, config["boundary_margin"]),
                               "quality_by_length": by_length,
                               "document_counts": document_counts(gold, output["labels"]).tolist(),
                               "windows": len(output["windows"]), "forward_calls": output["forward_calls"],
                               "content_tokens": output["content_tokens"],
                               "processed_content_tokens": output["processed_content_tokens"],
                               "overlap_work_factor": output["processed_content_tokens"] / output["content_tokens"],
                               "padding_factor": output["profile"]["padded_tokens"] / output["profile"]["input_tokens_including_special"],
                               "prediction_sha256": fingerprint}
    draws = np.random.default_rng(config["seed"] + 2).integers(
        0, len(documents), size=(config["bootstrap_samples"], len(documents)))
    boot = {name: bootstrap_scores(np.asarray(r["document_counts"]), draws) for name, r in saved.items()}
    for name, record in saved.items():
        samples = timings[name]
        seconds = [s["seconds"] for s in samples]
        record.update({"measurements": samples, "median_seconds": statistics.median(seconds),
                       "stdev_seconds": statistics.stdev(seconds), "min_seconds": min(seconds), "max_seconds": max(seconds),
                       "characters_per_second": sum(map(len, texts)) / statistics.median(seconds),
                       "peak_allocated_bytes": max(s["peak_allocated_bytes"] for s in samples),
                       "peak_reserved_bytes": max(s["peak_reserved_bytes"] for s in samples),
                       "repeat_predictions_identical": len(set(fingerprints[name])) == 1,
                       "micro_f1_bootstrap_95ci": np.quantile(boot[name], [.025, .975]).tolist()})
    for name, record in saved.items():
        c = record["condition"]
        if c["strategy"] != "fixed":
            continue
        reference_name = f"{c['window_id']}-b1"
        reference = saved[reference_name]
        predictions = read_rows(target / name / "predictions.jsonl")
        previous = read_rows(target / reference_name / "predictions.jsonl")
        reference_windows = read_rows(target / reference_name / "windows.jsonl")
        if read_rows(target / name / "windows.jsonl") != reference_windows:
            raise AssertionError("Batch size changed fixed window intervals")
        record["different_characters_vs_batch1"] = sum(a != b for row, ref in zip(predictions, previous, strict=True)
                                                      for a, b in zip(row["predicted"], ref["predicted"], strict=True))
        record["speedup_vs_batch1"] = reference["median_seconds"] / record["median_seconds"]
        record["comparisons"] = {}
        for key, ref_name in (("batch1", reference_name), ("adaptive", c["adaptive_reference"])):
            ref = saved[ref_name]
            record["comparisons"][key] = {
                "reference": ref_name, "speedup": ref["median_seconds"] / record["median_seconds"],
                "delta_micro_f1_pp": 100 * (record["quality"]["overall"]["punctuation_micro"]["f1"] - ref["quality"]["overall"]["punctuation_micro"]["f1"]),
                "paired_95ci_pp": (100 * np.quantile(boot[name] - boot[ref_name], [.025, .975])).tolist()}
    assert_protected(receipt)
    if check_checkpoint(config) != checkpoint_sha or source_hashes() != initial_source:
        raise RuntimeError("Protected experiment or source changed during run")
    if selection is not None and digest(report / "selection.json") != selection_sha:
        raise RuntimeError("Locked selection changed during test")
    result = {"schema_version": 1, "date": "2026-10-17", "phase": phase, "config": config,
              "dataset": dataset, "documents": len(documents), "characters": sum(map(len, texts)),
              "environment": backend.environment, "model_load_seconds": backend.load_seconds,
              "conditions": [saved[c["id"]] for c in conditions if c["id"] in saved],
              "failed_conditions": list(failed.values()), "warmup_order": [conditions[i]["id"] for i in warm_order],
              "measurement_orders": orders, "source_files_sha256": initial_source,
              "accepted_checkpoint_sha256": checkpoint_sha, "previous_experiment_unchanged": True,
              "task1_tag": receipt["task1_tag"], "task1_commit": receipt["task1_commit"], "task1_unchanged": True,
              "model_files_sha256": receipt["model_files_sha256"],
              "bootstrap": {"samples": len(draws), "seed": config["seed"] + 2, "paired": True,
                            "unit": "anonymous document group", "multiple_comparison_adjustment": False},
              "timing": "warm model/tokenizer cache; encode/window/sort/pad/transfer/forward/argmax/decode/render; exclude model/data loading, provenance checks, scoring and file writes",
              "selection_sha256": None if selection is None else selection_sha,
              "test_labels_used_for_batch_selection": False}
    write_json(target / "results.json", result)
    write_json(report / f"{phase}.json", result)
    export_table(result, report / f"{phase}.csv")
    snapshot = ROOT / "outputs/task2/source-snapshots" / config["experiment"]
    for relative, sha in initial_source.items():
        dest = snapshot / relative
        if dest.exists():
            if digest(dest) != sha:
                raise ValueError("Run snapshot already contains different source")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, dest)
    if phase == "development":
        write_json(report / "selection.json", {
            "phase": "locked_before_test", "config": config, "selected": select_batches(result["conditions"], config),
            "rule": "within 0.02 percentage points of batch16, stable predictions; choose smallest batch within 3% of fastest median latency per window design",
            "source_files_sha256": initial_source, "development_results_sha256": digest(report / "development.json"),
            "test_labels_used": False})
    return {"experiment": config["experiment"], "phase": phase, "conditions": len(saved),
            "failed": len(failed), "results": str((report / f"{phase}.json").relative_to(ROOT))}


def export_table(result, path: Path):
    fields = ["condition", "strategy", "max_length", "core_tokens", "batch_size", "micro_f1_percent",
              "macro_f1_percent", "segmentation_f1_percent", "sentence_boundary_f1_percent", "median_seconds",
              "stdev_seconds", "characters_per_second", "windows", "forward_calls", "peak_allocated_gib",
              "speedup_vs_batch1", "speedup_vs_adaptive", "different_characters_vs_batch1"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in result["conditions"]:
            c, q = r["condition"], r["quality"]["overall"]
            writer.writerow({"condition": c["id"], **{k: c[k] for k in fields[1:5]},
                             "micro_f1_percent": q["punctuation_micro"]["f1"] * 100,
                             "macro_f1_percent": q["punctuation_macro_f1"] * 100,
                             "segmentation_f1_percent": q["segmentation_any_punctuation"]["f1"] * 100,
                             "sentence_boundary_f1_percent": q["sentence_boundary"]["f1"] * 100,
                             **{k: r[k] for k in fields[9:14]}, "peak_allocated_gib": r["peak_allocated_bytes"] / 2**30,
                             "speedup_vs_batch1": r.get("speedup_vs_batch1"),
                             "speedup_vs_adaptive": r.get("comparisons", {}).get("adaptive", {}).get("speedup"),
                             "different_characters_vs_batch1": r.get("different_characters_vs_batch1")})
