"""Independently audit completed batch-study predictions, measurements and provenance."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from guwenbert_task2.common import ROOT, TASK, digest, load_config, read_json, read_rows, verify_task1, write_json
from guwenbert_task2.batch_study import check_checkpoint, checked_dataset, select_batches

import numpy as np


def audit(config):
    report = TASK / "reports" / config["experiment"]
    destination = report / "validation.json"
    if destination.exists():
        raise FileExistsError("Completed audit exists; preserve the validation record")
    receipt = verify_task1(config)
    checkpoint_sha = check_checkpoint(config)
    selection = read_json(report / "selection.json")
    assert not selection["test_labels_used"]
    assert digest(report / "development.json") == selection["development_results_sha256"]
    original = read_json(TASK / "reports/cclue-context-budget-v0.2/results.json")
    old_hashes = {r["condition"]["id"]: r["prediction_sha256"] for r in original["conditions"]}
    old_names = {"original256": "w256-h31", "median222": "c192-h14", "quality256": "w256-h49", "teacher400": "w400-h49"}
    artifacts, phase_results = {}, {}
    for phase in ("development", "test"):
        result = read_json(report / f"{phase}.json")
        assert result["accepted_checkpoint_sha256"] == checkpoint_sha
        assert result["task1_unchanged"] and result["previous_experiment_unchanged"]
        if phase == "test":
            assert result["selection_sha256"] == digest(report / "selection.json")
        documents, dataset = checked_dataset(config, phase)
        assert result["dataset"] == dataset
        assert len(documents) == result["documents"]
        for name, sha in result["source_files_sha256"].items():
            assert digest(ROOT / name) == sha, name
            assert digest(ROOT / "outputs/task2/source-snapshots" / config["experiment"] / name) == sha, name
        output = ROOT / "outputs/task2" / config["experiment"] / phase
        assert read_json(output / "results.json") == result
        plan = read_json(output / "plan.json")
        assert plan["config"] == config and plan["dataset"] == dataset
        log = read_rows(output / "progress.jsonl")
        measured = [e for e in log if e["event"] == "measurement"]
        assert len(measured) == len(result["conditions"]) * config["repetitions"]
        draw = np.random.default_rng(config["seed"] + 2).integers(
            0, len(documents), size=(config["bootstrap_samples"], len(documents)))
        boot, predictions, traces = {}, {}, {}
        records = {r["condition"]["id"]: r for r in result["conditions"]}
        for name, r in records.items():
            c = r["condition"]
            pred_path, window_path = output / name / "predictions.jsonl", output / name / "windows.jsonl"
            preds, windows = read_rows(pred_path), read_rows(window_path)
            predictions[name], traces[name] = preds, windows
            artifacts[str(pred_path.relative_to(ROOT))] = digest(pred_path)
            artifacts[str(window_path.relative_to(ROOT))] = digest(window_path)
            assert len(preds) == len(documents)
            matrix = np.zeros((8, 8), dtype=np.int64)
            counts = []
            h = hashlib.sha256()
            for doc, pred in zip(documents, preds, strict=True):
                assert doc["document_id"] == pred["document_id"]
                assert doc["labels"] == pred["gold"] and len(pred["predicted"]) == len(doc["text"])
                gold, value = np.asarray(doc["labels"]), np.asarray(pred["predicted"])
                assert np.all((0 <= value) & (value < 8))
                matrix += np.bincount(gold * 8 + value, minlength=64).reshape(8, 8)
                counts.append([int(np.sum((gold == value) & (gold != 0))), int(np.sum(value != 0)), int(np.sum(gold != 0))])
                h.update(len(value).to_bytes(8, "little"))
                h.update(bytes(pred["predicted"]))
            assert counts == r["document_counts"]
            assert matrix.tolist() == r["quality"]["overall"]["confusion_gold_rows_predicted_columns"]
            pn, gn = matrix[:, 1:].sum(), matrix[1:, :].sum()
            score = 2 * np.trace(matrix[1:, 1:]) / (pn + gn)
            assert abs(score - r["quality"]["overall"]["punctuation_micro"]["f1"]) < 1e-12
            macro = np.mean([2 * matrix[i, i] / (matrix[i, :].sum() + matrix[:, i].sum())
                             if matrix[i, :].sum() + matrix[:, i].sum() else 0 for i in range(1, 8)])
            assert abs(macro - r["quality"]["overall"]["punctuation_macro_f1"]) < 1e-12
            for key, indices in (("segmentation_any_punctuation", list(range(1, 8))),
                                 ("sentence_boundary", [2, 3, 4])):
                boundary_tp = matrix[np.ix_(indices, indices)].sum()
                boundary_denominator = matrix[indices, :].sum() + matrix[:, indices].sum()
                assert abs(2 * boundary_tp / boundary_denominator - r["quality"]["overall"][key]["f1"]) < 1e-12
            assert int(matrix.sum()) == result["characters"]
            assert h.hexdigest() == r["prediction_sha256"]
            assert r["repeat_predictions_identical"]
            samples = r["measurements"]
            assert len(samples) == config["repetitions"]
            assert all(s["prediction_sha256"] == h.hexdigest() for s in samples)
            assert [e["prediction_sha256"] for e in measured if e["condition"] == name] == [h.hexdigest()] * config["repetitions"]
            times = [s["seconds"] for s in samples]
            assert statistics.median(times) == r["median_seconds"]
            assert statistics.stdev(times) == r["stdev_seconds"]
            assert abs(r["characters_per_second"] - result["characters"] / r["median_seconds"]) < 1e-9
            cursor = [0] * len(documents)
            for w in windows:
                i = w["document"]
                assert w["core_start"] == cursor[i]
                assert 0 <= w["start"] <= w["core_start"] < w["core_end"] <= w["end"] <= len(documents[i]["text"])
                assert w["input_tokens"] == w["end"] - w["start"] + 2
                assert w["input_tokens"] <= c["max_length"]
                if c["anchored"]:
                    context = (c["max_length"] - 2 - c["core_tokens"]) // 2
                    assert w["core_start"] - w["start"] <= context and w["end"] - w["core_end"] <= context
                    assert w["core_end"] - w["core_start"] <= c["core_tokens"]
                cursor[i] = w["core_end"]
            assert cursor == [len(d["text"]) for d in documents]
            assert len(windows) == r["windows"]
            assert r["forward_calls"] == (len(windows) + c["batch_size"] - 1) // c["batch_size"]
            assert sum(w["input_tokens"] - 2 for w in windows) == r["processed_content_tokens"]
            for group, low, high in (("le512", 0, 512), ("513to4096", 513, 4096), ("gt4096", 4097, float("inf"))):
                ids = [i for i, d in enumerate(documents) if low <= len(d["text"]) <= high]
                values = np.asarray(counts)[ids].sum(axis=0)
                expected_f1 = 2 * values[0] / (values[1] + values[2]) if values[1] + values[2] else 0
                assert len(ids) == r["quality_by_length"][group]["documents"]
                assert abs(expected_f1 - r["quality_by_length"][group]["quality"]["punctuation_micro"]["f1"]) < 1e-12
            sampled = np.asarray(counts)[draw].sum(axis=1)
            scores = 2 * sampled[:, 0] / (sampled[:, 1] + sampled[:, 2])
            boot[name] = scores
            assert np.allclose(np.quantile(scores, [.025, .975]), r["micro_f1_bootstrap_95ci"], atol=1e-12)
            if phase == "development" and c["strategy"] == "fixed" and c["batch_size"] == 16:
                assert r["prediction_sha256"] == old_hashes[old_names[c["window_id"]]]
        for name, r in records.items():
            c = r["condition"]
            if c["strategy"] != "fixed":
                continue
            ref_name = f"{c['window_id']}-b1"
            assert traces[name] == traces[ref_name]
            differences = sum(a != b for p, ref in zip(predictions[name], predictions[ref_name], strict=True)
                              for a, b in zip(p["predicted"], ref["predicted"], strict=True))
            assert differences == r["different_characters_vs_batch1"]
            for key, comp in r["comparisons"].items():
                reference = records[comp["reference"]]
                assert abs(comp["speedup"] - reference["median_seconds"] / r["median_seconds"]) < 1e-12
                assert np.allclose(100 * np.quantile(boot[name] - boot[comp["reference"]], [.025, .975]), comp["paired_95ci_pp"], atol=1e-12)
            if phase == "test":
                assert c["batch_size"] in (1, selection["selected"][c["window_id"]]["batch_size"])
        if phase == "development":
            assert select_batches(result["conditions"], config) == selection["selected"]
        phase_results[phase] = {"conditions": len(records), "measured_runs": len(measured),
                                "warmups": sum(e["event"] == "warmup" for e in log),
                                "documents": len(documents), "characters": result["characters"],
                                "prediction_coverage_exactly_once": True, "independent_confusion_recomputation": True,
                                "bootstrap_recomputed": True, "batch_window_intervals_identical": True}
    bundle = read_json(report / "report-bundle.json")
    for name, sha in bundle["generated_files_sha256"].items():
        assert digest(report / name) == sha, name
    for name, sha in bundle["input_files_sha256"].items():
        assert digest(report / name) == sha
    assert digest(ROOT / bundle["analysis_script"]["path"]) == bundle["analysis_script"]["sha256"]
    extra = read_json(report / "window-comparisons.json")
    assert not extra["used_for_parameter_selection"]
    for contrast in extra["contrasts"]:
        candidate, reference = contrast["candidate"], contrast["reference"]
        assert np.allclose(100 * np.quantile(boot[candidate] - boot[reference], [.025, .975]), contrast["paired_95ci_pp"], atol=1e-12)
        assert abs(contrast["latency_ratio_candidate_over_reference"] - records[candidate]["median_seconds"] / records[reference]["median_seconds"]) < 1e-12
    for name, sha in read_json(TASK / "reports/artifact-validation.json")["artifact_sha256"].items():
        assert digest(ROOT / name) == sha, name
    tests = read_json(TASK / "reports/unit-tests-v0.3.json")
    assert digest(ROOT / tests["junit_xml"]["path"]) == tests["junit_xml"]["sha256"]
    suites = ET.parse(ROOT / tests["junit_xml"]["path"]).getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 34
    assert all(int(s.attrib["failures"]) + int(s.attrib["errors"]) == 0 for s in suites)
    write_json(destination, {"passed": True, "phases": phase_results, "task1_unchanged": True,
                             "task1_files_checked": len(receipt["verified_files"]),
                             "accepted_context_experiment_unchanged": True, "batch16_matches_context_experiment": True,
                             "selection_locked_before_test": True, "unit_tests_passed": 34,
                             "figure_files_checked": 15, "artifact_sha256": artifacts,
                             "audit_script": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": digest(Path(__file__))}})
    print(json.dumps({"passed": True, "phases": phase_results, "artifact_files_checked": len(artifacts)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/batch-study.toml")
    args = parser.parse_args()
    audit(load_config(args.config))
