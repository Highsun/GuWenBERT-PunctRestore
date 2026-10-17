from __future__ import annotations

import time

import numpy as np

from .common import render
from .context_study import anchored_windows
from .runtime import TorchBackend
from .windows import EncodedText, fixed_windows


def center_weights(encoded, window, power):
    """Distance uses whole-character token boundaries, excluding special tokens."""
    if power not in (0, 1, 2):
        raise ValueError("Unsupported center weighting power")
    prefix = np.asarray(encoded.prefix[window.start:window.end + 1], dtype=np.int64)
    first = prefix[:-1] - prefix[0]
    after = prefix[1:] - prefix[0]
    tokens = prefix[-1] - prefix[0]
    distance = np.minimum(first, tokens - after) + 1
    return (distance / ((tokens + 1) // 2)) ** power


def combine_probabilities(encoded, windows, probabilities, mode, power=1):
    """Reference implementation used in tests and independent saved-cache audits."""
    if mode not in ("prob-core", "uniform", "weighted"):
        raise ValueError("Unknown probability output mode")
    if len(windows) != len(probabilities):
        raise ValueError("Probability window count mismatch")
    scores = [np.zeros((len(e.text), 8), dtype=np.float64) for e in encoded]
    sums = [np.zeros(len(e.text), dtype=np.float64) for e in encoded]
    coverage = [np.zeros(len(e.text), dtype=np.int32) for e in encoded]
    for w, probs in zip(windows, probabilities, strict=True):
        probs = np.asarray(probs)
        if probs.shape != (w.end - w.start, 8) or not np.all(np.isfinite(probs)):
            raise ValueError("Invalid character probability matrix")
        if np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Invalid probability distribution")
        if mode == "prob-core":
            start, end = w.core_start, w.core_end
            values = probs[start - w.start:end - w.start]
            weights = np.ones(end - start)
        else:
            start, end, values = w.start, w.end, probs
            weights = center_weights(encoded[w.document], w, power) if mode == "weighted" else np.ones(end - start)
        scores[w.document][start:end] += values * weights[:, None]
        sums[w.document][start:end] += weights
        coverage[w.document][start:end] += 1
    if any(np.any(s <= 0) for s in sums):
        raise AssertionError("Uncovered character in probability aggregation")
    if mode == "prob-core" and any(np.any(c != 1) for c in coverage):
        raise AssertionError("Core predictions must be assigned exactly once")
    normalized = [score / total[:, None] for score, total in zip(scores, sums)]
    labels = [score.argmax(axis=1).tolist() for score in normalized]
    return labels, sums, coverage


class ProbabilityBackend(TorchBackend):
    def begin_run(self):
        super().begin_run()
        self.probability_events = []
        self.transfer_seconds = 0.0
        self.transfer_bytes = 0
        self.probability_mode = False

    def predict_probabilities(self, encoded, windows):
        torch = self.torch
        self.probability_mode = True
        inputs, batch_indices, positions, counts = [], [], [], []
        for batch_index, w in enumerate(windows):
            item = encoded[w.document]
            origin = item.prefix[w.start]
            inputs.append([self.tokenizer.cls_token_id, *item.token_ids[origin:item.prefix[w.end]],
                           self.tokenizer.sep_token_id])
            counts.append(w.end - w.start)
            batch_indices.extend([batch_index] * counts[-1])
            positions.extend(1 + item.prefix[i] - origin for i in range(w.start, w.end))
        length = (max(map(len, inputs)) + 7) // 8 * 8
        if length > self.max_length:
            raise ValueError("Probability batch exceeds padded input budget")
        mask = [[1] * len(ids) + [0] * (length - len(ids)) for ids in inputs]
        padded = [ids + [self.tokenizer.pad_token_id] * (length - len(ids)) for ids in inputs]
        self.input_tokens += sum(map(len, inputs))
        self.padded_tokens += len(inputs) * length
        ids = torch.tensor(padded, device="cuda")
        attention_mask = torch.tensor(mask, device="cuda")
        batch_indices = torch.tensor(batch_indices, device="cuda")
        positions = torch.tensor(positions, device="cuda")
        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before.record()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.precision == "bf16"):
            logits = self.model(input_ids=ids, attention_mask=attention_mask).logits
        after.record()
        self.events.append((before, after))
        p_before, p_after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        p_before.record()
        with torch.inference_mode():
            compact = logits[batch_indices, positions].float().softmax(dim=-1)
        p_after.record()
        self.probability_events.append((p_before, p_after))
        started = time.perf_counter()
        values = compact.cpu().numpy()
        self.transfer_seconds += time.perf_counter() - started
        self.transfer_bytes += values.nbytes
        ends = np.cumsum([0, *counts])
        return [values[a:b] for a, b in zip(ends[:-1], ends[1:])]

    def finish_run(self):
        profile = super().finish_run()
        if self.probability_mode:
            profile["gpu_forward_seconds"] = profile.pop("gpu_forward_argmax_seconds")
            profile["gpu_probability_seconds"] = sum(a.elapsed_time(b) for a, b in self.probability_events) / 1000
            profile["d2h_probability_bytes"] = self.transfer_bytes
            profile["d2h_wait_and_copy_seconds"] = self.transfer_seconds
        else:
            profile["d2h_integer_bytes"] = profile["padded_tokens"] * 8
        return profile


def infer_probabilities(texts, backend, window_config, mode, power=1):
    if mode not in ("prob-core", "uniform", "weighted"):
        raise ValueError("Invalid fusion mode")
    backend.begin_run()
    started = time.perf_counter()
    encoded = [EncodedText.build(text, backend.encode_character) for text in texts]
    builder = anchored_windows if window_config["anchored"] else fixed_windows
    windows = [w for i, e in enumerate(encoded) for w in builder(
        e, i, window_config["max_length"], window_config["core_tokens"])]
    windows.sort(key=lambda w: w.input_tokens(encoded[w.document]), reverse=True)
    # Keep the same probability tape in all probability modes for a controlled comparison.
    tape, calls = [], 0
    for start in range(0, len(windows), window_config["batch_size"]):
        batch = windows[start:start + window_config["batch_size"]]
        tape.extend(backend.predict_probabilities(encoded, batch))
        calls += 1
    aggregation_started = time.perf_counter()
    labels, sums, coverage = combine_probabilities(encoded, windows, tape, mode, power)
    aggregation_seconds = time.perf_counter() - aggregation_started
    restored = [render(text, row) for text, row in zip(texts, labels, strict=True)]
    profile = backend.finish_run()
    seconds = time.perf_counter() - started
    profile["aggregation_seconds"] = aggregation_seconds
    # Serialization and hashing occur after the timed inference returns.
    trace = sorted([w.metadata(encoded[w.document]) for w in windows], key=lambda w: (w["document"], w["core_start"]))
    return {"labels": labels, "restored": restored, "windows": trace, "seconds": seconds,
            "forward_calls": calls, "profile": profile, "characters": sum(map(len, texts)),
            "content_tokens": sum(e.prefix[-1] for e in encoded),
            "processed_content_tokens": sum(w.input_tokens(encoded[w.document]) - 2 for w in windows),
            "probabilities": tape, "probability_windows": windows, "encoded": encoded,
            "weight_sums": sums, "coverage_counts": coverage}
