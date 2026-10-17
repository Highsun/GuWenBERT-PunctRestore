from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
import time

import numpy as np

from .common import render
from .fusion import center_weights
from .windows import EncodedText, Window, fixed_windows


@dataclass(frozen=True)
class Candidate:
    document: int
    cut: int
    start: int
    end: int
    band_start: int
    band_end: int
    uncertainty: float
    disagreement: float
    random: float

    def window(self):
        return Window(self.document, self.start, self.end, self.band_start, self.band_end)


def centered_window(encoded, document, cut, max_length, radius):
    budget = max_length - 2
    origin = max(0, min(encoded.prefix[cut] - budget // 2, encoded.prefix[-1] - budget))
    start = bisect_left(encoded.prefix, origin)
    end = encoded.end_within(start, budget)
    left, right = max(start, cut - radius), min(end, cut + radius)
    if not start <= left < right <= end or not start <= cut <= end:
        raise ValueError("Review window cannot cover its boundary")
    return Window(document, start, end, left, right)


def first_posterior(encoded, windows, tape, collect_votes):
    scores = [np.zeros((len(e.text), 8), dtype=np.float64) for e in encoded]
    sums = [np.zeros(len(e.text)) for e in encoded]
    low = [np.full(len(e.text), 8, dtype=np.int8) for e in encoded] if collect_votes else None
    high = [np.full(len(e.text), -1, dtype=np.int8) for e in encoded] if collect_votes else None
    for w, p in zip(windows, tape, strict=True):
        if p.shape != (w.end-w.start, 8) or not np.all(np.isfinite(p)):
            raise ValueError("Invalid first-pass probabilities")
        if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Invalid first-pass distribution")
        weight = center_weights(encoded[w.document], w, 1)
        scores[w.document][w.start:w.end] += p * weight[:, None]
        sums[w.document][w.start:w.end] += weight
        if collect_votes:
            labels = p.argmax(axis=1)
            np.minimum(low[w.document][w.start:w.end], labels, out=low[w.document][w.start:w.end])
            np.maximum(high[w.document][w.start:w.end], labels, out=high[w.document][w.start:w.end])
    if any(np.any(s <= 0) for s in sums):
        raise AssertionError("Missing first-pass coverage")
    posterior = [score / total[:, None] for score, total in zip(scores, sums, strict=True)]
    disagreement = [a != b for a, b in zip(low, high, strict=True)] if collect_votes else None
    return posterior, disagreement


def make_candidates(encoded, windows, posterior, disagreement, config):
    cuts = [[] for _ in encoded]
    for w in windows:
        if w.core_end < len(encoded[w.document].text):
            cuts[w.document].append(w.core_end)
    result = []
    for document, positions in enumerate(cuts):
        positions = sorted(set(positions))
        random_scores = np.random.default_rng(config["seed"] + document).random(len(positions))
        for cut, random_score in zip(positions, random_scores, strict=True):
            w = centered_window(encoded[document], document, cut, config["review_max_length"], config["review_radius"])
            p = posterior[document][w.core_start:w.core_end]
            top = np.partition(p, -2, axis=1)[:, -2:]
            uncertainty = float(np.mean(1 - (top[:, 1] - top[:, 0])))
            dispute = float(np.mean(disagreement[document][w.core_start:w.core_end]))
            result.append(Candidate(document, cut, w.start, w.end, w.core_start, w.core_end,
                                    uncertainty, dispute, float(random_score)))
    return result


def choose_candidates(candidates, selector, fraction):
    if selector not in ("uncertainty", "disagreement", "random", "all") or not 0 <= fraction <= 1:
        raise ValueError("Invalid review policy")
    groups = {}
    for c in candidates:
        groups.setdefault(c.document, []).append(c)
    result = []
    for rows in groups.values():
        count = len(rows) if selector == "all" else math.floor(len(rows) * fraction)
        if selector == "all":
            result.extend(rows)
        else:
            ordered = sorted(rows, key=lambda c: (-getattr(c, selector),
                             -c.uncertainty if selector == "disagreement" else 0, c.cut))
            result.extend(ordered[:count])
    return sorted(result, key=lambda c: (c.document, c.cut))


def apply_reviews(encoded, first_labels, candidates, probabilities):
    if len(candidates) != len(probabilities):
        raise ValueError("Review probability count mismatch")
    labels = [np.asarray(row, dtype=np.int64).copy() for row in first_labels]
    distances = [np.full(len(e.text), np.inf) for e in encoded]
    owners = [np.full(len(e.text), -1, dtype=np.int64) for e in encoded]
    for c, p in zip(candidates, probabilities, strict=True):
        if p.shape != (c.end-c.start, 8) or not np.all(np.isfinite(p)):
            raise ValueError("Invalid review probabilities")
        if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Invalid review distribution")
        d, a, b = c.document, c.band_start, c.band_end
        prefix = encoded[d].prefix
        distance = np.abs(np.asarray(prefix[a:b]) - prefix[c.cut])
        previous = distances[d][a:b]
        owner = owners[d][a:b]
        take = (distance < previous) | ((distance == previous) & (c.cut < owner))
        values = p[a-c.start:b-c.start].argmax(axis=1)
        labels[d][a:b][take] = values[take]
        previous[take] = distance[take]
        owner[take] = c.cut
    return [row.tolist() for row in labels], owners


def infer_review(texts, backend, config, condition):
    backend.begin_run()
    started = time.perf_counter()
    encoded = [EncodedText.build(text, backend.encode_character) for text in texts]
    windows = [w for i, e in enumerate(encoded) for w in fixed_windows(e, i, config["max_length"], config["core_tokens"])]
    windows.sort(key=lambda w: w.input_tokens(encoded[w.document]), reverse=True)
    tape = []
    for start in range(0, len(windows), config["batch_size"]):
        tape.extend(backend.predict_probabilities(encoded, windows[start:start+config["batch_size"]]))
    first_calls = len(backend.events)
    need_scores = condition["mode"] != "baseline"
    posterior, disputes = first_posterior(encoded, windows, tape, need_scores)
    first_labels = [p.argmax(axis=1).tolist() for p in posterior]
    first_seconds = time.perf_counter() - started
    selection_started = time.perf_counter()
    candidates = make_candidates(encoded, windows, posterior, disputes, config) if need_scores else []
    selected = choose_candidates(candidates, condition["selector"], condition["fraction"]) if condition["mode"] == "review" else []
    selected.sort(key=lambda c: c.window().input_tokens(encoded[c.document]), reverse=True)
    selection_seconds = time.perf_counter() - selection_started
    review_started = time.perf_counter()
    review_tape = []
    for start in range(0, len(selected), config["batch_size"]):
        review_tape.extend(backend.predict_probabilities(encoded, [c.window() for c in selected[start:start+config["batch_size"]]]))
    second_seconds = time.perf_counter() - review_started
    apply_started = time.perf_counter()
    labels, owners = apply_reviews(encoded, first_labels, selected, review_tape)
    apply_seconds = time.perf_counter() - apply_started
    restored = [render(text, row) for text, row in zip(texts, labels, strict=True)]
    profile = backend.finish_run()
    seconds = time.perf_counter() - started
    profile.update({"first_pass_seconds": first_seconds, "selection_seconds": selection_seconds,
                    "second_pass_seconds": second_seconds, "review_apply_seconds": apply_seconds,
                    "gpu_first_forward_seconds": sum(a.elapsed_time(b) for a,b in backend.events[:first_calls])/1000,
                    "gpu_review_forward_seconds": sum(a.elapsed_time(b) for a,b in backend.events[first_calls:])/1000})
    trace = sorted([w.metadata(encoded[w.document]) for w in windows], key=lambda w: (w["document"],w["core_start"]))
    total_candidates = sum(w.core_end < len(encoded[w.document].text) for w in windows)
    return {"labels":labels, "first_labels":first_labels, "restored":restored, "seconds":seconds, "profile":profile,
            "windows":trace, "forward_calls":len(backend.events), "first_forward_calls":first_calls,
            "review_forward_calls":len(backend.events)-first_calls, "characters":sum(map(len,texts)),
            "content_tokens":sum(e.prefix[-1] for e in encoded), "processed_content_tokens":sum(w.input_tokens(encoded[w.document])-2 for w in windows)
              +sum(c.window().input_tokens(encoded[c.document])-2 for c in selected),
            "probability_windows":windows, "probabilities":tape, "encoded":encoded,
            "candidates":candidates, "selected":selected, "review_probabilities":review_tape, "owners":owners,
            "candidate_count":total_candidates, "review_windows":len(selected),
            "reviewed_characters":sum(int(np.sum(row>=0)) for row in owners)}
