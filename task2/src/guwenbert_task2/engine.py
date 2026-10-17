from __future__ import annotations

from dataclasses import replace
import time

from .windows import EncodedText, adaptive_commit_end, adaptive_window, fixed_windows


def infer(texts: list[str], backend, max_length: int, core_tokens: int,
          strategy: str = "fixed", batch_size: int = 8, sort_windows: bool = True,
          window_builder=None) -> dict:
    if strategy not in ("fixed", "adaptive") or batch_size < 1:
        raise ValueError("Invalid strategy or batch size")
    if strategy == "adaptive" and batch_size != 1:
        raise ValueError("Adaptive sentence-dependent baseline uses batch size one")
    if strategy == "adaptive" and window_builder is not None:
        raise ValueError("Custom windows require fixed strategy")
    backend.begin_run()
    started = time.perf_counter()
    encoded = [EncodedText.build(text, backend.encode_character) for text in texts]
    predictions = [[None] * len(text) for text in texts]
    trace, calls = [], 0

    def commit(window, labels):
        if len(labels) != window.end - window.start:
            raise ValueError("Backend lost character predictions")
        target = predictions[window.document]
        for position in range(window.core_start, window.core_end):
            if target[position] is not None:
                raise AssertionError("Character assigned more than once")
            label = labels[position - window.start]
            if not 0 <= label < 8:
                raise ValueError("Unknown predicted label")
            target[position] = label
        trace.append(window.metadata(encoded[window.document]))

    if strategy == "fixed":
        builder = window_builder or fixed_windows
        windows = [window for i, item in enumerate(encoded)
                   for window in builder(item, i, max_length, core_tokens)]
        if sort_windows:
            windows.sort(key=lambda w: w.input_tokens(encoded[w.document]), reverse=True)
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            values = backend.predict_batch(encoded, batch)
            if len(values) != len(batch):
                raise ValueError("Backend lost windows")
            calls += 1
            for window, labels in zip(batch, values):
                commit(window, labels)
    else:
        for document, item in enumerate(encoded):
            start = 0
            while start < len(item.text):
                window = adaptive_window(item, document, start, max_length)
                labels = backend.predict_batch(encoded, [window])[0]
                calls += 1
                end = adaptive_commit_end(item, window, labels, core_tokens)
                commit(replace(window, core_end=end), labels)
                start = end
    if any(label is None for row in predictions for label in row):
        raise AssertionError("Incomplete character coverage")
    from .common import render
    restored = [render(text, labels) for text, labels in zip(texts, predictions)]
    profile = backend.finish_run()
    seconds = time.perf_counter() - started
    trace.sort(key=lambda w: (w["document"], w["core_start"]))
    return {"labels": predictions, "restored": restored, "windows": trace,
            "seconds": seconds, "forward_calls": calls, "profile": profile,
            "characters": sum(map(len, texts)), "content_tokens": sum(e.prefix[-1] for e in encoded),
            "processed_content_tokens": sum(w["input_tokens"] - 2 for w in trace)}
