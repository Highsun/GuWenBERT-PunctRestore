import numpy as np
import pytest

from guwenbert_task2.context_study import (anchored_windows, bootstrap_scores,
                                         conditions_from_stats, document_counts, sentence_statistics)
from guwenbert_task2.engine import infer
from guwenbert_task2.windows import EncodedText


def test_sentence_spans_do_not_cross_document_or_removed_record_gaps():
    encode = lambda c: (1, 2) if c == "𠀀" else (1,)
    rows = [{"document_index": 0, "record_index": 0, "text": "甲", "labels": [2]},
            {"document_index": 0, "record_index": 1, "text": "𠀀乙", "labels": [0, 3]},
            {"document_index": 0, "record_index": 2, "text": "丙", "labels": [4]},
            {"document_index": 0, "record_index": 4, "text": "丁", "labels": [2]},
            {"document_index": 1, "record_index": 5, "text": "戊己", "labels": [2, 0]}]
    stats, lengths = sentence_statistics(rows, encode)
    assert lengths == [3, 1]
    assert stats["continuous_runs"] == 3
    assert stats["discarded_initial_spans"] == 3
    assert stats["discarded_incomplete_endings"] == 1
    assert not stats["all_character_and_token_lengths_equal"]


@pytest.mark.parametrize("length", [0, 1, 7, 10, 29])
def test_anchored_core_intervals_are_invariant_to_context_and_whole_window_fit(length):
    item = EncodedText.build("甲" * length, lambda c: (1,))
    intervals = []
    for h in (0, 1, 4, 8):
        windows = anchored_windows(item, 0, 4 + 2 * h + 2, 4)
        intervals.append([(w.core_start, w.core_end) for w in windows])
        assert [i for w in windows for i in range(w.core_start, w.core_end)] == list(range(length))
        assert all(w.input_tokens(item) <= 4 + 2 * h + 2 for w in windows)
    assert all(value == intervals[0] for value in intervals)


def test_conditions_use_statistics_and_exact_context_budgets():
    config = {"extra_contexts": [0, 31], "window_lengths": [256, 400], "anchor_core_tokens": 192}
    stats = {"median": 14, "mean": 18.3, "p75": 23, "p90": 34, "p95": 42, "p99": 66}
    conditions = conditions_from_stats(config, stats)
    assert len(conditions) == 24
    assert len({c["id"] for c in conditions}) == 24
    for c in conditions:
        assert c["max_length"] - 2 - c["core_tokens"] == 2 * c["context_tokens"]
        assert c["padded_limit"] % 8 == 0 and c["padded_limit"] >= c["max_length"]


def test_paired_bootstrap_uses_document_counts_and_zero_delta_for_identical_predictions():
    counts = document_counts([[1, 0, 2], [0, 1]], [[1, 1, 0], [0, 1]])
    assert counts.tolist() == [[1, 2, 2], [1, 1, 1]]
    draws = np.array([[0, 1], [0, 0], [1, 1]])
    values = bootstrap_scores(counts, draws)
    assert values.tolist() == pytest.approx([2 / 3, .5, 1])
    assert np.all(values - bootstrap_scores(counts, draws) == 0)


def test_custom_builder_is_used_without_changing_default_inference():
    class Backend:
        def encode_character(self, c): return (1,)
        def begin_run(self): pass
        def finish_run(self): return {}
        def predict_batch(self, encoded, windows): return [[0] * (w.end - w.start) for w in windows]
    result = infer(["甲" * 7], Backend(), 12, 4, batch_size=2, window_builder=anchored_windows)
    assert [(w['core_start'], w['core_end']) for w in result['windows']] == [(0, 4), (4, 7)]
    assert result['labels'] == [[0] * 7]
