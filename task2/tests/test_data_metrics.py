import pytest

from guwenbert_task2.data import reconstruct
from guwenbert_task2.metrics import calculate, quality


def test_reconstruction_preserves_text_labels_order_and_joins():
    rows = [{"article_id": "b", "record_index": 2, "document_index": 0, "text": "乙丙", "labels": [1, 2]},
            {"article_id": "c", "record_index": 3, "document_index": 1, "text": "丁", "labels": [3]},
            {"article_id": "a", "record_index": 1, "document_index": 0, "text": "甲", "labels": [0]}]
    docs = reconstruct(rows)
    assert docs[0]["text"] == "甲乙丙" and docs[0]["labels"] == [0, 1, 2]
    assert docs[0]["joins"] == [1]
    assert docs[0]["source_spans"][1] == {"article_id": "b", "start": 1, "end": 3}
    assert docs[1]["text"] == "丁"


def test_type_and_boundary_f1_have_distinct_denominators():
    metrics = calculate([[0, 1, 2, 3]], [[1, 1, 3, 0]])
    assert metrics["characters"] == 4
    assert metrics["punctuation_micro"]["f1"] == pytest.approx(1 / 3)
    assert metrics["segmentation_any_punctuation"]["f1"] == pytest.approx(2 / 3)
    assert metrics["sentence_boundary"]["f1"] == pytest.approx(2 / 3)
    assert metrics["punctuation_macro_f1"] == pytest.approx(2 / 3 / 7)


def test_boundary_masks_are_unions_and_partition_characters():
    rows = [{"labels": [1] * 12, "joins": [4, 5]}]
    inference = {"labels": [[1] * 12], "windows": [{"document": 0, "core_end": n} for n in [4, 8, 12]]}
    result = quality(rows, inference, margin=3)
    assert result["window_edges"]["characters"] == 10
    assert result["window_interiors"]["characters"] == 2
    assert result["reconstruction_joins"]["characters"] == 7
    assert result["overall"]["punctuation_micro"]["f1"] == 1


def test_misalignment_and_unknown_labels_fail():
    with pytest.raises(ValueError):
        calculate([[0]], [[]])
    with pytest.raises(ValueError):
        calculate([[8]], [[0]])
