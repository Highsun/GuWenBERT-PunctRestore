import numpy as np

from guwenbert_task1.common import LABELS
from guwenbert_task1.encoding import CharacterEncoder, Collator
from guwenbert_task1.metrics import add_confusion, scores


class Tokenizer:
    cls_token_id = 1
    sep_token_id = 2
    unk_token_id = 3

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, char, add_special_tokens=False):
        return [4, 5] if char == "𠀀" else ([] if char == "\u0301" else [6])


def test_token_budget_multitoken_character_unknown_and_exact_coverage():
    encoder = CharacterEncoder(Tokenizer(), max_length=8)
    text = "甲𠀀乙丙丁戊己\u0301"
    labels = [1] * len(text)
    windows = encoder.windows(text, labels)
    assert len(windows) == 2
    assert [(row["char_start"], row["char_end"]) for row in windows] == [(0, 5), (5, 8)]
    positions = [position for row in windows for position in row["character_positions"] if position >= 0]
    assert positions == list(range(len(text)))
    assert all(len(row["input_ids"]) <= 8 for row in windows)
    assert windows[0]["labels"][3] == -100
    assert sum(label != -100 for row in windows for label in row["labels"]) == len(text)
    assert encoder.encode_character("\u0301") == (3,)
    assert encoder.windows("") == []


def test_padding_is_ignored_and_does_not_create_character_labels():
    encoder = CharacterEncoder(Tokenizer(), 8)
    rows = encoder.windows("甲𠀀", [1, 2]) + encoder.windows("乙", [0])
    batch = Collator(0)(rows)
    assert batch["input_ids"].shape == (2, 8)
    assert (batch["labels"] != -100).sum().item() == 3
    assert batch["attention_mask"][1].sum().item() == 3


def test_punctuation_and_boundary_metrics_have_distinct_meaning():
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    add_confusion(matrix, [0, 1, 2, 3, -100], [1, 1, 3, 0, 7])
    result = scores(matrix)
    assert result["characters"] == 4
    assert result["accuracy_including_O"] == 0.25
    assert np.isclose(result["punctuation_micro"]["f1"], 1 / 3)
    assert np.isclose(result["sentence_boundary"]["f1"], 2 / 3)
    assert np.isclose(result["segmentation_any_punctuation"]["f1"], 2 / 3)
