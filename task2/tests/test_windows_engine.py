import pytest

from guwenbert_task2.engine import infer
from guwenbert_task2.windows import EncodedText, adaptive_commit_end, adaptive_window, fixed_windows


class Backend:
    def encode_character(self, character):
        return (9, 10) if character == "𠀀" else (ord(character),)

    def begin_run(self):
        pass

    def finish_run(self):
        return {}

    def predict_batch(self, encoded, windows):
        return [[position % 8 for position in range(w.start, w.end)] for w in windows]


@pytest.mark.parametrize("text", ["", "甲", "甲" * 6, "甲" * 7, "甲" * 37, "甲𠀀乙𠀀丙" * 9])
def test_actual_token_budget_and_exact_core_coverage(text):
    encoded = EncodedText.build(text, Backend().encode_character)
    windows = fixed_windows(encoded, 0, 8, 4)
    positions = [i for w in windows for i in range(w.core_start, w.core_end)]
    assert positions == list(range(len(text)))
    assert all(w.input_tokens(encoded) <= 8 for w in windows)
    assert all(w.start <= w.core_start < w.core_end <= w.end for w in windows)


def test_multitoken_character_too_large_is_rejected():
    encoded = EncodedText.build("𠀀" * 8, Backend().encode_character)
    with pytest.raises(ValueError, match="core"):
        fixed_windows(encoded, 0, 8, 1)


def test_batch_sorting_preserves_original_documents_and_empty_inputs():
    texts = ["甲" * 30, "", "甲𠀀乙" * 7, "甲乙"]
    one = infer(texts, Backend(), 8, 4, batch_size=1)
    many = infer(texts, Backend(), 8, 4, batch_size=5)
    assert one["labels"] == many["labels"] == [[i % 8 for i in range(len(t))] for t in texts]
    assert one["windows"] == many["windows"]
    assert many["forward_calls"] < one["forward_calls"]
    assert many["restored"][1] == ""


@pytest.mark.parametrize("periods,expected", [([], 4), ([0], 1), ([1, 3], 4), ([4, 5], 4)])
def test_adaptive_last_period_or_progress_fallback(periods, expected):
    encoded = EncodedText.build("甲" * 20, Backend().encode_character)
    window = adaptive_window(encoded, 0, 0, 8)
    labels = [2 if i in periods else 0 for i in range(window.end)]
    assert adaptive_commit_end(encoded, window, labels, 4) == expected


def test_adaptive_final_window_is_fully_committed():
    encoded = EncodedText.build("甲" * 4, Backend().encode_character)
    window = adaptive_window(encoded, 0, 0, 8)
    assert adaptive_commit_end(encoded, window, [2, 0, 0, 0], 2) == 4


def test_adaptive_inference_progresses_and_covers_all_characters():
    output = infer(["甲" * 43], Backend(), 8, 4, strategy="adaptive", batch_size=1)
    assert output["labels"][0] == [i % 8 for i in range(43)]
    assert output["windows"][-1]["core_end"] == 43
    with pytest.raises(ValueError, match="batch"):
        infer(["甲"], Backend(), 8, 4, strategy="adaptive", batch_size=2)


def test_backend_character_loss_is_detected():
    class Broken(Backend):
        def predict_batch(self, encoded, windows):
            return [[] for _ in windows]
    with pytest.raises(ValueError, match="lost"):
        infer(["甲" * 20], Broken(), 8, 4)
