from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass


@dataclass
class EncodedText:
    text: str
    token_ids: list[int]
    prefix: list[int]

    @classmethod
    def build(cls, text: str, encode_character):
        ids, prefix = [], [0]
        for character in text:
            pieces = encode_character(character)
            if not pieces:
                raise ValueError("Backend must replace empty encodings with UNK")
            ids.extend(pieces)
            prefix.append(len(ids))
        return cls(text, ids, prefix)

    def end_within(self, start: int, budget: int) -> int:
        return bisect_right(self.prefix, self.prefix[start] + budget) - 1


@dataclass(frozen=True)
class Window:
    document: int
    start: int
    end: int
    core_start: int
    core_end: int

    def input_tokens(self, encoded: EncodedText) -> int:
        return encoded.prefix[self.end] - encoded.prefix[self.start] + 2

    def metadata(self, encoded: EncodedText) -> dict:
        return {"document": self.document, "start": self.start, "end": self.end,
                "core_start": self.core_start, "core_end": self.core_end,
                "input_tokens": self.input_tokens(encoded)}


def fixed_windows(encoded: EncodedText, document: int, max_length: int, core_tokens: int) -> list[Window]:
    budget = max_length - 2
    if not 0 < core_tokens <= budget:
        raise ValueError("Invalid core budget")
    if not encoded.text:
        return []
    if encoded.prefix[-1] <= budget:
        return [Window(document, 0, len(encoded.text), 0, len(encoded.text))]
    left_budget = (budget - core_tokens) // 2
    right_budget = budget - core_tokens - left_budget
    result, start = [], 0
    while start < len(encoded.text):
        end = encoded.end_within(start, core_tokens)
        if end <= start:
            raise ValueError("A character exceeds core token budget")
        left = bisect_left(encoded.prefix, encoded.prefix[start] - left_budget)
        right = encoded.end_within(end, right_budget)
        window = Window(document, left, right, start, end)
        if window.input_tokens(encoded) > max_length:
            raise AssertionError("Window exceeds actual token budget")
        result.append(window)
        start = end
    return result


def adaptive_window(encoded: EncodedText, document: int, start: int, max_length: int) -> Window:
    end = encoded.end_within(start, max_length - 2)
    if end <= start:
        raise ValueError("A character exceeds input token budget")
    return Window(document, start, end, start, end)


def adaptive_commit_end(encoded: EncodedText, window: Window, predicted: list[int], core_tokens: int) -> int:
    if len(predicted) != window.end - window.start:
        raise ValueError("Prediction length differs from window")
    if window.end == len(encoded.text):
        return window.end
    target = min(window.end, encoded.end_within(window.start, core_tokens))
    if target <= window.start:
        raise ValueError("A character exceeds core token budget")
    periods = [i for i in range(target - window.start) if predicted[i] == 2]
    return window.start + periods[-1] + 1 if periods else target
