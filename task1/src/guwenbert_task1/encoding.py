from __future__ import annotations

from functools import lru_cache


class CharacterEncoder:
    """Keep every Unicode character; budget actual tokens, never truncate silently."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.special_count = tokenizer.num_special_tokens_to_add(pair=False)
        if self.special_count != 2 or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
            raise ValueError("This task-one encoder requires BERT CLS/SEP single sequences")
        if max_length <= self.special_count:
            raise ValueError("No room for content tokens")

    @lru_cache(maxsize=65536)
    def encode_character(self, char: str) -> tuple[int, ...]:
        pieces = self.tokenizer.encode(char, add_special_tokens=False)
        return tuple(pieces or [self.tokenizer.unk_token_id])

    def windows(self, text: str, labels: list[int] | None = None, article_id: str = "input") -> list[dict]:
        if labels is not None and len(text) != len(labels):
            raise ValueError("Character/label length mismatch")
        windows = []
        start, ids, aligned, positions = 0, [], [], []

        def finish(end: int) -> None:
            windows.append({"input_ids": [self.tokenizer.cls_token_id, *ids, self.tokenizer.sep_token_id], "labels": [-100, *aligned, -100], "character_positions": [-1, *positions, -1], "article_id": article_id, "char_start": start, "char_end": end})

        for index, char in enumerate(text):
            pieces = self.encode_character(char)
            if len(pieces) > self.max_length - self.special_count:
                raise ValueError("A single character exceeds the token budget")
            if ids and len(ids) + len(pieces) > self.max_length - self.special_count:
                finish(index)
                start, ids, aligned, positions = index, [], [], []
            ids.extend(pieces)
            aligned.extend([labels[index] if labels is not None else 0] + [-100] * (len(pieces) - 1))
            positions.extend([index] + [-1] * (len(pieces) - 1))
        if ids:
            finish(len(text))
        return windows


def make_windows(rows: list[dict], encoder: CharacterEncoder) -> list[dict]:
    return [window for row in rows for window in encoder.windows(row["text"], row["labels"], row["article_id"])]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict]) -> dict:
        import torch

        length = ((max(len(row["input_ids"]) for row in rows) + 7) // 8) * 8
        ids, masks, labels = [], [], []
        for row in rows:
            padding = length - len(row["input_ids"])
            ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            masks.append([1] * len(row["input_ids"]) + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels)}
