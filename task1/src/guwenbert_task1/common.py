from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "task1"
LABELS = ["O", "，", "。", "？", "！", "、", "：", "；"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TERMINALS = {LABEL_TO_ID[mark] for mark in "。？！"}
NATIVE_TAGS = ["O", "B-,", "B-.", "B-?", "B-!", "B-\\", "B-:", "B-;"]
TAG_TO_ID = {tag: index for index, tag in enumerate(NATIVE_TAGS)}


def setup_cache() -> None:
    # Set before importing Hugging Face libraries. Never use home-directory caches.
    os.environ["HF_HOME"] = str(TASK / ".cache/huggingface")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["XDG_CACHE_HOME"] = str(TASK / ".cache")


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = TASK / path
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for key in ("experiment", "dataset_version"):
        if Path(config[key]).name != config[key] or config[key] in (".", ".."):
            raise ValueError(f"Invalid {key}")
    if config["epochs"] < 1 or config["max_length"] < 3:
        raise ValueError("epochs and max_length must be positive and usable")
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def render(text: str, labels: list[int]) -> str:
    if len(text) != len(labels):
        raise ValueError("Character/label length mismatch")
    return "".join(char + (LABELS[label] if label else "") for char, label in zip(text, labels))
