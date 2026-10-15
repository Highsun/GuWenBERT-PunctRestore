from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

from .common import LABELS, ROOT, TAG_TO_ID, render, sha256, write_json, write_jsonl

REPOSITORY = "https://github.com/Ethan-yt/CCLUE"
UPSTREAM_FILES = [
    "data/punctuation/train.txt", "data/punctuation/dev.txt", "data/punctuation/test.txt",
    "data/punctuation/punctuation.py", "data/punctuation.ipynb", "README.md", "LICENSE",
    "run_sequence_labeling.py",
]
INPUT_REMOVALS = set("，。？！、：；“”‘’《》〈〉")


def fetch_data(config: dict) -> dict:
    import requests

    revision = config["dataset_revision"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("Dataset revision must be an immutable Git commit")
    entries = []
    for relative in UPSTREAM_FILES:
        url = f"https://raw.githubusercontent.com/Ethan-yt/CCLUE/{revision}/{relative}"
        path = ROOT / "data/raw/cclue" / revision / relative
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        content = response.content
        content.decode("utf-8", errors="strict")
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Existing immutable source differs: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append({"upstream_path": relative, "path": str(path.relative_to(ROOT)), "url": url, "size_bytes": len(content), "sha256": sha256(path)})
    manifest = {"schema_version": 2, "dataset": "CCLUE punctuation", "repository": REPOSITORY, "revision": revision, "encoding": "utf-8", "repository_license": "Apache-2.0", "files": entries}
    write_json(ROOT / "data/manifest.json", manifest)
    return manifest


def parse_conll(path: Path, split: str) -> list[dict]:
    """Read supplied character labels, keeping each nonempty upstream record intact."""
    rows, tokens, labels, segments, quotes, books = [], [], [], [], [], []
    document = -1
    flags: list[str] = []

    def finish() -> None:
        nonlocal tokens, labels, segments, quotes, books
        if not tokens:
            return
        text = "".join(tokens)
        rows.append({"article_id": f"{split}/doc-{document:05d}/record-{len(rows):06d}", "record_index": len(rows), "document_index": document, "document_flags": flags.copy(), "text": text, "labels": labels, "seg_tags": segments, "quote_tags": quotes, "book_tags": books, "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "punctuated": render(text, labels)})
        tokens, labels, segments, quotes, books = [], [], [], [], []

    with path.open(encoding="utf-8", errors="strict") as stream:
        for number, line in enumerate(stream, 1):
            if line.startswith("-DOCSTART-"):
                finish()
                document += 1
                flags = line.split()[1:]
            elif not line.strip():
                finish()
            else:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"{path}:{number}: expected five CoNLL columns")
                char, seg, punc, quote, book = fields
                if len(char) != 1 or char == "\ufffd" or unicodedata.category(char) == "Cc":
                    raise ValueError(f"{path}:{number}: invalid character token {char!r}")
                if punc not in TAG_TO_ID or seg not in ("O", "B") or quote not in ("O", "B", "I") or book not in ("O", "B", "I"):
                    raise ValueError(f"{path}:{number}: unknown label")
                if (seg == "B") != (punc != "O"):
                    raise ValueError(f"{path}:{number}: segmentation/punctuation disagreement")
                tokens.append(char)
                labels.append(TAG_TO_ID[punc])
                segments.append(seg)
                quotes.append(quote)
                books.append(book)
    finish()
    if not rows:
        raise ValueError(f"No nonempty records: {path}")
    return rows


def clean_training(rows: list[dict], validation: list[dict], test: list[dict]) -> tuple[list[dict], list[dict]]:
    # Holdout labels are not used: prevent exact text leakage, preserve holdout sets.
    holdout = {row["content_sha256"] for row in validation + test}
    seen = set()
    kept, removed = [], []
    for row in rows:
        digest = row["content_sha256"]
        reason = "exact_text_in_holdout" if digest in holdout else ("duplicate_train_text" if digest in seen else None)
        if reason:
            removed.append({"article_id": row["article_id"], "content_sha256": digest, "reason": reason})
        else:
            kept.append(row)
            seen.add(digest)
    return kept, removed


def normalize_input(text: str) -> tuple[str, dict]:
    # Match the upstream target definition; never normalize supplied training text.
    kept, audit = [], Counter()
    for char in text:
        if char in INPUT_REMOVALS or char.isspace() or unicodedata.category(char) == "Cc":
            audit[f"removed:U+{ord(char):04X}"] += 1
        else:
            kept.append(char)
    return "".join(kept), dict(audit)


def prepare(config: dict) -> dict:
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    if manifest["revision"] != config["dataset_revision"]:
        raise ValueError("CCLUE revision mismatch")
    for item in manifest["files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Raw file checksum mismatch: {item['path']}")
    raw_dir = ROOT / "data/raw/cclue" / config["dataset_revision"] / "data/punctuation"
    official = {name: parse_conll(raw_dir / f"{upstream}.txt", name) for name, upstream in (("train", "train"), ("validation", "dev"), ("test", "test"))}
    train, removed = clean_training(official["train"], official["validation"], official["test"])
    splits = {"train": train, "validation": official["validation"], "test": official["test"]}
    processed = ROOT / "data/processed/task1" / config["dataset_version"]
    split_dir = ROOT / "data/splits/task1" / config["dataset_version"]
    write_json(processed / "removed-training-records.json", removed)
    summary = {"dataset_version": config["dataset_version"], "repository": REPOSITORY, "revision": config["dataset_revision"], "sources": manifest["files"], "labels": LABELS, "native_label_mapping": TAG_TO_ID, "official_nonempty_records": {key: len(rows) for key, rows in official.items()}, "official_readme_records": {"train": 26935, "validation": 4075, "test": 3992}, "removed_training": dict(Counter(row["reason"] for row in removed)), "split_policy": "preserve official dev/test; remove training exact text overlap and within-train duplicates; no random resplit", "splits": {}}
    for name, rows in splits.items():
        path = split_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        counts = Counter(label for row in rows for label in row["labels"])
        duplicates = Counter(row["content_sha256"] for row in rows)
        summary["splits"][name] = {"records": len(rows), "articles": len(rows), "documents": len({row["document_index"] for row in rows}), "characters": sum(len(row["text"]) for row in rows), "min_characters": min(len(row["text"]) for row in rows), "max_characters": max(len(row["text"]) for row in rows), "duplicate_text_records": sum(count - 1 for count in duplicates.values()), "label_counts": {label: counts[index] for index, label in enumerate(LABELS)}, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
    write_json(processed / "summary.json", summary)
    write_json(split_dir / "manifest.json", summary)
    return summary
