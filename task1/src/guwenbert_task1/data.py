from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
import re
import unicodedata

from .common import LABELS, LABEL_TO_ID, ROOT, render, sha256, write_json, write_jsonl

LINE_ID = re.compile(r"(\d{8}-\d{2}-\d{3})-(\d{3})/m")
ASCII_MARKS = str.maketrans({",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："})


def parse_line(line: str) -> tuple[str, str]:
    fields = line.split()
    if not fields or (match := LINE_ID.fullmatch(fields[0])) is None:
        raise ValueError(f"Malformed line identifier: {line[:80]!r}")
    words = []
    for field in fields[1:]:
        if "/" not in field:
            raise ValueError(f"Missing POS annotation: {field!r}")
        word, tag = field.rsplit("/", 1)
        if not tag:
            raise ValueError(f"Empty POS annotation: {field!r}")
        # A leading '[' wraps a compound annotation; closing ']' is in the tag.
        words.append(word.lstrip("["))
    return match[1], "".join(words)


def normalize_text(text: str) -> tuple[str, list[int], Counter]:
    chars: list[str] = []
    labels: list[int] = []
    audit: Counter = Counter()
    for index, char in enumerate(text):
        # Preserve numeric decimal/fraction separators in either character width.
        numeric_separator = char in ".．/／" and 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit()
        mark = char.translate(ASCII_MARKS) if not numeric_separator else char
        if not numeric_separator and mark in LABEL_TO_ID and mark != "O":
            if not chars:
                audit["leading_target_punctuation"] += 1
                continue
            label = LABEL_TO_ID[mark]
            if labels[-1]:
                audit["collapsed_target_sequences"] += 1
                # A sentence terminator wins over a following nonterminal mark.
                if LABELS[labels[-1]] in "。？！" and mark not in "。？！":
                    continue
            labels[-1] = label
            audit["target_punctuation"] += 1
        elif unicodedata.category(char) == "Cc" or char.isspace():
            audit[f"removed_control_or_space:U+{ord(char):04X}"] += 1
        elif unicodedata.category(char).startswith("P") and not numeric_separator and char not in "%％‰":
            audit[f"excluded_punctuation:{char}"] += 1
        else:
            chars.append(char)
            labels.append(0)
    return "".join(chars), labels, audit


def split_articles(articles: list[dict], seed: int, train_fraction: float, validation_fraction: float) -> dict[str, list[dict]]:
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 - train_fraction):
        raise ValueError("Invalid split fractions")
    unique = sorted(articles, key=lambda row: row["article_id"])
    random.Random(seed).shuffle(unique)
    train_end = int(len(unique) * train_fraction)
    validation_end = train_end + int(len(unique) * validation_fraction)
    result = {"train": unique[:train_end], "validation": unique[train_end:validation_end], "test": unique[validation_end:]}
    if not all(result.values()):
        raise ValueError("Each split must contain at least one article")
    hashes = [{row["content_sha256"] for row in rows} for rows in result.values()]
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Duplicate normalized article across splits")
    return result


def prepare(config: dict) -> dict:
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    known = {item["path"]: item for item in manifest["files"]}
    by_article: dict[str, list[str]] = {}
    sources = []
    nonempty_lines = 0
    for relative in config["raw_files"]:
        path = ROOT / relative
        digest = sha256(path)
        if relative not in known or digest != known[relative]["sha256"]:
            raise ValueError(f"Raw source differs from audited manifest: {relative}")
        sources.append({"path": relative, "sha256": digest})
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
            if not line.strip():
                continue
            try:
                article_id, text = parse_line(line)
            except ValueError as error:
                raise ValueError(f"{relative}:{number}: {error}") from error
            by_article.setdefault(article_id, []).append(text)
            nonempty_lines += 1
    audit: Counter = Counter()
    articles = []
    seen: dict[str, dict] = {}
    duplicates = []
    for article_id, paragraphs in sorted(by_article.items()):
        text, labels, counts = normalize_text("".join(paragraphs))
        audit.update(counts)
        if not text:
            audit["empty_articles"] += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates.append({"article_id": article_id, "kept_article_id": seen[digest]["article_id"], "conflicting_labels": labels != seen[digest]["labels"]})
            continue
        row = {"article_id": article_id, "content_sha256": digest, "text": text, "labels": labels, "punctuated": render(text, labels)}
        articles.append(row)
        seen[digest] = row
    splits = split_articles(articles, config["seed"], config["train_fraction"], config["validation_fraction"])
    processed = ROOT / "data/processed/task1" / config["dataset_version"]
    split_dir = ROOT / "data/splits/task1" / config["dataset_version"]
    write_jsonl(processed / "articles.jsonl", articles)
    write_json(processed / "duplicates.json", duplicates)
    summaries = {}
    for name, rows in splits.items():
        path = split_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        counts = Counter(label for row in rows for label in row["labels"])
        summaries[name] = {"articles": len(rows), "characters": sum(len(row["text"]) for row in rows), "label_counts": {label: counts[index] for index, label in enumerate(LABELS)}, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
    summary = {"dataset_version": config["dataset_version"], "sources": sources, "seed": config["seed"], "nonempty_lines": nonempty_lines, "source_articles": len(by_article), "deduplicated_articles": len(articles), "duplicate_articles_removed": len(duplicates), "duplicate_label_conflicts": sum(item["conflicting_labels"] for item in duplicates), "labels": LABELS, "audit": dict(audit), "splits": summaries}
    write_json(processed / "summary.json", summary)
    write_json(split_dir / "manifest.json", summary)
    return summary
