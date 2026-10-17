from __future__ import annotations

from collections import defaultdict

from .common import ROOT, digest, read_json, read_rows, write_json, write_rows


def reconstruct(fragments: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in fragments:
        if len(row["text"]) != len(row["labels"]):
            raise ValueError("Source character alignment mismatch")
        groups[row["document_index"]].append(row)
    documents = []
    for document, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["record_index"])
        pieces, labels, spans, joins, offset = [], [], [], [], 0
        for index, row in enumerate(rows):
            if index:
                joins.append(offset)
            pieces.append(row["text"])
            labels.extend(row["labels"])
            spans.append({"article_id": row["article_id"], "start": offset, "end": offset + len(row["text"])})
            offset += len(row["text"])
        documents.append({"document_id": f"test/doc-{document:05d}", "document_index": document,
                          "text": "".join(pieces), "labels": labels, "source_spans": spans, "joins": joins})
    return documents


def prepare(config: dict, receipt: dict) -> dict:
    source = ROOT / receipt["test_source"]["path"]
    if digest(source) != receipt["test_source"]["sha256"]:
        raise ValueError("Task-one test split changed")
    fragments = read_rows(source)
    documents = reconstruct(fragments)
    target = ROOT / "data/splits/task2" / config["dataset_version"]
    write_rows(target / "documents.jsonl", documents, immutable=True)
    write_rows(target / "fragments.jsonl", fragments, immutable=True)
    summary = {"dataset_version": config["dataset_version"],
               "construction": "concatenate official test fragments within anonymous DOCSTART groups; preserve every character and supplied label; no inserted separators",
               "source_path": str(source.relative_to(ROOT)), "source_sha256": digest(source),
               "records": len(documents), "source_fragments": len(fragments),
               "characters": sum(len(row["text"]) for row in documents),
               "min_characters": min(len(row["text"]) for row in documents),
               "max_characters": max(len(row["text"]) for row in documents),
               "longer_than_512_characters": sum(len(row["text"]) > 512 for row in documents),
               "reconstruction_joins": sum(len(row["joins"]) for row in documents),
               "files": {name: digest(target / name) for name in ("documents.jsonl", "fragments.jsonl")}}
    write_json(target / "manifest.json", summary)
    write_json(ROOT / "data/processed/task2" / config["dataset_version"] / "summary.json", summary)
    return summary


def load_dataset(config: dict, kind: str = "documents", limit: int | None = None):
    if kind not in ("documents", "fragments") or (limit is not None and limit < 1):
        raise ValueError("Invalid dataset or limit")
    directory = ROOT / "data/splits/task2" / config["dataset_version"]
    manifest = read_json(directory / "manifest.json")
    if manifest["dataset_version"] != config["dataset_version"]:
        raise ValueError("Dataset version mismatch")
    if digest(ROOT / manifest["source_path"]) != manifest["source_sha256"]:
        raise ValueError("Source test data changed")
    path = directory / f"{kind}.jsonl"
    if digest(path) != manifest["files"][path.name]:
        raise ValueError("Task-two dataset changed")
    rows = read_rows(path)
    if limit:
        rows = rows[:limit]
    return rows, manifest
