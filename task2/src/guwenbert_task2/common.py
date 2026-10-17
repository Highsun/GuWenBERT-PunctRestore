from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "task2"
LABELS = ["O", "，", "。", "？", "！", "、", "：", "；"]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict], immutable: bool = False) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing dataset differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_config(value: str) -> dict:
    path = Path(value)
    if not path.is_absolute():
        path = TASK / path
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in ("experiment", "dataset_version", "model_experiment"):
        if Path(config[key]).name != config[key] or config[key] in (".", ".."):
            raise ValueError(f"Invalid {key}")
    if not 2 < config["max_length"] <= 512 or config["max_length"] % 8:
        raise ValueError("max_length must be a multiple of eight, at most 512")
    if not 0 < config["core_tokens"] <= config["max_length"] - 2:
        raise ValueError("core_tokens exceeds content budget")
    if config["batch_size"] < 1 or any(b < 1 for b in config["batch_sizes"]):
        raise ValueError("batch sizes must be positive")
    if config["repetitions"] < 2 or config["warmup_runs"] < 1 or config["boundary_margin"] < 0:
        raise ValueError("Use warmup >= 1, repetitions >= 2 and margin >= 0")
    if config["precision"] not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    return config


def setup_cache() -> None:
    os.environ["HF_HOME"] = str(TASK / ".cache/huggingface")
    os.environ["XDG_CACHE_HOME"] = str(TASK / ".cache")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def source_hashes() -> dict:
    paths = [*TASK.glob("src/**/*.py"), *TASK.glob("scripts/*.py"),
             *TASK.glob("configs/*.toml"), TASK / "pyproject.toml", TASK / "uv.lock"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def verify_task1(config: dict) -> dict:
    """Read-only checks of the frozen code, model, corpora and run artifacts."""
    path = ROOT / "task1/reports/freeze-manifest.json"
    manifest = read_json(path)
    if manifest["intended_tag"] != config["task1_tag"]:
        raise ValueError("Task-one freeze tag mismatch")
    checks = dict(manifest["reports"])
    for entry in manifest["experiments"].values():
        checks.update(entry["source_files_sha256"])
        checks.update({str(Path(entry["model_path"]) / name): sha
                       for name, sha in entry["model_files_sha256"].items()})
        checks.update(entry["outputs"])
        for item in entry["dataset"]["sources"]:
            checks[item["path"]] = item["sha256"]
        for item in entry["dataset"]["splits"].values():
            checks[item["path"]] = item["sha256"]
    tracked = subprocess.check_output(["git", "ls-files", "task1"], cwd=ROOT, text=True).splitlines()
    for relative in tracked:
        checks.setdefault(relative, digest(ROOT / relative))
    checks[str(path.relative_to(ROOT))] = digest(path)
    for relative, expected in checks.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen task-one file differs: {relative}")
    entry = manifest["experiments"][config["model_experiment"]]
    commit = subprocess.check_output(["git", "rev-parse", config["task1_tag"] + "^{}"], cwd=ROOT, text=True).strip()
    return {"task1_tag": config["task1_tag"], "task1_commit": commit,
            "freeze_manifest_sha256": digest(path), "model_path": entry["model_path"],
            "model_files_sha256": entry["model_files_sha256"],
            "test_source": entry["dataset"]["splits"]["test"], "verified_files": checks}


def assert_protected(receipt: dict) -> None:
    for relative, expected in receipt["verified_files"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Task-one file changed during task-two run: {relative}")


def render(text: str, labels: list[int]) -> str:
    if len(text) != len(labels) or any(not 0 <= label < len(LABELS) for label in labels):
        raise ValueError("Invalid character predictions")
    return "".join(char + (LABELS[label] if label else "") for char, label in zip(text, labels))
