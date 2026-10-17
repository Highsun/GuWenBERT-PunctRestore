from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, assert_protected, load_config, setup_cache, verify_task1, write_json


def main():
    parser = argparse.ArgumentParser(description="Independent task-two long-text punctuation inference")
    parser.add_argument("command", choices=["prepare", "verify", "predict", "evaluate", "benchmark"])
    parser.add_argument("--config", default="configs/initial.toml")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--text")
    inputs.add_argument("--input", type=Path, help="UTF-8 text file")
    parser.add_argument("--output", type=Path, help="JSON prediction output below code/outputs/task2")
    parser.add_argument("--strategy", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dataset", choices=["documents", "fragments"], default="documents")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    setup_cache()
    receipt = verify_task1(cfg)
    if args.command == "prepare":
        from .data import prepare
        result = prepare(cfg, receipt)
    else:
        from . import runtime
        if args.command == "verify":
            result = runtime.verify(cfg, receipt)
        elif args.command == "benchmark":
            result = runtime.benchmark(cfg, receipt, args.limit)
        elif args.command == "evaluate":
            batch = args.batch_size if args.batch_size is not None else (1 if args.strategy == "adaptive" else cfg["batch_size"])
            result = runtime.evaluate(cfg, receipt, args.dataset, args.limit, args.strategy, batch)
        else:
            if args.text is None and args.input is None:
                parser.error("predict requires --text or --input")
            text = args.text if args.text is not None else args.input.read_text(encoding="utf-8", errors="strict")
            clean = runtime.normalize(text)
            backend = runtime.TorchBackend(cfg, receipt)
            batch = args.batch_size if args.batch_size is not None else (1 if args.strategy == "adaptive" else cfg["batch_size"])
            value = runtime.run(cfg, backend, [{"text": clean}], args.strategy, batch)
            result = {"input_characters": len(text), "normalized_characters": len(clean),
                      "normalized": clean, "output": value["restored"][0], "labels": value["labels"][0],
                      "windows": len(value["windows"]), "seconds": value["seconds"],
                      "strategy": args.strategy, "batch_size": batch, "profile": value["profile"]}
            if args.output:
                path = args.output if args.output.is_absolute() else ROOT / args.output
                if not path.resolve().is_relative_to((ROOT / "outputs/task2").resolve()):
                    parser.error("prediction output must be below code/outputs/task2")
                if path.exists():
                    parser.error("output already exists; choose a new filename")
                write_json(path, result)
    assert_protected(receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
