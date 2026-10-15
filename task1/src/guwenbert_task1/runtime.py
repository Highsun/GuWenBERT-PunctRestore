from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from .common import LABELS, ROOT, TASK, read_jsonl, render, setup_cache, sha256, write_json, write_jsonl

setup_cache()

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .encoding import CharacterEncoder, Collator, make_windows
from .metrics import add_confusion, scores


def paths(config: dict) -> tuple[Path, Path]:
    return ROOT / "models/task1" / config["experiment"], ROOT / "outputs/task1" / config["experiment"]


def download_model(config: dict) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(config["model_id"], revision=config["model_revision"])
    target = ROOT / "models/task1/pretrained" / config["model_id"].replace("/", "--") / info.sha
    available = {item.rfilename for item in info.siblings}
    weight = "model.safetensors" if "model.safetensors" in available else "pytorch_model.bin"
    snapshot_download(repo_id=config["model_id"], revision=info.sha, local_dir=target, allow_patterns=["config.json", "vocab.txt", weight, "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md"])
    result = {"model_id": config["model_id"], "requested_revision": config["model_revision"], "resolved_revision": info.sha, "local_path": str(target.relative_to(ROOT)), "files": {path.name: sha256(path) for path in target.iterdir() if path.is_file()}}
    write_json(TASK / "reports" / config["experiment"] / "pretrained-model.json", result)
    return result


def device_and_precision(config: dict):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run with WSL GPU access; do not silently substitute CPU training.")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but is unsupported")
    if config["precision"] not in ("bf16", "fp32"):
        raise ValueError("Supported precision: bf16 or fp32")
    return torch.device("cuda"), torch.bfloat16 if config["precision"] == "bf16" else None


def autocast(dtype):
    return torch.autocast("cuda", dtype=dtype) if dtype else nullcontext()


def evaluate_loader(model, loader, device, dtype, prediction_path: Path | None = None) -> dict:
    model.eval()
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    baseline_confusion = np.zeros_like(confusion)
    loss_sum, count = 0.0, 0
    prediction_rows = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with autocast(dtype):
                output = model(**batch)
            gold = batch["labels"].cpu().numpy()
            predicted = output.logits.argmax(-1).cpu().numpy()
            valid_count = int((gold != -100).sum())
            loss_sum += float(output.loss) * valid_count
            count += valid_count
            add_confusion(confusion, gold, predicted)
            add_confusion(baseline_confusion, gold, np.zeros_like(predicted))
            if prediction_path is not None:
                for target, pred in zip(gold, predicted):
                    valid = target != -100
                    prediction_rows.append({"gold": target[valid].tolist(), "predicted": pred[valid].tolist()})
    result = scores(confusion)
    result["loss"] = loss_sum / count if count else 0.0
    result["all_O_baseline"] = scores(baseline_confusion)
    if prediction_path is not None:
        write_jsonl(prediction_path, prediction_rows)
    return result


def validate_dataset(config: dict) -> tuple[dict, Path]:
    split_dir = ROOT / "data/splits/task1" / config["dataset_version"]
    manifest = json.loads((split_dir / "manifest.json").read_text())
    if manifest["dataset_version"] != config["dataset_version"] or manifest["revision"] != config["dataset_revision"] or manifest["labels"] != LABELS:
        raise ValueError("Dataset version mismatch")
    for relative in manifest["sources"]:
        if sha256(ROOT / relative["path"]) != relative["sha256"]:
            raise ValueError("Raw corpus changed since preprocessing")
    for name, item in manifest["splits"].items():
        if sha256(split_dir / f"{name}.jsonl") != item["sha256"]:
            raise ValueError(f"Split checksum mismatch: {name}")
    return manifest, split_dir


def environment() -> dict:
    import transformers

    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0)), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory, "torch_arch_list": torch.cuda.get_arch_list(), "platform": platform.platform(), "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "git_dirty_at_run": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)), "task1_lock_sha256": sha256(TASK / "uv.lock")}


def source_hashes() -> dict:
    files = [*TASK.glob("src/**/*.py"), *TASK.glob("scripts/*.py"), *TASK.glob("configs/*.toml"), TASK / "pyproject.toml", TASK / "uv.lock"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)}


def token_limit(model_config) -> int:
    offset = model_config.pad_token_id + 1 if model_config.model_type == "roberta" else 0
    return model_config.max_position_embeddings - offset


def train(config: dict) -> dict:
    device, dtype = device_and_precision(config)
    torch.set_num_threads(config["num_threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    # Seeded but not a promise of bit-for-bit CUDA reproducibility.
    torch.backends.cudnn.benchmark = False
    manifest, split_dir = validate_dataset(config)
    initial_source = source_hashes()
    source = json.loads((TASK / "reports" / config["experiment"] / "pretrained-model.json").read_text())
    if source["model_id"] != config["model_id"] or source["resolved_revision"] != config["model_revision"]:
        raise ValueError("Pretrained model source mismatch")
    pretrained = ROOT / source["local_path"]
    for name, digest in source["files"].items():
        if sha256(pretrained / name) != digest:
            raise ValueError(f"Pretrained file changed: {name}")
    model_dir, output_dir = paths(config)
    if (output_dir / "training.json").exists():
        raise FileExistsError("Completed experiment exists; choose a new experiment identifier")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(pretrained, local_files_only=True, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={label: index for index, label in enumerate(LABELS)}, attn_implementation="sdpa")
    if tokenizer.pad_token_id != model.config.pad_token_id:
        raise ValueError("Tokenizer and model disagree about padding token")
    if config["max_length"] > token_limit(model.config) or config["max_length"] % 8:
        raise ValueError("max_length must be a multiple of 8 within model position limit")
    model.to(device)
    encoder = CharacterEncoder(tokenizer, config["max_length"])
    datasets = {name: make_windows(read_jsonl(split_dir / f"{name}.jsonl"), encoder) for name in ("train", "validation")}
    collator = Collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=True, generator=generator, collate_fn=collator, pin_memory=True)
    validation_loader = DataLoader(datasets["validation"], batch_size=config["eval_batch_size"], shuffle=False, collate_fn=collator, pin_memory=True)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}], lr=config["learning_rate"])
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config["warmup_fraction"]), total_steps)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    best_f1, best_epoch, step = -1.0, 0, 0
    history = []
    env = environment()
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "environment.json", env)
    write_json(output_dir / "dataset.json", manifest)
    with (output_dir / "progress.jsonl").open("w", encoding="utf-8", buffering=1) as progress:
        def log(value):
            line = json.dumps(value, ensure_ascii=False)
            print(line, flush=True)
            progress.write(line + "\n")

        log({"event": "start", "train_windows": len(datasets["train"]), "validation_windows": len(datasets["validation"]), "steps_per_epoch": len(train_loader), "epochs": config["epochs"], "parameters": sum(parameter.numel() for parameter in model.parameters()), "gpu": env["gpu"]})
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            loss_sum, token_count = 0.0, 0
            for batch in train_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with autocast(dtype):
                    output = model(**batch)
                if not torch.isfinite(output.loss):
                    raise RuntimeError(f"Nonfinite training loss at step {step}")
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                valid = int((batch["labels"] != -100).sum())
                loss_sum += float(output.loss.detach()) * valid
                token_count += valid
                step += 1
                if step % 50 == 0:
                    log({"event": "step", "epoch": epoch, "step": step, "total_steps": total_steps, "mean_train_loss": loss_sum / token_count, "elapsed_seconds": round(time.perf_counter() - started, 2)})
            validation = evaluate_loader(model, validation_loader, device, dtype)
            f1 = validation["punctuation_micro"]["f1"]
            entry = {"epoch": epoch, "step": step, "train_loss": loss_sum / token_count, "validation": validation}
            history.append(entry)
            write_json(output_dir / "history.json", history)
            log({"event": "epoch", "epoch": epoch, "train_loss": entry["train_loss"], "validation_micro_f1": f1, "validation_macro_f1": validation["punctuation_macro_f1"]})
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(model_dir, safe_serialization=True)
                tokenizer.save_pretrained(model_dir)
                write_json(model_dir / "labels.json", LABELS)
                write_json(model_dir / "inference_contract.json", {"schema_version": 2, "labels": LABELS, "prediction": "CCLUE punc_tags: one punctuation label after each Unicode character", "alignment": "tokenize each character independently; label first subtoken; remaining subtokens and special tokens ignored", "empty_character_encoding": "use tokenizer.unk_token_id", "max_length_including_special_tokens": config["max_length"], "effective_model_token_limit": token_limit(model.config), "special_tokens": 2, "training_normalization": "none: preserve supplied CCLUE tokens and punc_tags exactly", "demo_removals": "，。？！、：；“”‘’《》〈〉 plus whitespace and Cc controls; retain other characters", "sentence_boundary_labels": ["。", "？", "！"], "segmentation_labels": LABELS[1:], "pretrained_source": source, "dataset_version": config["dataset_version"], "precision": config["precision"]})
        torch.cuda.synchronize()
    if source_hashes() != initial_source:
        raise RuntimeError("Source changed while training; do not freeze this experiment")
    result = {"config": config, "environment": env, "source_files_sha256": initial_source, "pretrained_source": source, "dataset": manifest, "train_windows": len(datasets["train"]), "validation_windows": len(datasets["validation"]), "total_steps": step, "best_epoch": best_epoch, "best_validation_micro_f1": best_f1, "training_seconds_including_validation_and_export": time.perf_counter() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "history": history, "model_path": str(model_dir.relative_to(ROOT))}
    write_json(output_dir / "training.json", result)
    return result


def evaluate(config: dict) -> dict:
    device, dtype = device_and_precision(config)
    torch.set_num_threads(config["num_threads"])
    manifest, split_dir = validate_dataset(config)
    model_dir, output_dir = paths(config)
    training = json.loads((output_dir / "training.json").read_text())
    if training["config"] != config:
        raise ValueError("Evaluation config differs from the training experiment")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True, attn_implementation="sdpa").to(device)
    encoder = CharacterEncoder(tokenizer, config["max_length"])
    windows = make_windows(read_jsonl(split_dir / "test.jsonl"), encoder)
    loader = DataLoader(windows, batch_size=config["eval_batch_size"], shuffle=False, collate_fn=Collator(tokenizer.pad_token_id), pin_memory=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = evaluate_loader(model, loader, device, dtype, output_dir / "test-predictions.jsonl")
    torch.cuda.synchronize()
    metrics.update(experiment=config["experiment"], best_epoch=training["best_epoch"], test_windows=len(windows), test_records=manifest["splits"]["test"]["records"], evaluation_seconds_including_prediction_write=time.perf_counter() - started, model_files={path.name: sha256(path) for path in model_dir.iterdir() if path.is_file()}, test_data_sha256=manifest["splits"]["test"]["sha256"])
    # Store window identity separately so test predictions can be traced to articles.
    write_jsonl(output_dir / "test-window-index.jsonl", [{key: row[key] for key in ("article_id", "char_start", "char_end")} for row in windows])
    write_json(output_dir / "test.json", metrics)
    return metrics


def predict(config: dict, text: str, model_dir: Path | None = None) -> dict:
    from .data import normalize_input

    model_dir = model_dir or paths(config)[0]
    clean, audit = normalize_input(text)
    if not clean:
        return {"input": text, "normalized": "", "output": "", "labels": [], "normalization_audit": dict(audit)}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    contract = json.loads((model_dir / "inference_contract.json").read_text())
    encoder = CharacterEncoder(tokenizer, contract["max_length_including_special_tokens"])
    windows = encoder.windows(clean)
    if len(windows) != 1:
        raise ValueError("Task-one demo accepts one token window; long-text inference belongs to task two")
    device, dtype = device_and_precision(config)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True, attn_implementation="sdpa").to(device).eval()
    batch = Collator(tokenizer.pad_token_id)(windows)
    with torch.inference_mode(), autocast(dtype):
        predicted = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits.argmax(-1)[0].cpu().tolist()
    labels = [label for label, position in zip(predicted, windows[0]["character_positions"]) if position >= 0]
    return {"input": text, "normalized": clean, "output": render(clean, labels), "labels": labels, "normalization_audit": dict(audit)}


def verify(config: dict) -> dict:
    model_dir, output_dir = paths(config)
    texts = ["学而时习之不亦说乎有朋自远方来不亦乐乎", "太宗谓侍臣曰为君之道必须先存百姓", "ABC123𠀀\u0301", ""]
    references = [predict(config, text) for text in texts]
    try:
        predict(config, "甲" * 1000)
    except ValueError:
        pass
    else:
        raise AssertionError("Long input was not rejected")
    write_json(model_dir / "reference_predictions.json", references)
    subprocess.run([sys.executable, str(TASK / "scripts/verify_export.py"), "--model", str(model_dir), "--output", str(output_dir / "export-validation.json"), "--precision", config["precision"]], check=True)
    return json.loads((output_dir / "export-validation.json").read_text())
