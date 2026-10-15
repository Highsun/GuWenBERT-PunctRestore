"""Verify exported predictions in a fresh process without task-one source imports."""
import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
args = parser.parse_args()
torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(args.model, local_files_only=True).cuda().eval()
references = json.loads((args.model / "reference_predictions.json").read_text())
labels = json.loads((args.model / "labels.json").read_text())
for item in references:
    text = item["normalized"]
    if not text:
        assert item["labels"] == [] and item["output"] == ""
        continue
    ids, first_positions = [tokenizer.cls_token_id], []
    for character in text:
        first_positions.append(len(ids))
        ids.extend(tokenizer.encode(character, add_special_tokens=False) or [tokenizer.unk_token_id])
    ids.append(tokenizer.sep_token_id)
    padding = (-len(ids)) % 8
    mask = [1] * len(ids) + [0] * padding
    ids += [tokenizer.pad_token_id] * padding
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        predicted = model(input_ids=torch.tensor([ids], device="cuda"), attention_mask=torch.tensor([mask], device="cuda")).logits.argmax(-1)[0].cpu().tolist()
    chosen = [predicted[position] for position in first_positions]
    output = "".join(character + (labels[label] if label else "") for character, label in zip(text, chosen))
    assert chosen == item["labels"] and output == item["output"]
assert not any(name.startswith("guwenbert_task1") for name in sys.modules)
result = {"passed": True, "reference_cases": len(references), "task1_source_imported": False, "precision": args.precision, "long_input_rejected_by_task1": True}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
