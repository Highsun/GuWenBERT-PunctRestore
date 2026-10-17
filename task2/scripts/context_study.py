from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guwenbert_task2.common import TASK, load_config, read_json, verify_task1

parser = argparse.ArgumentParser(description="Sentence-informed context-budget study on development documents")
parser.add_argument("command", choices=["run", "plot"])
parser.add_argument("--config", default="configs/context-study.toml")
args = parser.parse_args()
config = load_config(args.config)
if args.command == "run":
    from guwenbert_task2.context_study import run_study
    result = run_study(config, verify_task1(config))
else:
    from guwenbert_task2.context_figures import make_figures, write_report
    directory = TASK / "reports" / config["experiment"]
    saved = read_json(directory / "results.json")
    make_figures(saved, directory / "figures")
    write_report(saved, directory)
    result = {"report": str(directory / "experiment.md")}
print(json.dumps(result, ensure_ascii=False, indent=2))
