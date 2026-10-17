from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guwenbert_task2.common import load_config, verify_task1
from guwenbert_task2.batch_study import run_phase

parser = argparse.ArgumentParser(description="Development batch scaling and locked test evaluation")
parser.add_argument("phase", choices=["development", "test"])
parser.add_argument("--config", default="configs/batch-study.toml")
args = parser.parse_args()
config = load_config(args.config)
print(json.dumps(run_phase(config, verify_task1(config), args.phase), ensure_ascii=False, indent=2))
