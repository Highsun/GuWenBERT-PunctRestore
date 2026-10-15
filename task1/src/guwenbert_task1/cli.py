from __future__ import annotations

import argparse
import json

from .common import load_config, setup_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent task-one punctuation restoration workflow")
    parser.add_argument("command", choices=["fetch-data", "prepare", "download", "train", "evaluate", "predict", "verify", "report", "freeze"])
    parser.add_argument("--config", default="configs/cclue_bert.toml")
    parser.add_argument("--text", help="Short text for the single-window demo")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    setup_cache()
    if arguments.command in ("prepare", "fetch-data"):
        from .data import prepare, fetch_data

        result = (prepare if arguments.command == "prepare" else fetch_data)(config)
    elif arguments.command in ("report", "freeze"):
        from . import reporting

        result = getattr(reporting, arguments.command)(config)
    else:
        from . import runtime

        if arguments.command == "predict":
            if arguments.text is None:
                parser.error("predict requires --text")
            result = runtime.predict(config, arguments.text)
        else:
            result = getattr(runtime, {"download": "download_model"}.get(arguments.command, arguments.command))(config)
    if arguments.command == "train":
        result = {key: result[key] for key in ("model_path", "best_epoch", "best_validation_micro_f1", "training_seconds_including_validation_and_export", "peak_allocated_bytes")}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
