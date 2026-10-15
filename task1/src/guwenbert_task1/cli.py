from __future__ import annotations

import argparse
import json

from .common import load_config, setup_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent task-one punctuation restoration workflow")
    parser.add_argument("command", choices=["prepare", "download", "train", "evaluate", "predict", "report"])
    parser.add_argument("--config", default="configs/pd9801.toml")
    parser.add_argument("--text", help="Short text for the single-window demo")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    setup_cache()
    if arguments.command == "prepare":
        from .data import prepare

        result = prepare(config)
    elif arguments.command == "report":
        from .reporting import report

        result = report(config)
    else:
        from . import runtime

        if arguments.command == "predict":
            if arguments.text is None:
                parser.error("predict requires --text")
            result = runtime.predict(config, arguments.text)
        else:
            result = getattr(runtime, {"download": "download_model"}.get(arguments.command, arguments.command))(config)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
