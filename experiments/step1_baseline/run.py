import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from src.common.config import load_inference_config
from src.common.data import iter_jsonl
from src.common.paths import PROJECT_ROOT
from src.core.inference import infer_case
from src.core.model import load_model
from src.core.results import build_result


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step1.yaml"


def run_baseline(config_path):
    config = load_inference_config(config_path)
    model, processor, answer_token_ids = load_model(config["model_name"])
    config["output_dir"].mkdir(parents=True, exist_ok=True)

    for cases_path in config["cases_jsonl"]:
        output_name = f"{cases_path.stem.removesuffix('_cases')}_results.jsonl"
        output_path = config["output_dir"] / output_name
        with (
            cases_path.open(encoding="utf-8") as source,
            output_path.open("x", encoding="utf-8") as destination,
        ):
            total = sum(1 for line in source if line.strip())
            source.seek(0)
            for case in tqdm(iter_jsonl(source), total=total, desc=cases_path.stem, unit="case"):
                inference = infer_case(case, model, processor, answer_token_ids)
                result = build_result(case, inference, config)
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                destination.flush()
        print(f"저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="JSONL 케이스를 순서대로 baseline 추론")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run_baseline(args.config)


if __name__ == "__main__":
    main()
