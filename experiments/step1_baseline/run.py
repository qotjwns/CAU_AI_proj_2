import argparse
from itertools import islice
import json
from pathlib import Path

from tqdm import tqdm

from src.common.config import load_inference_config
from src.common.data import iter_jsonl
from src.common.paths import PROJECT_ROOT, project_path
from src.core.inference import infer_case
from src.core.model import load_model
from src.core.results import build_result


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step1.yaml"


def run_baseline(config_path, limit=None, output_dir=None):
    config = load_inference_config(config_path)
    destination_dir = project_path(output_dir) if output_dir else config["output_dir"]
    if limit is not None and output_dir is None:
        destination_dir /= f"pilot_{limit}"
    model, processor, answer_token_ids = load_model(config["model_name"])
    destination_dir.mkdir(parents=True, exist_ok=True)
    remaining = limit

    for cases_path in config["cases_jsonl"]:
        if remaining == 0:
            break
        output_name = f"{cases_path.stem.removesuffix('_cases')}_results.jsonl"
        output_path = destination_dir / output_name
        with cases_path.open(encoding="utf-8") as source:
            total = sum(1 for line in source if line.strip())
            if remaining is not None:
                total = min(total, remaining)
            if total == 0:
                continue
            source.seek(0)
            cases = islice(iter_jsonl(source), total)
            with output_path.open("x", encoding="utf-8") as destination:
                for case in tqdm(cases, total=total, desc=cases_path.stem, unit="case"):
                    inference = infer_case(case, model, processor, answer_token_ids)
                    result = build_result(case, inference, config)
                    destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                    destination.flush()
            if remaining is not None:
                remaining -= total
        print(f"저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="JSONL 케이스를 순서대로 baseline 추론")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, help="전체 입력에서 추론할 최대 케이스 수")
    parser.add_argument("--output-dir", type=Path, help="결과 저장 폴더")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다.")
    run_baseline(args.config, args.limit, args.output_dir)


if __name__ == "__main__":
    main()
