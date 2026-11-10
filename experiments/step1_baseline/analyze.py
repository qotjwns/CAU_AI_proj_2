import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.paths import project_path, project_relative_path
from src.core.metrics import compute_metrics
from src.core.results import load_results, save_metrics


def main():
    parser = argparse.ArgumentParser(description="추론 결과 JSONL에서 A/B 메트릭 계산")
    parser.add_argument("--results-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = [project_path(path) for path in args.results_jsonl]
    records = load_results(paths)
    report = compute_metrics(records)
    report["source_files"] = [project_relative_path(path) for path in paths]
    output_dir = project_path(args.output_dir)
    save_metrics(report, output_dir)
    for row in report["by_mode"]:
        print(f"Mode {row['mode']}: {row['correct']}/{row['total']} "
              f"({row['accuracy']:.2%}), invalid={row['invalid_count']}")
    print(f"저장: {output_dir}")


if __name__ == "__main__":
    main()
