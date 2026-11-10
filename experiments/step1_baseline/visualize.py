import argparse
from pathlib import Path

from src.common.data import read_json
from src.common.paths import project_path
from src.visualization.plots import plot_report


def main():
    parser = argparse.ArgumentParser(description="계산된 메트릭 JSON을 PNG 차트로 시각화")
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = project_path(args.metrics_json)
    report = read_json(source)
    output_dir = project_path(args.output_dir) if args.output_dir else source.parent / "plots"
    plot_report(report, output_dir)
    print(f"저장: {output_dir}")


if __name__ == "__main__":
    main()
