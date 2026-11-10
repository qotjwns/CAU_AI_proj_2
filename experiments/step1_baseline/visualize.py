import argparse
from pathlib import Path

from src.common.data import read_json
from src.common.paths import PROJECT_ROOT, project_path
from src.visualization.plots import plot_report


def main():
    parser = argparse.ArgumentParser(description="계산된 메트릭 JSON을 PNG 차트로 시각화")
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="기본: results/ 경로에 대응하는 reports/ 경로 아래 plots/")
    args = parser.parse_args()
    source = project_path(args.metrics_json).resolve()
    report = read_json(source)
    if args.output_dir:
        output_dir = project_path(args.output_dir)
    else:
        relative_dir = source.parent.relative_to(PROJECT_ROOT / "results")
        output_dir = PROJECT_ROOT / "reports" / relative_dir / "plots"
    plot_report(report, output_dir)
    print(f"저장: {output_dir}")


if __name__ == "__main__":
    main()
