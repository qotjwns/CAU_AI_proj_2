import argparse
from pathlib import Path

from src.common.data import read_json
from src.common.paths import PROJECT_ROOT, project_path
from src.visualization.probe import plot_probe_report


def main():
    parser = argparse.ArgumentParser(description="STEP 3 Linear Probe 결과를 PNG로 저장")
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="기본: results/ 경로에 대응하는 reports/ 경로")
    args = parser.parse_args()
    source = project_path(args.analysis_json).resolve()
    report = read_json(source)
    if args.output_dir:
        output_dir = project_path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "reports" / source.parent.relative_to(PROJECT_ROOT / "results")
    outputs = plot_probe_report(report, output_dir)
    for output in outputs:
        print(f"저장: {output}")
    print(f"총 {len(outputs)}개 그림 저장")


if __name__ == "__main__":
    main()
