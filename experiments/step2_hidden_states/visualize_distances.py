import argparse
from pathlib import Path

from src.common.data import read_json
from src.common.paths import PROJECT_ROOT, project_path
from src.visualization.feature_distances import plot_distances


def main():
    parser = argparse.ArgumentParser(description="STEP 2 Cosine Distance 집계와 과거 수치 참고 비교를 PNG로 저장")
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = project_path(args.analysis_json).resolve()
    report = read_json(source)
    if args.output_dir:
        output_dir = project_path(args.output_dir)
    else:
        relative_dir = source.parent.relative_to(PROJECT_ROOT / "results")
        if relative_dir.name == "analysis":
            relative_dir = relative_dir.parent
        output_dir = PROJECT_ROOT / "reports" / relative_dir / "distances"
    outputs = plot_distances(report, output_dir)
    for output in outputs:
        print(f"저장: {output}")
    print(f"총 {len(outputs)}개 그림 저장")


if __name__ == "__main__":
    main()
