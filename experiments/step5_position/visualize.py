import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import read_json
from src.common.paths import PROJECT_ROOT, project_path
from src.visualization.position import plot_position_report


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step5_position_eval.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 5-2b 평가 그림 (analysis.json만 읽음)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--analysis-json", type=Path, help="기본: 설정의 output_dir/analysis.json")
    parser.add_argument("--output-dir", type=Path, help="기본: 설정의 report_dir")
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    report = read_json(project_path(args.analysis_json or Path(config["output_dir"]) / "analysis.json"))
    for path in plot_position_report(report, project_path(args.output_dir or config["report_dir"])):
        print(f"그림: {path}")


if __name__ == "__main__":
    main()
