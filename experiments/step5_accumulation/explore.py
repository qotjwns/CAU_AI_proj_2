import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import write_csv, write_json
from src.common.paths import PROJECT_ROOT, project_path
from src.common.threads import configure_cpu_threads
from src.core.accumulation_explore import run_exploration


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step5_accumulation_explore.yaml"


def main():
    configure_cpu_threads()
    parser = argparse.ArgumentParser(description="STEP 5-2 탐색 분석 (CPU, exploratory, 판정과 무관)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    output_dir = project_path(config["output_dir"])
    tables, summary = run_exploration(config)
    for name, rows in tables.items():
        write_csv(rows, output_dir / f"{name}.csv", list(dict.fromkeys(key for row in rows for key in row)))
    write_json(summary, output_dir / "summary.json", allow_nan=False)
    print(f"저장: {output_dir} ({len(tables)}개 표 + summary.json) | 상태: exploratory, 판정과 무관")


if __name__ == "__main__":
    main()
