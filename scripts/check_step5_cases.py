import argparse
from pathlib import Path

from src.common.config import load_config, read_yaml
from src.common.data import write_json
from src.common.paths import PROJECT_ROOT, project_path
from src.data_manipulate.build_accumulation import paths
from src.data_manipulate.check_accumulation import check_cases


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "step5_accumulation.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 5-2 케이스 일관성 검사 (재생성 일치 + 설계 규칙, CPU)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    data_config = load_config(project_path(config["source_config"]))
    report = check_cases(config, data_config)
    report_path = paths(config)["check"]
    write_json(report, report_path, allow_nan=False)
    print(f"저장: {report_path}")
    print(f"상태 {report['status']} | 실패 {report['failure_count']}건 | {report['counts']}")
    for failure in report["failures"][:10]:
        print(f"  {failure['case_id']}: {failure['message']}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
