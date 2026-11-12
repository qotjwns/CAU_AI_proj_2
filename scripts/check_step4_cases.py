import argparse
from pathlib import Path

from src.common.config import load_config, read_yaml
from src.common.data import write_json
from src.common.paths import PROJECT_ROOT, project_path
from src.data_manipulate.check_independent import check_cases, default_paths


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "step4_independent.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 4 케이스 Annotation Consistency Check (생성 규칙과의 일치 검사, CPU)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    data_config = load_config(project_path(config["source_config"]))
    cases_path, groups_path, report_path = default_paths(config)
    report = check_cases(config, data_config, cases_path, groups_path)
    write_json(report, report_path, allow_nan=False)
    print(f"저장: {report_path}")
    print(f"상태 {report['status']} | 실패 {report['failure_count']}건 | {report['counts']}")
    for failure in report["failures"][:10]:
        print(f"  {failure['case_id']}: {failure['message']}")
    print("주의: 생성과 같은 COCO-REM 주석으로 검사하므로 독립적인 정답 검증이 아닙니다.")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
