"""STEP 5-3 실험 A·B 분할·계획 파일 생성 (CPU, 5-3 출력 전, 덮어쓰기 금지).

기준 답은 5-2b 본 추출(one-N)과 5-2 본 추출(All-P)에서만 읽는다. 결과: data/step5_causal/
"""
import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.paths import PROJECT_ROOT, project_path
from src.common.data import write_json
from src.core.causal_plan import build_plan, load_plan, sample_report


DEFAULT_SPLIT = PROJECT_ROOT / "configs" / "data" / "step5_causal_split.yaml"
DEFAULT_PATCHING = PROJECT_ROOT / "configs" / "experiments" / "step5_patching.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 5-3 분할·A 단위·B 항목 계획 생성 (CPU)")
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--patching-config", type=Path, default=DEFAULT_PATCHING)
    parser.add_argument("--report", action="store_true",
                        help="이미 만든 계획으로 freeze 전 표본 수·예상 CI 반폭 보고만 쓴다 (prefreeze_report.json)")
    args = parser.parse_args()
    split_config = read_yaml(project_path(args.split_config))
    if args.report:
        report = sample_report(load_plan(split_config))
        path = project_path(split_config["output_dir"]) / split_config["report_file"]
        write_json(report, path, allow_nan=False)
        print(f"저장: {path}")
        return
    report = build_plan(split_config, read_yaml(project_path(args.patching_config)))
    print(f"계획 검사: {report['status']} | {report['counts']}")
    for name, check in report["checks"].items():
        print(f"  {'OK ' if check['passed'] else 'FAIL'} {name}: {check['detail']}")
    if report["status"] != "passed":
        raise SystemExit(f"계획 검사 실패: {report['failed']}")


if __name__ == "__main__":
    main()
