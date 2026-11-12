import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import write_json
from src.common.paths import PROJECT_ROOT, project_path
from src.core.pilot_check import check_pilot


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step4.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 4 GPU 파일럿 무결성 검사 (평가 지표는 계산하지 않음, CPU)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=Path("results/step4/qwen2_5_vl_3b/pilot"))
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    run_dir = project_path(args.run_dir).resolve()
    report = check_pilot(run_dir, config)
    write_json(report, run_dir / "pilot_check.json", allow_nan=False)
    print(f"저장: {run_dir / 'pilot_check.json'} | 상태 {report['status']} | 레코드 {report['records']} | "
          f"모드 {report['modes']} | 구성 {report['compositions']}")
    for name, result in report["checks"].items():
        print(f"  {'OK ' if result['passed'] else 'FAIL'} {name}" + (f" — {result['detail']}" if not result["passed"] else ""))
    print("정답률·Probe 성능은 확인하지 않습니다. 실패 시 원인을 고치고, 고정 내용이 바뀌어야 하면 새 버전으로 freeze 하세요.")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
