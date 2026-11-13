import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import write_csv, write_json
from src.common.paths import PROJECT_ROOT, project_path
from src.common.threads import configure_cpu_threads
from src.core.position_check import check_run


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step5_position.yaml"


def run_check(kind, description):
    configure_cpu_threads()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    run_dir = project_path(config["pilot_dir" if kind == "pilot" else "run_dir"])
    report, rows = check_run(run_dir, config, kind)
    write_json(report, run_dir / f"{kind}_check.json", allow_nan=False)
    if rows:
        write_csv(rows, run_dir / "prefix_identity.csv", list(dict.fromkeys(k for row in rows for k in row)))
    print(f"저장: {run_dir / f'{kind}_check.json'} (고정 버전 {report['frozen_version']})")
    print(f"상태 {report['status']} | 레코드 {report['records']} | 장수 {report['n_images']} | 역할 {report['roles']}")
    for name, result in report["checks"].items():
        detail = result.get("detail")
        print(f"  [{'OK' if result['passed'] else 'FAIL'}] {name}" + (f": {detail}" if detail is not None else ""))
    timing = {k: round(v, 2) for k, v in report["median_seconds_per_case"].items()}
    print(f"최대 GPU 메모리 {report['peak_gpu_memory_gb']:.1f} GB | 케이스당 시간(장수별 중앙값) {timing}"
          + (f" | 본 실행 추정 {report['estimated_main_run_hours']:.1f}시간" if kind == "pilot" else ""))
    print(report["note"])
    if report["status"] != "passed":
        raise SystemExit(1)


def main():
    run_check("pilot", "STEP 5-2b GPU 파일럿 무결성 검사 (평가 지표는 계산하지 않음, CPU)")


if __name__ == "__main__":
    main()
