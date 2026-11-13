"""STEP 5-3 무결성 검사 (CPU, 서버에서 연습 실행·본 실행 직후). 실패하면 종료 코드 1 (GPU 스크립트가 멈춤).

python -m experiments.step5_causal.check --experiment A --stage pilot
python -m experiments.step5_causal.check --estimate      # 두 연습 실행의 실측 시간으로 본 실행 시간 추정 (8시간 초과면 실패)
"""
import argparse

from src.common.config import read_yaml
from src.common.data import write_json
from src.common.paths import project_path, project_relative_path
from src.core.causal_check import check_a, check_b, combined_estimate, finish, load_run
from src.core.causal_freeze import experiment_config, verify_causal_freeze
from src.core.causal_plan import load_plan


def main():
    parser = argparse.ArgumentParser(description="STEP 5-3 무결성 검사 (평가 지표 없음)")
    parser.add_argument("--experiment", choices=("A", "B"))
    parser.add_argument("--stage", choices=("pilot", "main"))
    parser.add_argument("--estimate", action="store_true")
    args = parser.parse_args()
    _, config_a = experiment_config("A")
    plan = load_plan(read_yaml(project_path(config_a["split_config"])))
    if args.estimate:
        _, config_b = experiment_config("B")
        pilot_a, pilot_b = load_run(project_path(config_a["pilot_dir"])), load_run(project_path(config_b["pilot_dir"]))
        limit = config_a["stop_conditions"]["max_estimated_hours"]
        report = combined_estimate(plan, pilot_a, pilot_b, pilot_a["meta"]["n_layers"], limit)
        path = project_path(config_a["stop_conditions"]["estimate_file"])
        write_json(report, path, allow_nan=False)
        print(f"본 실행 예상: A {report['hours_main']['A']:.2f}h + B {report['hours_main']['B']:.2f}h "
              f"(×{report['overhead_factor']}) = {report['total_hours_with_overhead']:.2f}h / 한도 {limit}h "
              f"→ {report['status']} ({project_relative_path(path)})")
        raise SystemExit(0 if report["status"] == "passed" else 1)
    if not (args.experiment and args.stage):
        parser.error("--experiment와 --stage를 함께 주거나 --estimate를 주세요")
    _, config = experiment_config(args.experiment)
    manifest = verify_causal_freeze(args.experiment, config)
    output_dir = project_path(config["pilot_dir" if args.stage == "pilot" else "main_dir"])
    run = load_run(output_dir)
    verification = manifest["verification"]
    if args.experiment == "A":
        checks = check_a(args.stage, plan, run, verification, run["meta"].get("n_layers", 36))
    else:
        checks = check_b(args.stage, plan, run, verification)
    checks["frozen_version_matches"] = {"passed": run["meta"].get("frozen", {}).get("version") == manifest["version"],
                                        "detail": run["meta"].get("frozen")}
    gpus = (run["meta"].get("environment") or {}).get("gpus")
    checks["environment_recorded"] = {"passed": bool(run["meta"].get("environment")), "detail": gpus}
    report = finish(checks, f"{args.experiment} {args.stage}", manifest["version"])
    path = output_dir / f"{args.stage}_check.json"
    write_json(report, path, allow_nan=False)
    print(f"{args.experiment} {args.stage} 검사: {report['status']} ({project_relative_path(path)})")
    for name, check in checks.items():
        detail = check["detail"]
        if isinstance(detail, dict):
            detail = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in detail.items()}
        print(f"  {'OK  ' if check['passed'] else 'FAIL'} {name}: {detail}")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
