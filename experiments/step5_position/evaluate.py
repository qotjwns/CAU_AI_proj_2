import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import read_json, write_csv, write_json
from src.common.paths import PROJECT_ROOT, project_path, project_relative_path
from src.common.threads import configure_cpu_threads
from src.core.accumulation_freeze import verify_accumulation_freeze
from src.core.position_eval import load_inputs, run_position_eval, structure_summary
from src.core.probe_freeze import sha256_file, verify_frozen


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step5_position_eval.yaml"


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:+.{digits}f}"


def columns(rows):
    return list(dict.fromkeys(key for row in rows for key in row))


def write_tables(report, rows, output_dir):
    write_json(report, output_dir / "analysis.json", allow_nan=False)
    secondary, exploratory = report["secondary"], report["exploratory"]
    tables = {
        "primary.csv": report["primary"],
        "secondary_contrasts.csv": secondary["contrasts"],
        "secondary_step4_comparison.csv": secondary["step4_comparison"]["rows"],
        "secondary_sign_consistency.csv": secondary["sign_consistency"]["rows"],
        "exploratory_grid.csv": exploratory["grid"],
        "exploratory_ab_slopes.csv": exploratory["ab_slopes"],
        "exploratory_position_influence.csv": exploratory["position_influence"],
        "exploratory_z_n_by_a.csv": exploratory.get("z_n_by_a", []),
        "exploratory_z_n_margin_relation.csv": exploratory.get("z_n_margin_relation", []),
        "exploratory_h_q_three_images.csv": exploratory.get("h_q_three_images", []),
        "exploratory_position_accuracy.csv": exploratory.get("position_accuracy", []),
    }
    for name, table in tables.items():
        if table:
            write_csv(table, output_dir / name, columns(table))
    write_csv(rows, output_dir / "inputs.csv", columns(rows))


def record_rerun(output_dir, reason):
    existing = output_dir / "analysis.json"
    if not existing.exists():
        return
    if not reason:
        raise SystemExit(f"평가 결과가 이미 있습니다: {existing}\n"
                         "확증 평가는 한 번만 봅니다. 꼭 다시 돌려야 하면 --rerun-reason '이유'를 주세요 (기록됨).")
    log_path = output_dir / "rerun_log.json"
    log = read_json(log_path) if log_path.exists() else []
    log.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reason": reason,
                "previous_analysis_sha256": sha256_file(existing)})
    write_json(log, log_path)


def main():
    configure_cpu_threads()
    parser = argparse.ArgumentParser(description="STEP 5-2b 평가: 사전 등록 v1 판정 그대로 적용 (CPU, confirmatory)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true", help="고정 해시·무결성 검사·케이스·설계 구조만 확인 (지표 계산 안 함)")
    parser.add_argument("--rerun-reason", help="평가 결과가 이미 있을 때 다시 실행하는 이유 (rerun_log.json에 기록)")
    parser.add_argument("--skip-features", action="store_true", help="특징 기반 탐색 분석 생략")
    args = parser.parse_args()
    eval_config = read_yaml(project_path(args.config))
    output_dir = project_path(eval_config["output_dir"])

    inputs = load_inputs(eval_config)
    if args.check_only:
        summary = structure_summary(inputs)
        write_json(summary, output_dir / "preflight.json", allow_nan=False)
        print(f"구조 확인 통과 (평가 지표 계산 안 함): {output_dir / 'preflight.json'}")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return

    record_rerun(output_dir, args.rerun_reason)
    report, rows, _ = run_position_eval(eval_config, inputs, with_features=not args.skip_features)
    report["eval_config"] = {"path": project_relative_path(args.config), "sha256": sha256_file(project_path(args.config))}
    report["output_dir"] = project_relative_path(output_dir)
    report["evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_tables(report, rows, output_dir)

    config = inputs["config"]
    step52_manifest = read_json(project_path(config["accumulation"]["frozen_dir"]) / "manifest.json")
    changed = (verify_accumulation_freeze(inputs["manifest"]) + verify_accumulation_freeze(step52_manifest)
               + verify_frozen(inputs["zi_dir"], inputs["zi_manifest"]))
    write_json({"frozen_dir": report["preregistration"]["frozen_dir"], "verified_before": True,
                "verified_after": not changed, "changed_files": changed}, output_dir / "frozen_verification.json")
    if changed:
        write_json({"status": "invalid", "reason": f"Frozen files changed during evaluation: {changed}"},
                   output_dir / "INVALID.json")
        raise SystemExit(f"무효: 평가 중 고정 파일이 바뀌었습니다 {changed}. 이 결과를 사용하지 마세요.")

    print(f"저장: {output_dir} | 상태 {report['status']} | 사전 등록 {report['preregistration']['version']}")
    for row in report["primary"]:
        print(f"  {row['hypothesis']} ({row['contrast']}) {fmt(row['estimate'])} "
              f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] 97.5% | 클래스 [{fmt(row['class_ci_low'])}, "
              f"{fmt(row['class_ci_high'])}] (최소 {row['minimum_effect']}) → {row['label']}")
    print("부가·탐색 결과는 analysis.json과 CSV. 그림: experiments.step5_position.visualize")


if __name__ == "__main__":
    main()
