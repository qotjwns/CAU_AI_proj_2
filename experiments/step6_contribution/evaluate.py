"""STEP 6-2 평가 (CPU, confirmatory, 한 번만). 결과를 본 뒤 대비와 판정 기준을 바꾸지 않는다.

python -m experiments.step6_contribution.evaluate --check-only   # 고정·검사·구조만 확인 (지표 계산 안 함)
python -m experiments.step6_contribution.evaluate                # 본 평가 (결과가 있으면 거부)

판정에 쓰는 값은 고정 manifest의 사전 등록에서 읽는다 (analysis.py). 평가 전후 고정 파일 해시를 확인하고
(서버 파일 + 로컬 파일: 겹침 검사에 쓴 케이스, 나란히 비교할 5-3B 본 실행 결과), 바뀌었으면 INVALID.json을 남긴다.
→ results/step6/qwen2_5_vl_3b/contribution/evaluation/analysis.json, primary.csv, secondary.csv, side_effects.csv
"""
import argparse
from datetime import datetime, timezone

from src.common.config import read_yaml
from src.common.data import read_json, write_csv, write_json
from src.common.paths import project_path, project_relative_path
from src.core.causal_check import load_run as load_run_53
from src.core.causal_eval import b_index
from src.core.causal_plan import load_plan
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file
from experiments.step6_contribution.analysis import analyze
from experiments.step6_contribution.check import load_run
from experiments.step6_contribution.freeze import changed_files, load_config, verify_freeze
from experiments.step6_contribution.plan import load_step6_plan


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:+.{digits}f}"


def flatten(row, prefix=""):
    out = {}
    for key, value in row.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        elif isinstance(value, list):
            out[f"{prefix}{key}"] = "|".join(str(v) for v in value)
        else:
            out[f"{prefix}{key}"] = value
    return out


def write_table(rows, path):
    rows = [flatten(row) for row in rows]
    write_csv(rows, path, list(dict.fromkeys(k for row in rows for k in row)))


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


def load_inputs(config):
    manifest = verify_freeze(config, with_local=True)
    plan6 = load_step6_plan(config["plan_dir"])
    plan53 = load_plan(read_yaml(project_path(config["split_config"])))
    main_dir = project_path(config["dirs"]["main"])
    check = read_json(main_dir / "main_check.json")
    require(check["status"] == "passed", "The main-run integrity check has not passed")
    run = load_run(main_dir)
    require(run["meta"]["frozen"]["version"] == manifest["version"], "The main run used a different frozen version")
    comparison = config["comparison_5_3b"]
    run53 = load_run_53(project_path(comparison["main_dir"]))
    table53, _ = b_index(plan53, run53["batches"])
    return manifest, plan6, plan53, run, {"table": table53, "band": comparison["band"]}


def main():
    parser = argparse.ArgumentParser(description="STEP 6-2 평가 (확증, 한 번만)")
    parser.add_argument("--check-only", action="store_true", help="고정·검사·구조만 확인하고 지표는 계산하지 않음")
    parser.add_argument("--rerun-reason", help="평가 결과가 이미 있을 때 다시 실행하는 이유 (rerun_log.json에 기록)")
    args = parser.parse_args()
    config = load_config()
    manifest, plan6, plan53, run, comparison = load_inputs(config)
    if args.check_only:
        print(f"확인: 고정 {manifest['version']}, 본 실행 batch {len(run['batches'])}, 추출 {len(run['extraction'])}, "
              f"5-3B 항목 {len(comparison['table'])} (지표 계산 안 함)")
        return
    output_dir = project_path(config["dirs"]["evaluation"])
    output_dir.mkdir(parents=True, exist_ok=True)
    record_rerun(output_dir, args.rerun_reason)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = analyze(plan53, plan6, run, manifest["preregistration"], comparison)
    changed = changed_files(manifest["files_sha256"]) + changed_files(manifest["local_sha256"])
    if changed:
        write_json({"reason": "frozen files changed during evaluation", "changed": changed}, output_dir / "INVALID.json")
        raise SystemExit(f"고정 파일이 평가 중 바뀌었습니다: {changed}")
    basis_summary = read_json(project_path(config["dirs"]["basis"]) / "basis_summary.json")
    report = {"evaluated_at": started, "frozen_version": manifest["version"],
              "frozen_manifest_sha256": sha256_file(project_path(config["frozen_dir"]) / "manifest.json"),
              "main_run": {k: run["meta"].get(k) for k in ("started_at", "finished_at", "basis_sha256", "basis_k")},
              "basis_summary": basis_summary, **result}
    write_json(report, output_dir / "analysis.json", allow_nan=False)
    write_table(result["primary"], output_dir / "primary.csv")
    write_table(result["secondary"], output_dir / "secondary.csv")
    write_table(result["all_positive_side_effects"]["rows"], output_dir / "side_effects.csv")
    if result["comparison_5_3b"]:
        write_table(result["comparison_5_3b"]["rows"], output_dir / "comparison_5_3b.csv")
    write_table(result["descriptive"]["vector_summary"], output_dir / "vector_summary.csv")
    print(f"평가: {project_relative_path(output_dir)} (분석 {result['counts']})")
    for row in result["primary"]:
        c, n, r = row["contrast"], row["n_arm"], row["reported_arm"]
        print(f"{row['hypothesis']}: {row['label']} {row['unmet_conditions'] or ''}\n"
              f"  대비 {fmt(c['estimate'])} [{fmt(c['ci_low'])}, {fmt(c['ci_high'])}] "
              f"(클래스 [{fmt(c['class_ci_low'])}, {fmt(c['class_ci_high'])}]), "
              f"N arm {row['n_arm_name']} {fmt(n['estimate'])} [{fmt(n['ci_low'])}, {fmt(n['ci_high'])}], "
              f"보고 {r['arm']} {fmt(r['estimate'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]")
    for warning in result["warnings"]:
        print(f"경고: {warning}")


if __name__ == "__main__":
    main()
