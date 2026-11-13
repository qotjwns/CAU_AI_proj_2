"""STEP 5-3 평가 (CPU, confirmatory, 한 번만).

python -m experiments.step5_causal.evaluate --experiment A --check-only   # 고정·검사·구조만 확인 (지표 계산 안 함)
python -m experiments.step5_causal.evaluate --experiment A                # 본 평가 (결과가 있으면 거부)

판정에 쓰는 값은 고정 manifest의 사전 등록에서 읽는다. 선택 절반의 층·구간 선택을 selection.json에 먼저 기록한 뒤
확증 절반을 계산한다. 평가 후 고정 파일 해시를 다시 확인하고, 바뀌었으면 INVALID.json을 남기고 실패한다.
고정본 v3: 본 실행(v2)과 평가 고정본이 다르면 평가 쪽 파일만 바뀌었는지 확인한다(run_compatibility). v2 평가가 멈추기
전에 남긴 선택 기록이 있으면(revision.prior_partial_evaluation) 새 선택이 그것과 같아야 하며, 다르면 확증 절반을
계산하기 전에 INVALID.json을 남기고 멈춘다.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json

from src.common.config import read_yaml
from src.common.data import read_json, write_csv, write_json
from src.common.paths import project_path, project_relative_path
from src.common.threads import configure_cpu_threads
from src.core.causal_check import load_run
from src.core.causal_eval import analyze_a, analyze_b
from src.core.causal_freeze import changed_files, experiment_config, run_compatibility, verify_causal_freeze
from src.core.causal_plan import load_plan
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:+.{digits}f}"


def columns(rows):
    return list(dict.fromkeys(key for row in rows for key in row))


def flatten(row, prefix=""):
    out = {}
    for key, value in row.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


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


def load_inputs(experiment):
    _, config = experiment_config(experiment)
    manifest = verify_causal_freeze(experiment, config, with_baseline=True)
    plan = load_plan(read_yaml(project_path(config["split_config"])))
    run_dir = project_path(config["main_dir"])
    check = read_json(run_dir / "main_check.json")
    require(check["status"] == "passed" and not check["failed"], "The main integrity check did not pass")
    run = load_run(run_dir)
    require(run["meta"]["status"] == "completed" and run["meta"]["stage"] == "main", "The main run is incomplete")
    require(check["frozen_version"] == run["meta"]["frozen"]["version"], "The main check and run used different versions")
    compatibility = run_compatibility(experiment, manifest, run["meta"]["frozen"])
    return config, manifest, plan, run, compatibility


def prior_selection(manifest):
    """v2 평가가 멈추기 전에 남긴 선택 기록 (revision.prior_partial_evaluation). 파일 해시가 기록과 같아야 한다."""
    prior = (manifest.get("revision") or {}).get("prior_partial_evaluation")
    if not prior:
        return None
    path = project_path(prior["selection_file"])
    require(sha256_file(path) == prior["selection_file_sha256"], f"The prior selection record changed: {path}")
    return {"file": prior["selection_file"], "sha256": read_json(path)["sha256"]}


def main():
    configure_cpu_threads()
    parser = argparse.ArgumentParser(description="STEP 5-3 평가: 사전 등록 그대로 적용 (CPU, confirmatory, 한 번만)")
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    parser.add_argument("--check-only", action="store_true", help="고정 해시·무결성 검사·구조만 확인 (지표 계산 안 함)")
    parser.add_argument("--rerun-reason", help="평가 결과가 이미 있을 때 다시 실행하는 이유 (rerun_log.json에 기록)")
    args = parser.parse_args()
    config, manifest, plan, run, compatibility = load_inputs(args.experiment)
    output_dir = project_path(config["evaluation_dir"])
    prior = prior_selection(manifest)
    if args.check_only:
        summary = {"note": "Structure only: no contrast is computed.", "frozen_version": manifest["version"],
                   "run_compatibility": compatibility, "prior_selection_record": prior is not None,
                   "batches": len(run["batches"]), "phase1": len(run["phase1"]), "plan_counts": plan["check"]["counts"]}
        write_json(summary, output_dir / "preflight.json", allow_nan=False)
        print(f"구조 확인 통과 (지표 계산 안 함): {summary}")
        return
    record_rerun(output_dir, args.rerun_reason)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_match = {}

    def on_selection(selection):
        text = json.dumps(selection, ensure_ascii=False, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(text.encode()).hexdigest()
        write_json({"recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "note": "Selected on the selection half before any confirmation-half value was computed.",
                    "selection": selection, "sha256": digest}, output_dir / "selection.json", allow_nan=False)
        if prior is not None:
            prior_match.update(file=prior["file"], prior_sha256=prior["sha256"], sha256=digest,
                               matches=digest == prior["sha256"])
            if not prior_match["matches"]:
                write_json({"status": "invalid", "reason": "The selection differs from the prior v2 selection record",
                            **prior_match}, output_dir / "INVALID.json")
                raise SystemExit("무효: 선택 결과가 v2 평가가 남긴 선택 기록과 다릅니다 (확증 절반 계산 전 중단)")

    preregistration = manifest["preregistration"]
    if args.experiment == "A":
        sections = analyze_a(plan, run, preregistration, run["meta"]["n_layers"], on_selection)
    else:
        sections = analyze_b(plan, run, preregistration, on_selection)
    report = {
        "status": config["status"], "experiment": args.experiment,
        "preregistration": {"frozen_dir": config["frozen_dir"], "version": manifest["version"],
                            "manifest_sha256": sha256_file(project_path(config["frozen_dir"]) / "manifest.json"),
                            "created_at": manifest["created_at"], "hypotheses": preregistration["hypotheses"],
                            "decision": preregistration["decision"], "statistics": preregistration["statistics"]},
        "run": {"dir": config["main_dir"], "environment_gpus": run["meta"]["environment"].get("gpus"),
                "versions": run["meta"].get("versions"), "frozen": run["meta"]["frozen"]},
        "run_compatibility": compatibility, "prior_selection_check": prior_match or None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **sections,
    }
    write_json(report, output_dir / "analysis.json", allow_nan=False)
    tables = {"primary.csv": [flatten(r) for r in report["primary"]], "secondary.csv": report["secondary"],
              "all_positive_side_effects.csv": report["all_positive_side_effects"]}
    curves = report["descriptive"].get("curves") or report["descriptive"].get("band_curves")
    tables["curves.csv"] = curves
    for name, rows in tables.items():
        if rows:
            write_csv(rows, output_dir / name, columns(rows))
    changed = changed_files(manifest["files_sha256"]) + changed_files(manifest["baseline_sha256"])
    write_json({"verified_before": True, "verified_after": not changed, "changed_files": changed},
               output_dir / "frozen_verification.json")
    if changed:
        write_json({"status": "invalid", "reason": f"Frozen files changed during evaluation: {changed}"},
                   output_dir / "INVALID.json")
        raise SystemExit(f"무효: 평가 중 고정 파일이 바뀌었습니다 {changed}")
    print(f"저장: {project_relative_path(output_dir)} | 사전 등록 {manifest['version']}")
    for row in report["primary"]:
        if args.experiment == "A":
            print(f"  {row['hypothesis']} ({row['direction']}, l*={row['layer']}) {fmt(row['estimate'])} "
                  f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] 97.5% | 클래스 [{fmt(row['class_ci_low'])}, "
                  f"{fmt(row['class_ci_high'])}] (최소 {row['minimum_effect']}) → {row['label']}")
        else:
            c, n, p = row["contrast"], row["n_arm"], row["p_arm_reported"]
            print(f"  {row['hypothesis']} ({row['population']}, 구간 {row['band']}) C {fmt(c['estimate'])} "
                  f"[{fmt(c['ci_low'])}, {fmt(c['ci_high'])}] | Δm(N) {fmt(n['estimate'])} [{fmt(n['ci_low'])}, "
                  f"{fmt(n['ci_high'])}] | Δm(P) {fmt(p['estimate'])} [{fmt(p['ci_low'])}, {fmt(p['ci_high'])}] "
                  f"→ {row['label']} {row['unmet_conditions'] or ''}")
    for warning in report.get("warnings", []):
        print(f"  경고: {warning}")


if __name__ == "__main__":
    main()
