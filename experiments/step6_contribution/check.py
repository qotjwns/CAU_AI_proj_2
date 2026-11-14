"""STEP 6-2 실행 무결성 검사와 시간 추정 (CPU, 서버에서 각 단계 직후). 실패하면 종료 코드 1 (GPU 스크립트가 멈춤).

python -m experiments.step6_contribution.check --stage basis|pilot|main
python -m experiments.step6_contribution.check --estimate       # 연습 실행 실측으로 본 실행 시간 추정 (8시간 한도)

평가 지표(가설 대비)는 계산하지 않는다. 기준값은 사전 등록 verification (STEP 5-3과 같은 답·margin 기준).
- 재구성: 입력별로 항등식·attention·출력 allclose가 L21–24 모두 통과하면 유효. 유효 비율 ≥ 0.99 (연습 실행은 전부)
- 무개입 재현: batch 0번 행(빠른 경로 무개입) 대 기준 실행 (5-2b·5-2 본 추출). 답 일치 ≥ 0.99(동점 근처 제외),
  |fp32 변형 margin 차이| ≤ 0.25인 비율 ≥ 0.99
- 연습 실행: 크기 0 행 = 같은 batch 무개입 행, 빠른 경로 = 참조 경로(전체 prefill model.generate + 같은 벡터) 모든 행
- 벡터: Parallel + Orthogonal = c_N (상대 오차 ≤ 1e-5), Random·크기 맞춤 행의 노름 = ‖c_N‖ (상대 오차 ≤ 1e-5)
"""
import argparse
from collections import defaultdict
import math
import statistics

import torch

from src.common.config import read_yaml
from src.common.data import read_json, read_jsonl, write_json
from src.common.paths import project_path, project_relative_path
from src.core.causal_check import agreement, baseline_row, equivalence, finite, finish, spec_matches
from src.core.causal_plan import load_plan
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file
from experiments.step6_contribution import directions as D
from experiments.step6_contribution.conditions import BASES, LAYERS, ROW_KEYS, condition_rows, stage_work
from experiments.step6_contribution.freeze import load_config, verify_freeze
from experiments.step6_contribution.plan import load_step6_plan


OVERHEAD = 1.10


def rows_of(path):
    return list(read_jsonl(path)) if path.exists() else []


def load_run(output_dir):
    extraction = {}
    for row in rows_of(output_dir / "extraction.jsonl"):
        extraction[row.get("item_id") or row["input_id"]] = row          # 이어서 한 실행은 마지막 기록
    return {"meta": read_json(output_dir / "run.json") if (output_dir / "run.json").exists() else {},
            "batches": {row["batch"]: row for row in rows_of(output_dir / "conditions.jsonl")},
            "reference": {row["batch"]: row for row in rows_of(output_dir / "reference.jsonl")},
            "extraction": extraction}


def record(checks, name, ok, detail=None):
    checks[name] = {"passed": bool(ok), "detail": detail}


def common_checks(checks, run, manifest):
    meta = run["meta"]
    record(checks, "run_completed", meta.get("status") == "completed", meta.get("status"))
    record(checks, "frozen_version_matches", meta.get("frozen", {}).get("version") == manifest["version"],
           meta.get("frozen"))
    record(checks, "environment_recorded", bool(meta.get("environment")),
           (meta.get("environment") or {}).get("nvidia_smi"))


# ---------------------------------------------------------------- basis

def check_basis(config, plan6, manifest, expected_layers=LAYERS):
    output_dir = project_path(config["dirs"]["basis"])
    run = load_run(output_dir)
    verification = manifest["verification"]
    checks = {}
    common_checks(checks, run, manifest)
    pool = {row["input_id"]: row for row in plan6["pool"]}
    written = run["extraction"]
    record(checks, "all_pool_inputs_extracted", set(written) == set(pool),
           {"planned": len(pool), "written": len(written)})
    valid = [i for i, row in written.items() if row["valid"]]
    share = len(valid) / len(pool) if pool else 0.0
    record(checks, "reconstruction_valid_share", share >= verification["extraction_valid_share_min"],
           {"valid": len(valid), "planned": len(pool), "share": share,
            "worst_output_excess": max((layer["output_worst_excess"] for row in written.values()
                                        for layer in row["layers"].values()), default=None)})
    summary_path = output_dir / "basis_summary.json"
    basis_path = output_dir / "basis.pt"
    if not (summary_path.exists() and basis_path.exists()):
        record(checks, "basis_files", False, "basis.pt or basis_summary.json is missing")
        return finish(checks, "basis", manifest["version"]), None
    summary = read_json(summary_path)
    digest = sha256_file(basis_path)
    record(checks, "basis_hash_matches_summary", digest == summary["basis_sha256"], digest)
    payload = torch.load(basis_path)
    expected = {"p_basis": 0, "p_mixed": 0, "n_mixed": 0}
    n_images = defaultdict(set)
    for input_id in valid:
        row = pool[input_id]
        labels = row["labels"]
        length = written[input_id]["n_question"]
        expected["p_basis"] += labels.count("P") * length
        if row["mixed"]:
            expected["p_mixed"] += labels.count("P") * length
            expected["n_mixed"] += labels.count("N") * length
            for image_id, label in zip(row["case"]["image_ids"], labels):
                if label == "N":
                    n_images[row["class"]].add(image_id)
    record(checks, "accumulated_counts_match", payload["counts"]["vectors"] == expected,
           {"accumulated": payload["counts"]["vectors"], "expected": expected})
    criteria = manifest["plan_sufficiency"]["criteria"]
    slots = payload["counts"]["slots"]
    ready = (len(n_images) == 20 and min(len(v) for v in n_images.values()) >= criteria["min_n_images_per_class_mixed"]
             and slots.get("n_mixed", 0) >= criteria["min_n_slots_mixed"]
             and slots.get("p_mixed", 0) >= criteria["min_p_slots_mixed"]
             and slots.get("p_basis", 0) >= criteria["min_p_slots_basis"])
    record(checks, "mean_difference_ready", ready,
           {"slots": slots, "min_n_images_per_class": min((len(v) for v in n_images.values()), default=0)})
    shapes, orthonormal, finite_ok = {}, True, True
    layers = payload["layers"]
    for name in BASES:
        for layer in layers:
            basis = payload["bases"][name][layer]
            shapes[f"{name}/L{layer + 1}"] = list(basis.shape)
            try:
                D.check_orthonormal(basis, tolerance=1e-5)
            except ValueError:
                orthonormal = False
            finite_ok &= bool(torch.isfinite(basis).all())
    for layer in layers:
        finite_ok &= bool(torch.isfinite(payload["mu_n"][layer]).all() and torch.isfinite(payload["mu_p"][layer]).all())
    record(checks, "bases_orthonormal_and_finite", orthonormal and finite_ok, shapes)
    k_ok = all(1 <= payload["k"]["v90"][l] <= payload["k"]["v95"][l] <= payload["bases"]["v95"][l].shape[0]
               for l in layers)
    record(checks, "basis_dimensions", k_ok, {name: payload["k"][name] for name in BASES})
    difference_ok = all(float((payload["mu_n"][l] - payload["mu_p"][l]).norm()) > 0 for l in layers)
    record(checks, "layers_preregistered", tuple(layers) == tuple(expected_layers), list(layers))
    record(checks, "mean_difference_nonzero", difference_ok)
    report = finish(checks, "basis", manifest["version"])
    report["basis_sha256"] = digest
    return report, summary


# ---------------------------------------------------------------- pilot·main

def check_items(config, plan6, plan53, manifest, stage):
    output_dir = project_path(config["dirs"][stage])
    run = load_run(output_dir)
    verification = manifest["verification"]
    work = stage_work(plan6, plan53, stage)
    inputs = plan53["inputs"]
    checks = {}
    common_checks(checks, run, manifest)
    basis_check = read_json(project_path(config["dirs"]["basis"]) / "basis_check.json")
    record(checks, "basis_hash_matches", run["meta"].get("basis_sha256") == basis_check["basis_sha256"],
           run["meta"].get("basis_sha256"))
    planned = {b["batch"]: b for b in work["batches"]}
    bad = [name for name, batch in planned.items()
           if name not in run["batches"] or not spec_matches(batch["rows"], run["batches"][name]["rows"], ROW_KEYS)]
    record(checks, "batches_complete", not bad and len(run["batches"]) == len(planned),
           {"planned": len(planned), "written": len(run["batches"]), "missing_or_wrong": bad[:5]})
    rows = [row for batch in run["batches"].values() for row in batch["rows"]]
    record(checks, "finite_values", finite(rows))
    extraction = run["extraction"]
    valid = [i for i in work["items"] if i in extraction and extraction[i]["valid"]]
    share = len(valid) / len(work["items"]) if work["items"] else 0.0
    minimum = 1.0 if stage == "pilot" else verification["extraction_valid_share_min"]
    record(checks, "reconstruction_valid_share", share >= minimum,
           {"valid": len(valid), "planned": len(work["items"]), "share": share, "minimum": minimum,
            "missing": len(set(work["items"]) - set(extraction))})
    pairs, slot_gaps = [], []
    by_item = defaultdict(list)
    for name, batch in run["batches"].items():
        by_item[batch["item_id"]].append(batch)
    for item_id, batches in by_item.items():
        primary = [b for b in batches if b["primary_slot"]]
        if primary:
            pairs.append((primary[0]["rows"][0], baseline_row(inputs[primary[0]["input_id"]])))
        margins = [b["rows"][0]["margin"] for b in batches]
        slot_gaps.append(max(margins) - min(margins))
    summary = agreement(pairs, verification)
    record(checks, "reproduction_no_intervention", summary["passed"], summary)
    checks["none_rows_across_slot_batches_reported"] = {
        "passed": True, "detail": {"items": len(slot_gaps), "max_margin_gap": max(slot_gaps, default=0.0)}}
    tolerance = verification["vector_identity_tolerance"]
    worst = {"sum_identity_relative_error_max": 0.0, "norm_relative_error_max": 0.0}
    degenerate = 0
    for batch in run["batches"].values():
        for key in worst:
            worst[key] = max(worst[key], batch["vector_checks"][key])
        degenerate += batch["vector_checks"]["degenerate_match_rows"]
    record(checks, "vector_identities", all(v <= tolerance for v in worst.values()),
           {**worst, "tolerance": tolerance, "degenerate_match_rows_reported": degenerate})
    if stage == "pilot":
        zero = [(row, batch["rows"][0]) for batch in run["batches"].values() for row in batch["rows"]
                if row["condition"] == "full" and row["lam"] == 0.0]
        summary = agreement(zero, verification)
        record(checks, "zero_intervention_identity", summary["passed"] and len(zero) == len(planned), summary)
        equivalence(checks, work["reference_batches"], run, verification, ROW_KEYS)
        changed = [abs(row["margin"] - batch["rows"][0]["margin"]) for batch in run["batches"].values()
                   for row in batch["rows"][1:] if row["lam"] != 0.0]
        record(checks, "interventions_change_output", bool(changed) and statistics.median(changed) > 0,
               {"median_abs_margin_change": statistics.median(changed) if changed else None})
    return finish(checks, stage, manifest["version"])


# ---------------------------------------------------------------- 시간 추정

def mean_by(rows, key, value):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(value(row))
    return {k: statistics.fmean(v) for k, v in groups.items()}


def fallback(table, n):
    return table.get(n, statistics.fmean(table.values()))


def estimate(config, plan6, plan53, limit):
    pilot = load_run(project_path(config["dirs"]["pilot"]))
    require(pilot["meta"].get("status") == "completed", "The pilot must be completed before the time estimate")
    batches = list(pilot["batches"].values())
    first = [b for b in batches if b["prefix_seconds"] > 0]
    setup = mean_by(first, "n_images", lambda b: b["prefix_seconds"] + b["extract_seconds"])
    per_row = mean_by(batches, "n_images", lambda b: b["seconds"] / len(b["rows"]))
    work = stage_work(plan6, plan53, "main")
    rows = len(condition_rows(pilot=False))
    seconds = sum(fallback(setup, item["n_images"]) for item in work["items"].values())
    seconds += sum(fallback(per_row, batch["n_images"]) * rows for batch in work["batches"])
    hours = seconds / 3600
    total = hours * OVERHEAD
    return {"status": "passed" if total <= limit else "failed", "hours_main": hours, "overhead_factor": OVERHEAD,
            "total_hours_with_overhead": total, "limit_hours": limit,
            "main_items": len(work["items"]), "main_batches": len(work["batches"]), "rows_per_batch": rows,
            "rule": "measured pilot seconds (means by image count: prepare + extraction + prefix per item, "
                    "suffix seconds per row) x planned main work; stop if above the limit"}


def main():
    parser = argparse.ArgumentParser(description="STEP 6-2 무결성 검사 (평가 지표 없음)")
    parser.add_argument("--stage", choices=("basis", "pilot", "main"))
    parser.add_argument("--estimate", action="store_true")
    args = parser.parse_args()
    config = load_config()
    manifest = verify_freeze(config)
    plan6 = load_step6_plan(config["plan_dir"])
    plan53 = load_plan(read_yaml(project_path(config["split_config"])))
    if args.estimate:
        limit = config["stop_conditions"]["max_estimated_hours"]
        report = estimate(config, plan6, plan53, limit)
        path = project_path(config["stop_conditions"]["estimate_file"])
        write_json(report, path, allow_nan=False)
        print(f"본 실행 예상: {report['hours_main']:.2f}h × {OVERHEAD} = {report['total_hours_with_overhead']:.2f}h "
              f"/ 한도 {limit}h → {report['status']} ({project_relative_path(path)})")
        raise SystemExit(0 if report["status"] == "passed" else 1)
    if not args.stage:
        parser.error("--stage 또는 --estimate를 주세요")
    if args.stage == "basis":
        report, summary = check_basis(config, plan6, manifest)
    else:
        report = check_items(config, plan6, plan53, manifest, args.stage)
    path = project_path(config["dirs"][args.stage]) / f"{args.stage}_check.json"
    write_json(report, path, allow_nan=False)
    print(f"{args.stage} 검사: {report['status']} ({project_relative_path(path)})")
    for name, check in report["checks"].items():
        detail = check["detail"]
        if isinstance(detail, dict):
            detail = {k: (round(v, 6) if isinstance(v, float) and math.isfinite(v) else v) for k, v in detail.items()}
        print(f"  {'OK  ' if check['passed'] else 'FAIL'} {name}: {str(detail)[:300]}")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
