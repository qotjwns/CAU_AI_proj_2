"""STEP 5-3 실행 무결성 검사와 시간 추정 (CPU, 서버에서 연습 실행·본 실행 직후).

평가 지표(가설 대비)는 계산하지 않는다. 기준값(사전 등록 verification, 사용자 승인):
- 무개입 재현: 파싱한 답 일치 ≥ 99%, 그리고 99% 이상의 입력에서 |fp32 변형 margin 차이| ≤ 0.25
- 빠른 경로 동등성(연습 실행): 참조 경로(기준 추출과 같은 전체 prefill)와 같은 기준
- A ℓ = 0 이식, B 크기 0 개입: 같은 batch의 무개입 행과 같은 기준 (정확히 같은 비율도 보고)
- A ℓ = 36 이식: 모든 행에서 donor의 무개입 margin과 0.01 이내, 답 일치 ≥ 99%
보고만 하는 값: 기준 bf16 margin이 한 칸(0.125) 이내인 입력을 뺀 답 일치율 (bf16 동률은 경로가 바뀌면 다르게 깨질 수 있음)
"""
from collections import defaultdict
import math
import statistics

from src.common.data import read_json, read_jsonl
from src.core.causal_conditions import a_work, b_work
from src.core.feature_validation import require


NEAR_TIE = 0.125
OVERHEAD = 1.10


def read_rows(path):
    return list(read_jsonl(path)) if path.exists() else []


def load_run(output_dir):
    return {"meta": read_json(output_dir / "run.json") if (output_dir / "run.json").exists() else {},
            "phase1": {row["input_id"]: row for row in read_rows(output_dir / "phase1.jsonl")},
            "batches": {row["batch"]: row for row in read_rows(output_dir / "conditions.jsonl")},
            "reference": {row["batch"]: row for row in read_rows(output_dir / "reference.jsonl")},
            "reproduction": {row["input_id"]: row for row in read_rows(output_dir / "reproduction.jsonl")}}


def agreement(pairs, verification, baseline_bf16=None):
    """pairs: [(값, 기준값)] (답, margin). 기준 통과 여부와 요약."""
    n = len(pairs)
    if n == 0:
        return {"passed": False, "n": 0}
    same = sum(left["answer"] == right["answer"] for left, right in pairs)
    gaps = [abs(left["margin"] - right["margin"]) for left, right in pairs]
    within = sum(gap <= verification["margin_abs_tolerance"] for gap in gaps)
    summary = {
        "n": n, "answer_agreement": same / n, "margin_within_share": within / n,
        "margin_gap_median": statistics.median(gaps), "margin_gap_max": max(gaps),
        "exact_share": sum(gap == 0 for gap in gaps) / n,
        "passed": same / n >= verification["answer_agreement_min"]
        and within / n >= verification["margin_within_share_min"],
    }
    if baseline_bf16 is not None:
        kept = [(l, r) for (l, r), b in zip(pairs, baseline_bf16) if abs(b) > NEAR_TIE]
        summary["answer_agreement_outside_bf16_near_ties"] = (
            sum(l["answer"] == r["answer"] for l, r in kept) / len(kept) if kept else None)
        summary["near_tie_inputs"] = n - len(kept)
    return summary


def baseline_row(entry):
    base = entry["baseline"]
    return {"answer": base["prediction"], "margin": base["fp32_variant_logit_margin"]}


def finite(rows):
    keys = ("margin", "margin_fp32", "margin_bf16", "margin_bf16_variant")
    return all(math.isfinite(row[k]) for row in rows for k in keys)


def spec_matches(planned, written, keys):
    return len(planned) == len(written) and all(all(p[k] == w[k] for k in keys) for p, w in zip(planned, written))


# ---------------------------------------------------------------- 실험 A

def check_a(stage, plan, run, verification, n_layers):
    work = a_work(plan, stage, n_layers)
    inputs = plan["inputs"]
    checks = {}

    def record(name, passed, detail=None):
        checks[name] = {"passed": bool(passed), "detail": detail}

    planned = {b["batch"]: b for b in work["batches"]}
    record("run_completed", run["meta"].get("status") == "completed", run["meta"].get("status"))
    record("phase1_complete", set(run["phase1"]) == set(work["phase1"]),
           {"planned": len(work["phase1"]), "written": len(run["phase1"])})
    keys = ("donor_kind", "donor", "layer")
    bad = [name for name, batch in planned.items()
           if name not in run["batches"] or not spec_matches(batch["rows"], run["batches"][name]["rows"], keys)]
    record("batches_complete", not bad and len(run["batches"]) == len(planned),
           {"planned": len(planned), "written": len(run["batches"]), "missing_or_wrong": bad[:5]})
    rows = [row for batch in run["batches"].values() for row in batch["rows"]] + list(run["phase1"].values())
    record("finite_values", finite(rows))
    phase1 = run["phase1"]
    ids = [i for i in work["phase1"] if i in phase1]
    summary = agreement([(phase1[i], baseline_row(inputs[i])) for i in ids], verification,
                        [inputs[i]["baseline"]["logit_margin"] for i in ids])
    record("reproduction_no_intervention", summary["passed"], summary)
    none_pairs, l0_pairs, final_pairs = [], [], []
    for name in planned:
        batch = run["batches"].get(name)
        if batch is None:
            continue
        none = batch["rows"][0]
        if batch["recipient"] in phase1:
            none_pairs.append((none, phase1[batch["recipient"]]))
        for row in batch["rows"][1:]:
            if row["layer"] == 0:
                l0_pairs.append((row, none))
            if row["layer"] == n_layers and row["donor"] in phase1:
                final_pairs.append((row, phase1[row["donor"]]))
    checks["none_rows_vs_phase1_reported"] = {"passed": True, "detail": agreement(none_pairs, verification)}
    l0 = agreement(l0_pairs, verification)
    record("l0_patch_identity", l0["passed"], l0)
    tolerance = verification["final_layer_donor_tolerance"]
    final = agreement(final_pairs, verification)
    final_ok = bool(final_pairs) and all(abs(r["margin"] - d["margin"]) <= tolerance for r, d in final_pairs)
    record("final_layer_reproduces_donor",
           final_ok and final["answer_agreement"] >= verification["answer_agreement_min"],
           {**final, "margin_within_0.01_all": final_ok})
    if stage == "pilot":
        equivalence(checks, work["reference_batches"], run, verification, ("donor_kind", "donor", "layer"))
    return checks


# ---------------------------------------------------------------- 실험 B

def check_b(stage, plan, run, verification):
    work = b_work(plan, stage)
    inputs = plan["inputs"]
    checks = {}

    def record(name, passed, detail=None):
        checks[name] = {"passed": bool(passed), "detail": detail}

    planned = {b["batch"]: b for b in work["batches"]}
    record("run_completed", run["meta"].get("status") == "completed", run["meta"].get("status"))
    keys = ("query", "kind", "target", "band")
    bad = [name for name, batch in planned.items()
           if name not in run["batches"] or not spec_matches(batch["rows"], run["batches"][name]["rows"], keys)]
    record("batches_complete", not bad and len(run["batches"]) == len(planned),
           {"planned": len(planned), "written": len(run["batches"]), "missing_or_wrong": bad[:5]})
    rows = [row for batch in run["batches"].values() for row in batch["rows"]] + list(run["reproduction"].values())
    record("finite_values", finite(rows))
    pairs, ties = [], []
    for name in planned:
        batch = run["batches"].get(name)
        if batch is not None:
            entry = inputs[batch["input_id"]]
            pairs.append((batch["rows"][0], baseline_row(entry)))
            ties.append(entry["baseline"]["logit_margin"])
    for input_id in work["reproduction"]:
        if input_id in run["reproduction"]:
            pairs.append((run["reproduction"][input_id], baseline_row(inputs[input_id])))
            ties.append(inputs[input_id]["baseline"]["logit_margin"])
    summary = agreement(pairs, verification, ties)
    record("reproduction_no_intervention", summary["passed"], summary)
    if stage == "pilot":
        if work["reproduction"]:
            record("reproduction_sample_complete", set(work["reproduction"]) <= set(run["reproduction"]),
                   {"planned": len(work["reproduction"]), "written": len(run["reproduction"])})
        zero = [(row, batch["rows"][0]) for batch in run["batches"].values() for row in batch["rows"]
                if row["kind"] == "zero"]
        summary = agreement(zero, verification)
        record("zero_intervention_identity", summary["passed"], summary)
        equivalence(checks, work["reference_batches"], run, verification, ("query", "kind", "target", "band"))
    return checks


def equivalence(checks, names, run, verification, keys):
    pairs, none_vs_baseline = [], []
    missing = [name for name in names if name not in run["reference"] or name not in run["batches"]]
    for name in names:
        if name in missing:
            continue
        fast, reference = run["batches"][name]["rows"], run["reference"][name]["rows"]
        if not spec_matches(fast, reference, keys):
            missing.append(name)
            continue
        pairs += list(zip(fast, reference))
    summary = agreement(pairs, verification)
    checks["fast_path_equivalence"] = {"passed": bool(summary["passed"]) and not missing,
                                       "detail": {**summary, "missing_reference_batches": missing[:5]}}


# ---------------------------------------------------------------- 시간 추정

def mean_by(rows, key, value):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(value(row))
    return {k: statistics.fmean(v) for k, v in groups.items()}


def fallback(table, n):
    return table.get(n, statistics.fmean(table.values()))


def estimate_a(plan, pilot_run, n_layers):
    batches = list(pilot_run["batches"].values())
    phase1 = mean_by(pilot_run["phase1"].values(), "n_images", lambda r: r["seconds"])
    prefix = mean_by([b for b in batches if b["prefix_seconds"] > 0], "n_images", lambda b: b["prefix_seconds"])
    per_row = mean_by(batches, "n_images", lambda b: b["seconds"] / len(b["rows"]))
    work = a_work(plan, "main", n_layers)
    inputs = plan["inputs"]
    seconds = sum(fallback(phase1, inputs[i]["case"]["n_images"]) for i in work["phase1"])
    recipients = {b["recipient"]: b["n_images"] for b in work["batches"]}
    seconds += sum(fallback(prefix, n) for n in recipients.values())
    seconds += sum(fallback(per_row, b["n_images"]) * len(b["rows"]) for b in work["batches"])
    return seconds / 3600


def estimate_b(plan, pilot_run):
    batches = list(pilot_run["batches"].values())
    prefix = mean_by(batches, "n_images", lambda b: b["prefix_seconds"])
    per_row = mean_by(batches, "n_images", lambda b: b["seconds"] / len(b["rows"]))
    work = b_work(plan, "main")
    seconds = sum(fallback(prefix, b["n_images"]) + fallback(per_row, b["n_images"]) * len(b["rows"])
                  for b in work["batches"])
    return seconds / 3600


def combined_estimate(plan, pilot_a, pilot_b, n_layers, limit_hours):
    require(pilot_a["meta"].get("status") == "completed" and pilot_b["meta"].get("status") == "completed",
            "Both pilots must be completed before the time estimate")
    hours = {"A": estimate_a(plan, pilot_a, n_layers), "B": estimate_b(plan, pilot_b)}
    total = (hours["A"] + hours["B"]) * OVERHEAD
    return {"status": "passed" if total <= limit_hours else "failed", "hours_main": hours,
            "overhead_factor": OVERHEAD, "total_hours_with_overhead": total, "limit_hours": limit_hours,
            "rule": "measured pilot seconds (means by image count) x planned main work; stop if above the limit"}


def finish(checks, kind, version):
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {"status": "passed" if not failed else "failed", "kind": kind, "frozen_version": version,
            "failed": failed, "checks": checks,
            "note": "Integrity only: no hypothesis contrast or decision is computed."}
