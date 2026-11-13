"""STEP 5-3 가짜 결과 시험 (CPU): 실제 계획 파일 구조로 가짜 실행 결과를 만들어 검사·평가 코드를 시험한다.

시나리오 (모델 출력 없음, 판정 기대값을 미리 정해 둔다)
- A 효과 0: 두 donor의 성공 확률이 모든 층에서 같다 → 지지가 나오면 안 된다.
- A 효과 심음: 같은 묶음 donor 성공 확률이 층 20에서 +0.35 → ℓ* = 20, 두 방향 지지.
- A 동률: 층 5와 7에서 같은 점수 → 가장 이른 층 5.
- B 효과 0 → 지지 없음. B 효과 심음: 차단 구간 25–28에서 Δm(N) = +1.2 → B-H1 지지.
  강화에서 대비만 P 쪽에서 생긴 경우(Δm(N) = 0, Δm(P) = +1.0) → B-H2 결론 유보 (조건: N 대상).
  강화 Δm(N) = −1.0 → B-H2 지지. 동률 구간 → 가장 이른 구간. All-P 뒤집힘 10% → 경고.
- 무결성 검사: 정상 → 통과, 동점 근처가 아닌 입력 3% 불일치 → 실패, 동점 근처 입력만 불일치 → 통과(전체 일치율은
  참고값으로 99% 미만), 참조 경로 불일치 → 실패.
- 제외 규칙(고정본 v2): 무개입 답이 바뀐 입력(A: 짝 입력·batch 무개입 행, B: 무개입 행)은 분석에서 빠지고 개수가
  보고되며, 심은 효과는 여전히 지지된다.
- 고정본 v3 회귀 시험: C1 donor가 없는 단위의 짝 입력·무개입 행이 바뀌어도 평가가 끝까지 돌고, '바뀐 단위 포함'
  민감도 분석은 C1 donor가 있는 단위 전체가 된다 (v2 코드는 여기서 KeyError로 멈췄다).
"""
import copy
import random

from src.common.config import read_yaml
from src.common.paths import project_path
from src.core.causal_check import check_a, check_b
from src.core.causal_conditions import A_DIRECTIONS, A_TARGET, a_work, b_work
from src.core.causal_eval import LABELS, analyze_a, analyze_b
from src.core.causal_freeze import experiment_config
from src.core.causal_plan import load_plan
from src.core.feature_validation import require


N_LAYERS = 36
SUPPORT = (LABELS["robust"], LABELS["support"])


def result(answer, margin):
    return {"answer": answer, "raw": answer or "?", "first_token": 0, "margin": margin, "margin_fp32": margin,
            "margin_bf16": margin, "margin_bf16_variant": margin}


def baseline_result(entry):
    base = entry["baseline"]
    row = result(base["prediction"], base["fp32_variant_logit_margin"])
    row["margin_bf16"] = base["logit_margin"]
    return row


def flipped(row):
    return {**row, "answer": "No" if row["answer"] == "Yes" else "Yes"}


def near_tie(entry):
    return abs(entry["baseline"]["logit_margin"]) <= 0.125


# ---------------------------------------------------------------- 가짜 A

def fake_a(plan, stage, probability, seed=7, allp_flip=0.02, changed=(), changed_none=()):
    """probability(direction, donor_kind, layer) → 목표 답이 나올 확률.
    changed: 무개입 답(phase 1)이 기준과 달라진 입력, changed_none: 무개입 행만 달라진 batch."""
    rng = random.Random(seed)
    work = a_work(plan, stage, N_LAYERS)
    inputs = plan["inputs"]
    phase1 = {i: {"input_id": i, "n_images": inputs[i]["case"]["n_images"], "seconds": 0.3,
                  **baseline_result(inputs[i])} for i in work["phase1"]}
    for i in changed:
        phase1[i] = flipped(phase1[i])
    batches = {}
    for batch in work["batches"]:
        none = {k: phase1[batch["recipient"]][k] for k in ("answer", "raw", "first_token", "margin", "margin_fp32",
                                                            "margin_bf16", "margin_bf16_variant")}
        if batch["batch"] in changed_none:
            none = flipped(none)
        rows = []
        shared = {layer: rng.random() for layer in range(N_LAYERS + 1)}
        for spec in batch["rows"]:
            layer = spec["layer"]
            if spec["donor_kind"] == "none" or layer == 0:
                out = dict(none)
            elif layer == N_LAYERS:
                donor = phase1[spec["donor"]]
                out = {k: donor[k] for k in none}
            elif batch["kind"] == "allp":
                flip = rng.random() < allp_flip
                out = result("No" if flip else none["answer"], none["margin"] - (2.0 if flip else 0.0))
            else:
                target = A_TARGET[batch["direction"]]
                p = probability(batch["direction"], spec["donor_kind"], layer)
                draw = shared[layer] if probability is tie else rng.random()     # 동률 시나리오만 두 donor가 같은 난수
                hit = draw < p
                sign = 1.0 if target == "Yes" else -1.0
                out = result(target if hit else none["answer"], none["margin"] + (sign * 2.0 if hit else 0.0))
            rows.append({**spec, **out})
        batches[batch["batch"]] = {**{k: batch[k] for k in ("batch", "kind", "unit_id", "direction", "recipient",
                                                            "n_images", "half")},
                                   "seconds": 0.2, "prefix_seconds": 0.3, "rows": rows}
    return {"meta": {"status": "completed", "n_layers": N_LAYERS, "stage": stage}, "phase1": phase1,
            "batches": batches, "reference": {}, "reproduction": {}}


def null(direction, donor, layer):
    return 0.3


def planted(direction, donor, layer):
    bump = {20: 0.35, 19: 0.15, 21: 0.15}.get(layer, 0.0)
    return 0.3 + (bump if donor == "same" else 0.0)


def tie(direction, donor, layer):
    if layer in (5, 7):
        return 1.0 if donor == "same" else 0.0
    return 0.3


# ---------------------------------------------------------------- 가짜 B

def fake_b(plan, stage, effect, seed=11, allp_flip=0.0, noise=1.0, changed=()):
    """effect(population or kind, query, target, band) → Δm 평균. changed: 무개입 행 답이 기준과 달라진 항목."""
    rng = random.Random(seed)
    work = b_work(plan, stage)
    inputs = plan["inputs"]
    batches = {}
    for batch in work["batches"]:
        item = work["items"][batch["item_id"]]
        none = baseline_result(inputs[batch["input_id"]])
        if batch["item_id"] in changed:
            none = flipped(none)
        rows = [{**batch["rows"][0], **none}]
        for spec in batch["rows"][1:]:
            if spec["kind"] == "zero":
                rows.append({**spec, **none})
                continue
            band = tuple(spec["band"])
            if item["population"] == "all_positive":
                flip = spec["kind"] == "boost" and spec["target"] == "N" and band == (17, 20) and rng.random() < allp_flip
                rows.append({**spec, **result("No" if flip else none["answer"], none["margin"] - (3.0 if flip else 0.0))})
                continue
            mean_delta = effect(spec["kind"], spec["query"], spec["target"], band)
            delta = mean_delta + (rng.gauss(0, noise) if noise else 0.0)
            margin = none["margin"] + delta
            rows.append({**spec, **result("Yes" if margin > 0 else "No", margin)})
        batches[batch["batch"]] = {**{k: batch[k] for k in ("batch", "item_id", "input_id", "n_images", "half")},
                                   "seconds": 0.2, "prefix_seconds": 0.3, "rows": rows}
    reproduction = {i: {"input_id": i, "n_images": inputs[i]["case"]["n_images"], "seconds": 0.3,
                        **baseline_result(inputs[i])} for i in work["reproduction"]}
    return {"meta": {"status": "completed", "n_layers": N_LAYERS, "stage": stage}, "phase1": {}, "batches": batches,
            "reference": {}, "reproduction": reproduction}


def b_null(kind, query, target, band):
    return 0.0


def b_planted(kind, query, target, band):
    if kind == "block" and query == "post_image" and band == (25, 28) and target == "N":
        return 1.2
    if kind == "boost" and query == "post_image" and band == (17, 20):
        return -1.0 if target == "N" else 0.0
    return 0.0


def b_p_side(kind, query, target, band):
    if kind == "boost" and query == "post_image" and band == (17, 20):
        return 1.0 if target == "P" else 0.0
    return b_planted(kind, query, target, band) if kind == "block" else 0.0


def b_tie(kind, query, target, band):
    if kind == "block" and query == "post_image" and target == "N" and band in ((9, 12), (25, 28)):
        return 1.0
    return 0.0


# ---------------------------------------------------------------- 시험

def eval_test():
    _, config = experiment_config("A")
    plan = load_plan(read_yaml(project_path(config["split_config"])))
    preregistration_a = config["preregistration"]
    preregistration_b = experiment_config("B")[1]["preregistration"]
    verification = config["verification"]
    outcomes = {}

    run = fake_a(plan, "main", null)
    checks = check_a("main", plan, run, verification, N_LAYERS)
    require(all(c["passed"] for c in checks.values()), f"A clean fake run failed integrity: "
            f"{[k for k, c in checks.items() if not c['passed']]}")
    report = analyze_a(plan, run, preregistration_a, N_LAYERS)
    labels = [row["label"] for row in report["primary"]]
    require(not any(label in SUPPORT for label in labels), f"A null scenario produced support: {labels}")
    outcomes["A_null"] = {r["hypothesis"]: (round(r["estimate"], 3), r["label"]) for r in report["primary"]}

    run = fake_a(plan, "main", planted)
    recorded = {}
    report = analyze_a(plan, run, preregistration_a, N_LAYERS, on_selection=recorded.update)
    require(recorded and all(report["selection"][d]["layer"] == 20 for d in ("restore", "contaminate")),
            f"A planted layer not selected: {report['selection']}")
    require(all(row["label"] in SUPPORT and row["layer"] == 20 for row in report["primary"]),
            f"A planted effect not supported: {[r['label'] for r in report['primary']]}")
    first = [r for r in report["secondary"] if r["analysis"].startswith("original definition")]
    require(len(first) == 2 and all(r["n"] > 0 for r in first), "Original-definition secondary analysis missing")
    outcomes["A_planted"] = {r["hypothesis"]: (r["layer"], round(r["estimate"], 3), r["label"])
                             for r in report["primary"]}
    outcomes["A_planted_earliest"] = report["descriptive"]["earliest_restore_ge_half"]

    run = fake_a(plan, "main", tie)
    report = analyze_a(plan, run, preregistration_a, N_LAYERS)
    require(all(report["selection"][d]["layer"] == 5 for d in ("restore", "contaminate")),
            f"A tie not resolved to the earliest layer: {report['selection']}")
    outcomes["A_tie_layer"] = report["selection"]["restore"]["layer"]

    inputs = plan["inputs"]
    ids = sorted(fake_a(plan, "main", null)["phase1"])
    far = [i for i in ids if not near_tie(inputs[i])]
    bad = fake_a(plan, "main", null, changed=far[::33])
    checks = check_a("main", plan, bad, verification, N_LAYERS)
    detail = checks["reproduction_no_intervention"]["detail"]
    require(not checks["reproduction_no_intervention"]["passed"], "3% answer mismatch outside near-ties was not caught")
    outcomes["A_reproduction_mismatch_caught"] = round(detail["answer_agreement_outside_near_tie"], 4)
    ties = [i for i in ids if near_tie(inputs[i])]
    tie_run = fake_a(plan, "main", null, changed=ties)
    checks = check_a("main", plan, tie_run, verification, N_LAYERS)
    detail = checks["reproduction_no_intervention"]["detail"]
    require(checks["reproduction_no_intervention"]["passed"] and detail["answer_agreement_all_reported"] < 0.99,
            "Near-tie-only flips should pass the v2 answer rule and show below 99% in the all-pair reference value")
    outcomes["A_near_tie_flips_pass"] = {"near_tie_inputs": len(ties),
                                         "all_pairs_reference": round(detail["answer_agreement_all_reported"], 4),
                                         "outside_near_tie": detail["answer_agreement_outside_near_tie"]}

    confirmation = [u for u in plan["a_units"] if u["half"] == "confirmation"]
    changed = [u["false_yes_input"] for u in confirmation[:6]] + [confirmation[6]["correct_no_input"]]
    changed_none = {f"{confirmation[10]['unit_id']}::restore", f"{confirmation[11]['unit_id']}::contaminate"}
    run = fake_a(plan, "main", planted, changed=changed, changed_none=changed_none)
    report = analyze_a(plan, run, preregistration_a, N_LAYERS)
    rows = {r["hypothesis"]: r for r in report["primary"]}
    require(all(r["label"] in SUPPORT for r in rows.values()), "Planted effect lost after exclusions")
    require(rows["A-H1"]["excluded"]["changed_pair"] >= 6 and rows["A-H1"]["excluded"]["changed_none_row"] >= 1
            and rows["A-H2"]["excluded"]["changed_none_row"] >= 1 and set(changed) <= set(report["changed_inputs"]),
            f"Exclusions not applied: {rows['A-H1']['excluded']} {rows['A-H2']['excluded']}")
    sensitivity = [r for r in report["secondary"] if r["analysis"].startswith("sensitivity: including")]
    require(all(s["n"] > rows[h]["n"] for s, h in zip(sensitivity, ("A-H1", "A-H2"))),
            "Sensitivity analysis including changed units is missing")
    outcomes["A_exclusions"] = {h: r["excluded"] for h, r in rows.items()}

    no_c1 = [(u, d) for u in confirmation for d in A_DIRECTIONS if u[d]["c1_donor"] is None]
    require(len(no_c1) >= 2, "The plan has too few C1-less confirmation units for the v3 regression test")
    run = fake_a(plan, "main", planted, changed=[u["false_yes_input"] for u, _ in no_c1[::2]],
                 changed_none={f"{u['unit_id']}::{d}" for u, d in no_c1[1::2]})
    report = analyze_a(plan, run, preregistration_a, N_LAYERS)
    sensitivity = {r["direction"]: r["n"] for r in report["secondary"]
                   if r["analysis"].startswith("sensitivity: including")}
    expected = {d: sum(1 for u in confirmation if u[d]["c1_donor"] is not None) for d in A_DIRECTIONS}
    require(sensitivity == expected, f"Sensitivity population is not all units with a C1 donor: {sensitivity} vs {expected}")
    outcomes["A_no_c1_changed_v3"] = {"c1_less_units_changed": len(no_c1), "sensitivity_n": sensitivity}

    run = fake_b(plan, "main", b_null)
    checks = check_b("main", plan, run, verification)
    require(all(c["passed"] for c in checks.values()), f"B clean fake run failed integrity: "
            f"{[k for k, c in checks.items() if not c['passed']]}")
    report = analyze_b(plan, run, preregistration_b)
    require(not any(row["label"] in SUPPORT for row in report["primary"]), "B null scenario produced support")
    outcomes["B_null"] = {r["hypothesis"]: r["label"] for r in report["primary"]}

    changed_items = [i["item_id"] for i in plan["b_items"] if i["half"] == "confirmation"][:40:4]
    run = fake_b(plan, "main", b_planted, allp_flip=0.10, changed=changed_items)
    report = analyze_b(plan, run, preregistration_b)
    rows = {r["hypothesis"]: r for r in report["primary"]}
    excluded = sum(r["excluded_changed_no_intervention_answer"] for r in rows.values())
    expected = sum(1 for i in plan["b_items"] if i["item_id"] in set(changed_items) and i["population"] != "all_positive")
    require(expected > 0 and excluded == expected, f"B exclusions not applied: {excluded} vs {expected}")
    outcomes["B_exclusions"] = {k: v["excluded_changed_no_intervention_answer"] for k, v in rows.items()}
    require(rows["B-H1"]["label"] in SUPPORT and rows["B-H1"]["band"] == [25, 28], f"B-H1 planted: {rows['B-H1']}")
    require(rows["B-H2"]["label"] in SUPPORT and rows["B-H2"]["band"] == [17, 20], f"B-H2 planted: {rows['B-H2']}")
    require(report["warnings"], "All-P flips above 5% were not flagged")
    outcomes["B_planted"] = {k: (v["band"], round(v["contrast"]["estimate"], 3), v["label"]) for k, v in rows.items()}
    outcomes["B_warning"] = report["warnings"][:2]

    run = fake_b(plan, "main", b_p_side)
    report = analyze_b(plan, run, preregistration_b)
    row = next(r for r in report["primary"] if r["hypothesis"] == "B-H2")
    require(row["label"] == LABELS["other"] and row["unmet_conditions"] == ["n_arm"],
            f"P-driven contrast was not held back by the N-arm condition: {row['label']} {row['unmet_conditions']}")
    outcomes["B_p_driven"] = (round(row["contrast"]["estimate"], 3), round(row["n_arm"]["estimate"], 3), row["label"],
                              row["unmet_conditions"], row["p_arm_reported"]["sign"])

    run = fake_b(plan, "main", b_tie, noise=0.0)
    report = analyze_b(plan, run, preregistration_b)
    require(report["selection"]["block"]["band"] == [9, 12], f"B tie not resolved to the earliest band: "
            f"{report['selection']['block']}")
    outcomes["B_tie_band"] = report["selection"]["block"]["band"]

    pilot = fake_b(plan, "pilot", b_null)
    pilot["reference"] = {name: {"batch": name, "rows": copy.deepcopy(batch["rows"])}
                          for name, batch in pilot["batches"].items()}
    checks = check_b("pilot", plan, pilot, verification)
    require(checks["fast_path_equivalence"]["passed"] and checks["zero_intervention_identity"]["passed"],
            "Clean B pilot failed equivalence")
    for i, batch in enumerate(pilot["reference"].values()):
        for j, row in enumerate(batch["rows"]):
            if (i + j) % 10 == 0:
                row["margin"] += 0.5
    checks = check_b("pilot", plan, pilot, verification)
    require(not checks["fast_path_equivalence"]["passed"], "Reference mismatch was not caught")
    outcomes["B_equivalence_mismatch_caught"] = checks["fast_path_equivalence"]["detail"]["margin_within_share"]
    return outcomes
