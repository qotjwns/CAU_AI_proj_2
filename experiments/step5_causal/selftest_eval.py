"""STEP 5-3 가짜 결과 시험 (CPU): 실제 계획 파일 구조로 가짜 실행 결과를 만들어 검사·평가 코드를 시험한다.

시나리오 (모델 출력 없음, 판정 기대값을 미리 정해 둔다)
- A 효과 0: 두 donor의 성공 확률이 모든 층에서 같다 → 지지가 나오면 안 된다.
- A 효과 심음: 같은 묶음 donor 성공 확률이 층 20에서 +0.35 → ℓ* = 20, 두 방향 지지.
- A 동률: 층 5와 7에서 같은 점수 → 가장 이른 층 5.
- B 효과 0 → 지지 없음. B 효과 심음: 차단 구간 25–28에서 Δm(N) = +1.2 → B-H1 지지.
  강화에서 대비만 P 쪽에서 생긴 경우(Δm(N) = 0, Δm(P) = +1.0) → B-H2 결론 유보 (조건: N 대상).
  강화 Δm(N) = −1.0 → B-H2 지지. 동률 구간 → 가장 이른 구간. All-P 뒤집힘 10% → 경고.
- 무결성 검사: 정상 → 통과, 재현 3% 불일치 → 실패, 참조 경로 불일치 → 실패.
"""
import copy
import random

from src.common.config import read_yaml
from src.common.paths import project_path
from src.core.causal_check import check_a, check_b
from src.core.causal_conditions import A_TARGET, a_work, b_work
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
    return result(base["prediction"], base["fp32_variant_logit_margin"])


# ---------------------------------------------------------------- 가짜 A

def fake_a(plan, stage, probability, seed=7, allp_flip=0.02):
    """probability(direction, donor_kind, layer) → 목표 답이 나올 확률."""
    rng = random.Random(seed)
    work = a_work(plan, stage, N_LAYERS)
    inputs = plan["inputs"]
    phase1 = {i: {"input_id": i, "n_images": inputs[i]["case"]["n_images"], "seconds": 0.3,
                  **baseline_result(inputs[i])} for i in work["phase1"]}
    batches = {}
    for batch in work["batches"]:
        none = {k: phase1[batch["recipient"]][k] for k in ("answer", "raw", "first_token", "margin", "margin_fp32",
                                                            "margin_bf16", "margin_bf16_variant")}
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

def fake_b(plan, stage, effect, seed=11, allp_flip=0.0, noise=1.0):
    """effect(population or kind, query, target, band) → Δm 평균."""
    rng = random.Random(seed)
    work = b_work(plan, stage)
    inputs = plan["inputs"]
    batches = {}
    for batch in work["batches"]:
        item = work["items"][batch["item_id"]]
        none = baseline_result(inputs[batch["input_id"]])
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

    bad = copy.deepcopy(fake_a(plan, "main", null))
    flipped = 0
    for i, row in enumerate(bad["phase1"].values()):
        if i % 33 == 0:
            row["answer"] = "No" if row["answer"] == "Yes" else "Yes"
            flipped += 1
    checks = check_a("main", plan, bad, verification, N_LAYERS)
    require(not checks["reproduction_no_intervention"]["passed"], "3% answer mismatch was not caught")
    outcomes["A_reproduction_mismatch_caught"] = checks["reproduction_no_intervention"]["detail"]["answer_agreement"]

    run = fake_b(plan, "main", b_null)
    checks = check_b("main", plan, run, verification)
    require(all(c["passed"] for c in checks.values()), f"B clean fake run failed integrity: "
            f"{[k for k, c in checks.items() if not c['passed']]}")
    report = analyze_b(plan, run, preregistration_b)
    require(not any(row["label"] in SUPPORT for row in report["primary"]), "B null scenario produced support")
    outcomes["B_null"] = {r["hypothesis"]: r["label"] for r in report["primary"]}

    run = fake_b(plan, "main", b_planted, allp_flip=0.10)
    report = analyze_b(plan, run, preregistration_b)
    rows = {r["hypothesis"]: r for r in report["primary"]}
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
