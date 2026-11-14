"""STEP 6-2 평가 분석 (CPU, confirmatory). 고정 manifest의 사전 등록을 그대로 적용한다. 선택 단계는 없다.

단위 = 확증 절반 강화 모집단(False-Yes) one-N 항목. 행 값은 같은 batch 무개입 행과의 차이:
  Δm(조건) = 개입 margin − 무개입 margin (fp32 변형 margin, STEP 5-2·5-3과 같음)
  Δm(Random) = 추첨 5개의 Δm 평균 (입력마다)
주 가설 (97.5% N 이미지 군집 부트스트랩, 클래스 군집은 robust 표기, 판정 구조는 STEP 5-3B와 같음):
  H1: D1 = Δm(Full N) − Δm(Random),          N arm = Δm(Full N),     최소 효과 −0.25
  H2: D2 = Δm(Orthogonal) − Δm(Parallel),    N arm = Δm(Orthogonal), 최소 효과 −0.25 (기저 v90, 원래 크기)
  보고만: H1의 Δm(Random), H2의 Δm(Parallel) (부호, 추정치, 97.5% CI)
제외 (모든 분석, 개수 보고): 추출 기록 없음 → 재구성 검사 실패 → 같은 batch 무개입 답이 기준 실행 답과 다름.
부가 분석은 95%이고 분석 이름에 표시한다.
"""
from collections import defaultdict

from src.core.causal_eval import (LABELS, PRIMARY_LEVEL, SECONDARY_LEVEL, decide_joint, mean, sign_consistent,
                                  summarize)
from src.core.feature_validation import require
from experiments.step6_contribution.conditions import BASES, LAMBDA, LAYERS, RANDOM_DRAWS, row_key


def key(condition, basis=None, size="raw", draw=None, lam=LAMBDA):
    return (condition, basis, size, draw, lam)


NONE = key("none", lam=0.0)
RANDOM = [key("random", draw=d) for d in range(RANDOM_DRAWS)]
ARMS = {
    "full": [key("full")],
    "random": RANDOM,
    "parallel_v90": [key("parallel", "v90")], "orthogonal_v90": [key("orthogonal", "v90")],
    "parallel_v90_match": [key("parallel", "v90", "match")], "orthogonal_v90_match": [key("orthogonal", "v90", "match")],
    "parallel_v95": [key("parallel", "v95")], "orthogonal_v95": [key("orthogonal", "v95")],
    "parallel_r64": [key("parallel", "r64")], "orthogonal_r64": [key("orthogonal", "r64")],
    "mean_difference_raw": [key("mean_difference")], "mean_difference_match": [key("mean_difference", size="match")],
}
HYPOTHESES = {"H1": {"contrast": ("full", "random"), "n_arm": "full", "reported_arm": "random"},
              "H2": {"contrast": ("orthogonal_v90", "parallel_v90"), "n_arm": "orthogonal_v90",
                     "reported_arm": "parallel_v90"}}
CONTRASTS = {
    "D1 full - random": ("full", "random"),
    "D2 orthogonal - parallel (v90)": ("orthogonal_v90", "parallel_v90"),
    "D2 orthogonal - parallel (v90, size-matched)": ("orthogonal_v90_match", "parallel_v90_match"),
    "D2 orthogonal - parallel (v95)": ("orthogonal_v95", "parallel_v95"),
    "D2 orthogonal - parallel (r64)": ("orthogonal_r64", "parallel_r64"),
    "mean difference (raw) - random": ("mean_difference_raw", "random"),
    "mean difference (size-matched) - random": ("mean_difference_match", "random"),
    "full - mean difference (size-matched)": ("full", "mean_difference_match"),
    "full - mean difference (raw)": ("full", "mean_difference_raw"),
}


# ---------------------------------------------------------------- 표

def index(run):
    """batch 이름 → {행 키: 행}."""
    return {name: {row_key(row): row for row in batch["rows"]} for name, batch in run["batches"].items()}


def arm_delta(rows, arm, outcome="margin"):
    return mean([rows[k][outcome] - rows[NONE][outcome] for k in ARMS[arm]])


def arm_flip(rows, arm, flipped="No"):
    return mean([int(rows[k]["answer"] == flipped) for k in ARMS[arm]])


def status_of(item, batch_name, table, extraction, baseline_answer):
    if item["item_id"] not in extraction or batch_name not in table:
        return "missing"
    if not extraction[item["item_id"]]["valid"]:
        return "invalid_extraction"
    if table[batch_name][NONE]["answer"] != baseline_answer:
        return "changed_none_answer"
    return "valid"


def primary_batch(item):
    slot = item["target_index"] if item["kind"] == "one_negative" else 0
    return f"{item['item_id']}#s{slot}"


# ---------------------------------------------------------------- 분석

def analyze(plan53, plan6, run, preregistration, comparison=None, reps=None, seed=None):
    statistics = preregistration["statistics"]
    reps = statistics["reps"] if reps is None else reps
    seed = statistics["seed"] if seed is None else seed
    items53 = {i["item_id"]: i for i in plan53["b_items"]}
    inputs = plan53["inputs"]
    table, extraction = index(run), run["extraction"]
    one_n = [items53[i] for i in plan6["items"]["main"]["one_negative"]]
    require(all(i["half"] == "confirmation" and i["population"] == "boost" for i in one_n),
            "Primary items must be confirmation-half boost items")
    meta = {i["item_id"]: (i["anchor"], i["class"]) for i in items53.values()}
    status = {i["item_id"]: status_of(i, primary_batch(i), table, extraction,
                                      inputs[i["input_id"]]["baseline"]["prediction"]) for i in one_n}
    valid = [i for i in one_n if status[i["item_id"]] == "valid"]
    rows_of = lambda item: table[primary_batch(item)]
    hypotheses = preregistration["hypotheses"]

    def values(chosen, arm=None, contrast=None, outcome="margin"):
        keys, out = [], []
        for item in chosen:
            rows = rows_of(item)
            if contrast is not None:
                out.append(arm_delta(rows, contrast[0], outcome) - arm_delta(rows, contrast[1], outcome))
            else:
                out.append(arm_delta(rows, arm, outcome))
            keys.append(item["item_id"])
        return keys, out

    # (1) 주 판정
    primary = []
    for name, spec in HYPOTHESES.items():
        minimum = hypotheses[name]["minimum_effect"]
        keys, contrast = values(valid, contrast=spec["contrast"])
        _, n_arm = values(valid, arm=spec["n_arm"])
        _, reported = values(valid, arm=spec["reported_arm"])
        c_row = summarize(meta, keys, contrast, PRIMARY_LEVEL, reps, seed)
        n_row = summarize(meta, keys, n_arm, PRIMARY_LEVEL, reps, seed)
        r_row = summarize(meta, keys, reported, PRIMARY_LEVEL, reps, seed)
        label, unmet = decide_joint(c_row, n_row, minimum, "negative")
        primary.append({"hypothesis": name, "contrast_arms": list(spec["contrast"]), "n_arm_name": spec["n_arm"],
                        "minimum_effect": minimum, "direction": "negative", "label": label, "unmet_conditions": unmet,
                        "contrast": c_row, "n_arm": n_row,
                        "reported_arm": {"arm": spec["reported_arm"], **r_row, "sign": sign(r_row["estimate"])}})

    # (2) 부가 (95%)
    secondary = []

    def add(analysis, chosen, arm=None, contrast=None, outcome="margin"):
        keys, vals = values(chosen, arm, contrast, outcome)
        secondary.append({"analysis": analysis, "arm": arm, "contrast": list(contrast) if contrast else None,
                          "outcome": outcome, **summarize(meta, keys, vals, SECONDARY_LEVEL, reps, seed)})

    def add_flip(analysis, chosen, arm=None, contrast=None):
        keys, vals = [], []
        for item in chosen:
            rows = rows_of(item)
            vals.append(arm_flip(rows, contrast[0]) - arm_flip(rows, contrast[1]) if contrast else arm_flip(rows, arm))
            keys.append(item["item_id"])
        secondary.append({"analysis": analysis, "arm": arm, "contrast": list(contrast) if contrast else None,
                          "outcome": "answer_change_rate_yes_to_no",
                          **summarize(meta, keys, vals, SECONDARY_LEVEL, reps, seed)})

    for arm in ARMS:
        add("delta margin by condition", valid, arm=arm)
        add_flip("answer change rate (Yes -> No) by condition", valid, arm=arm)
    for label, contrast in CONTRASTS.items():
        add(f"contrast: {label}", valid, contrast=contrast)
    add_flip("answer change rate difference: full - random", valid, contrast=("full", "random"))
    add_flip("answer change rate difference: orthogonal - parallel (v90)", valid,
             contrast=("orthogonal_v90", "parallel_v90"))
    for n in (3, 4, 5):
        chosen = [i for i in valid if i["n_images"] == n]
        subgroup(add, add_flip, f"{n} images", chosen)
    for position in ("first", "middle", "last"):
        chosen = [i for i in valid if i["position"] == position]
        subgroup(add, add_flip, f"N position {position}", chosen)
    with_changed = [i for i in one_n if status[i["item_id"]] in ("valid", "changed_none_answer")]
    for label, contrast in (("D1", ("full", "random")), ("D2", ("orthogonal_v90", "parallel_v90"))):
        add(f"sensitivity: {label} including inputs whose no-intervention answer changed", with_changed,
            contrast=contrast)
        add(f"sensitivity: {label} with the bf16 margin", valid, contrast=contrast, outcome="margin_bf16")
        add(f"sensitivity: {label} without inputs whose baseline bf16 margin sign disagrees with the answer",
            [i for i in valid if sign_consistent(inputs[i["input_id"]])], contrast=contrast)

    comparison_rows = compare_5_3b(comparison, valid, rows_of, meta, inputs, reps, seed) if comparison else None
    side_effects, warnings = all_positive(plan6, items53, table, extraction, inputs, meta, reps, seed)
    descriptive = vectors_descriptive(run, valid, meta, reps, seed)
    counts = {"planned": len(one_n), "analyzed": len(valid)}
    for reason in ("missing", "invalid_extraction", "changed_none_answer"):
        counts[f"excluded_{reason}"] = sum(s == reason for s in status.values())
    return {"primary": primary, "secondary": secondary, "comparison_5_3b": comparison_rows,
            "all_positive_side_effects": side_effects, "warnings": warnings, "descriptive": descriptive,
            "counts": counts,
            "excluded_items": {reason: sorted(i for i, s in status.items() if s == reason)
                               for reason in ("missing", "invalid_extraction", "changed_none_answer")}}


def sign(value):
    return None if value is None else ("+" if value > 0 else "-" if value < 0 else "0")


def subgroup(add, add_flip, label, chosen):
    for arm in ("full", "random", "parallel_v90", "orthogonal_v90"):
        add(f"{label}: delta margin", chosen, arm=arm)
    add(f"{label}: D1", chosen, contrast=("full", "random"))
    add(f"{label}: D2", chosen, contrast=("orthogonal_v90", "parallel_v90"))
    add_flip(f"{label}: answer change rate", chosen, arm="full")


# ---------------------------------------------------------------- 5-3B와 나란히

def compare_5_3b(comparison, valid, rows_of, meta, inputs, reps, seed):
    """같은 입력에서 STEP 5-3B 강화(구간 21–24, 질문 구간, N 대상, softmax 4배)와 Full N(기여 4배)을 나란히 본다.

    두 실행에서 모두 제외되지 않은 입력만 쓴다 (5-3B 제외 = 그 batch 무개입 답이 기준 답과 다름).
    """
    table53, band = comparison["table"], tuple(comparison["band"])
    common = [i for i in valid if i["item_id"] in table53
              and table53[i["item_id"]]["none"]["answer"] == inputs[i["input_id"]]["baseline"]["prediction"]]
    rows = []

    def add(analysis, function):
        keys = [i["item_id"] for i in common]
        rows.append({"analysis": analysis, "outcome": analysis.split(":")[0],
                     **summarize(meta, keys, [function(i) for i in common], SECONDARY_LEVEL, reps, seed)})

    def boost(item):
        return table53[item["item_id"]][("post_image", "boost", "N", band)]

    def delta53(item):
        return boost(item)["margin"] - table53[item["item_id"]]["none"]["margin"]

    add("delta margin: STEP 5-3B boost N (attention odds x4, renormalized)", delta53)
    add("delta margin: STEP 6 full N (contribution x4, not renormalized)", lambda i: arm_delta(rows_of(i), "full"))
    add("delta margin: STEP 6 full N - STEP 5-3B boost N", lambda i: arm_delta(rows_of(i), "full") - delta53(i))
    add("answer change rate: STEP 5-3B boost N", lambda i: int(boost(i)["answer"] == "No"))
    add("answer change rate: STEP 6 full N", lambda i: arm_flip(rows_of(i), "full"))
    add("answer change rate: STEP 6 full N - STEP 5-3B boost N",
        lambda i: arm_flip(rows_of(i), "full") - int(boost(i)["answer"] == "No"))
    return {"band": list(band), "n_common": len(common), "n_step6_valid": len(valid), "rows": rows}


# ---------------------------------------------------------------- All-P 부작용

def all_positive(plan6, items53, table, extraction, inputs, meta, reps, seed):
    """확증 절반 All-P 항목. 주 규칙: 자리 0(첫 번째 P)을 가짜 N으로. 민감도: 모든 자리."""
    chosen = [items53[i] for i in plan6["items"]["main"]["all_positive"]]
    rows, warnings = [], []
    counts = defaultdict(int)
    usable = {}
    for item in chosen:
        reason = status_of(item, f"{item['item_id']}#s0", table, extraction,
                           inputs[item["input_id"]]["baseline"]["prediction"])
        if reason == "valid" and table[f"{item['item_id']}#s0"][NONE]["answer"] != "Yes":
            reason = "none_answer_not_yes"
        counts[reason] += 1
        if reason == "valid":
            usable[item["item_id"]] = item
    for arm in ARMS:
        keys = sorted(usable)
        flips = [arm_flip(table[f"{i}#s0"], arm) for i in keys]
        deltas = [arm_delta(table[f"{i}#s0"], arm) for i in keys]
        flip = summarize(meta, keys, flips, SECONDARY_LEVEL, reps, seed)
        rows.append({"rule": "slot 0 (primary)", "arm": arm, "outcome": "yes_to_no_rate", **flip})
        rows.append({"rule": "slot 0 (primary)", "arm": arm, "outcome": "delta_margin",
                     **summarize(meta, keys, deltas, SECONDARY_LEVEL, reps, seed)})
        if flip["estimate"] is not None and flip["estimate"] > 0.05:
            warnings.append(f"All-P Yes->No rate {flip['estimate']:.3f} > 0.05 (condition {arm}, slot 0)")
        for slot in range(5):
            keys_slot = [i for i in keys if usable[i]["n_images"] > slot
                         and table[f"{i}#s{slot}"][NONE]["answer"] == "Yes"]
            values = [arm_flip(table[f"{i}#s{slot}"], arm) for i in keys_slot]
            rows.append({"rule": f"slot {slot} (sensitivity)", "arm": arm, "outcome": "yes_to_no_rate",
                         **summarize(meta, keys_slot, values, SECONDARY_LEVEL, reps, seed)})
        every_keys, every = [], []
        for i in keys:
            slots = [s for s in range(usable[i]["n_images"]) if table[f"{i}#s{s}"][NONE]["answer"] == "Yes"]
            if slots:
                every_keys.append(i)
                every.append(mean([arm_flip(table[f"{i}#s{s}"], arm) for s in slots]))
        rows.append({"rule": "mean over all slots (sensitivity)", "arm": arm, "outcome": "yes_to_no_rate",
                     **summarize(meta, every_keys, every, SECONDARY_LEVEL, reps, seed)})
    return {"counts": dict(counts), "rows": rows}, warnings


# ---------------------------------------------------------------- 벡터 기술 통계

def vectors_descriptive(run, valid, meta, reps, seed):
    """기술 통계 (95%): 층·기저별 직교 잔차 비율 ‖(I − UUᵀ)c_N‖/‖c_N‖, 평행 성분 에너지, c_N 크기 등 (입력별 토큰 평균)."""
    rows = []
    fields = ["c_norm_mean", "c_over_o_norm", "target_attention_share", "cos_mean_difference"]
    fields += [f"{kind}_{name}" for name in BASES for kind in ("orthogonal_ratio", "parallel_energy")]
    keys = [i["item_id"] for i in valid]
    if not valid:
        return {"vector_summary": rows}
    layers = sorted(int(k) for k in run["batches"][primary_batch(valid[0])]["vector_summary"])   # 본 실행 = LAYERS
    for layer in layers:
        for field in fields:
            values = [run["batches"][primary_batch(i)]["vector_summary"][str(layer)][field] for i in valid]
            rows.append({"layer": layer + 1, "field": field,
                         **summarize(meta, keys, values, SECONDARY_LEVEL, reps, seed),
                         "quartiles": quartiles(values)})
    return {"vector_summary": rows}


def quartiles(values):
    if not values:
        return None
    ordered = sorted(values)

    def at(q):
        position = q * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"min": ordered[0], "q1": at(0.25), "median": at(0.5), "q3": at(0.75), "max": ordered[-1]}
