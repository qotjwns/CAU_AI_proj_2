"""STEP 5-3 평가 (CPU, confirmatory, 한 번만).

고정 manifest의 사전 등록을 그대로 적용한다. 순서: (1) 선택 절반 결과만으로 층(A)·구간(B)을 고르고 기록 →
(2) 확증 절반에서 주 판정 → (3) 부가·민감도·기술 통계. 결과를 본 뒤 대비와 판정 기준을 바꾸지 않는다.

제외 (사전 등록 v2): 새 계산 경로의 무개입 답이 기준 실행 답과 달라진 입력은 모든 분석(두 절반)에서 빼고 개수를
보고한다. 민감도 분석으로 그 입력을 포함한 결과도 낸다.
실험 A (단위 = 같은 계열·장수의 False-Yes 입력과 N-last Correct-No 입력 짝, 위치 첫·가운데):
  성공 = 이식 후 파싱한 답이 목표 답 (복원 No, 오염 Yes). 단위 값 = 성공(같은 묶음) − 성공(C1).
  단위·방향 제외 사유 (앞의 것부터): 짝 입력의 무개입 답 변화(phase 1) → 그 batch 무개입 행의 답 변화 →
  C1 없음 → C1 donor의 무개입 답 변화.
  ℓ*_방향 = 선택 절반 argmax_{ℓ=1..35} Σ 단위 값 (정수, 동률이면 가장 이른 층).
실험 B (항목 = 입력): Δm(대상) = 개입 margin − 같은 batch 무개입 margin (fp32 변형). C = Δm(N) − Δm(P).
  구간* = 선택 절반 주 query에서 차단은 평균 C 최대, 강화는 평균 C 최소 (1e-9 이내는 동률 → 가장 이른 구간).
통계: N 이미지 군집 percentile 부트스트랩 (확증 절반 안), 클래스 군집은 robust 표기. 주 판정 97.5%, 그 외 95%.
"""
from collections import defaultdict
import math

from src.core.accumulation_explore import Clusters
from src.core.causal_conditions import A_DIRECTIONS, A_TARGET, ALLP_DIRECTIONS, BANDS
from src.core.feature_validation import require


PRIMARY_LEVEL = 0.975
SECONDARY_LEVEL = 0.95
TIE = 1e-9
LABELS = {"robust": "robust support", "support": "support (cross-class generalization uncertain)",
          "none": "no meaningful difference", "other": "inconclusive", "missing": "not estimable"}


def mean(values):
    return math.fsum(values) / len(values) if values else None


def summarize(meta, keys, values, level, reps, seed):
    """meta: 단위 → (N 이미지, 클래스). 단위 목록의 N 이미지·클래스만으로 부트스트랩을 만든다."""
    if not keys:
        return {"n": 0, "n_anchors": 0, "estimate": None, "ci_low": None, "ci_high": None,
                "class_ci_low": None, "class_ci_high": None, "level": level}
    clusters = Clusters({key: meta[key] for key in keys}, reps, seed)
    row = clusters.summary(list(keys), list(values), level=level)
    return {**row, "n_anchors": len({meta[key][0] for key in keys}), "level": level}


def decide_rate(row, minimum):
    if row.get("estimate") is None or row.get("ci_low") is None:
        return LABELS["missing"]
    if row["estimate"] >= minimum and row["ci_low"] > 0:
        return LABELS["robust"] if row["class_ci_low"] is not None and row["class_ci_low"] > 0 else LABELS["support"]
    if row["ci_high"] < minimum:
        return LABELS["none"]
    return LABELS["other"]


def decide_joint(contrast, n_arm, minimum, direction):
    """B 판정: 대비 C 조건과 N 대상 Δm(N) 조건을 모두 충족해야 지지 (robust는 두 클래스 CI 모두)."""
    if contrast.get("estimate") is None or n_arm.get("estimate") is None:
        return LABELS["missing"], []
    if direction == "positive":
        c_ok = contrast["estimate"] >= minimum and contrast["ci_low"] > 0
        n_ok = n_arm["estimate"] > 0 and n_arm["ci_low"] > 0
        robust = contrast["class_ci_low"] > 0 and n_arm["class_ci_low"] > 0
        none = contrast["ci_high"] < minimum
    else:
        c_ok = contrast["estimate"] <= minimum and contrast["ci_high"] < 0
        n_ok = n_arm["estimate"] < 0 and n_arm["ci_high"] < 0
        robust = contrast["class_ci_high"] < 0 and n_arm["class_ci_high"] < 0
        none = contrast["ci_low"] > minimum
    unmet = [name for name, ok in (("contrast", c_ok), ("n_arm", n_ok)) if not ok]
    if not unmet:
        return (LABELS["robust"] if robust else LABELS["support"]), []
    if none:
        return LABELS["none"], unmet
    return LABELS["other"], unmet


# ================================================================ 실험 A

def a_index(plan, batches, n_layers):
    """단위·방향별 {"none": 행, ("same"|"c1", ℓ): 행} 과 All-P 짝."""
    table = defaultdict(dict)
    for batch in batches.values():
        key = (batch["unit_id"], batch["direction"])
        rows = table[key]
        for row in batch["rows"]:
            rows["none" if row["donor_kind"] == "none" else (row["donor_kind"], row["layer"])] = row
    units = {u["unit_id"]: u for u in plan["a_units"]}
    pairs = {p["pair_id"]: p for p in plan["a_allp"]}
    for unit in units.values():
        for direction in A_DIRECTIONS:
            rows = table[(unit["unit_id"], direction)]
            require("none" in rows and all(("same", l) in rows for l in range(n_layers + 1)),
                    f"Missing rows for {unit['unit_id']} {direction}")
            if unit[direction]["c1_donor"] is not None:
                require(all(("c1", l) in rows for l in range(n_layers + 1)), f"Missing C1 rows for {unit['unit_id']}")
    return table, units, pairs


def success(row, direction):
    return int(row["answer"] == A_TARGET[direction])


VALID = ("valid",)
WITH_C1_INCLUDING_CHANGED = ("valid", "changed_pair", "changed_none_row", "changed_c1")
SAME_SET_USABLE = ("valid", "no_c1", "changed_c1")


def changed_inputs(plan, rows_by_input):
    """새 계산 경로의 무개입 답이 기준 실행 답과 다른 입력."""
    return {i for i, row in rows_by_input.items() if row["answer"] != plan["inputs"][i]["baseline"]["prediction"]}


def a_status(plan, units, table, changed):
    status = {}
    for unit in units.values():
        for direction in A_DIRECTIONS:
            spec = unit[direction]
            baseline = plan["inputs"][spec["recipient"]]["baseline"]["prediction"]
            if unit["false_yes_input"] in changed or unit["correct_no_input"] in changed:
                reason = "changed_pair"
            elif table[(unit["unit_id"], direction)]["none"]["answer"] != baseline:
                reason = "changed_none_row"
            elif spec["c1_donor"] is None:
                reason = "no_c1"
            elif spec["c1_donor"] in changed:
                reason = "changed_c1"
            else:
                reason = "valid"
            status[(unit["unit_id"], direction)] = reason
    return status


def a_units_for(units, status, half, direction, allowed=VALID, keep=None):
    return [u for u in units.values() if u["half"] == half and status[(u["unit_id"], direction)] in allowed
            and (keep is None or keep(u))]


def a_select(units, table, status, direction, n_layers):
    chosen = a_units_for(units, status, "selection", direction)
    scores = {}
    for layer in range(1, n_layers):
        scores[layer] = sum(success(table[(u["unit_id"], direction)][("same", layer)], direction)
                            - success(table[(u["unit_id"], direction)][("c1", layer)], direction) for u in chosen)
    best = max(scores.values())
    layer = min(l for l, s in scores.items() if s == best)
    return {"layer": layer, "score": best, "n_units": len(chosen), "scores": scores}


def a_values(units, table, direction, layer, outcome="success"):
    keys, values = [], []
    for unit in units:
        rows = table[(unit["unit_id"], direction)]
        if outcome == "success":
            value = success(rows[("same", layer)], direction) - success(rows[("c1", layer)], direction)
        else:
            none = rows["none"][outcome]
            value = (rows[("same", layer)][outcome] - none) - (rows[("c1", layer)][outcome] - none)
        keys.append(unit["unit_id"])
        values.append(value)
    return keys, values


def a_meta(units):
    return {u["unit_id"]: (u["anchor"], u["class"]) for u in units.values()}


def analyze_a(plan, run, preregistration, n_layers, on_selection=None):
    statistics = preregistration["statistics"]
    reps, seed = statistics["reps"], statistics["seed"]
    table, units, pairs = a_index(plan, run["batches"], n_layers)
    meta = a_meta(units)
    hypotheses = preregistration["hypotheses"]
    changed = changed_inputs(plan, run["phase1"])
    status = a_status(plan, units, table, changed)

    # (1) 선택 (선택 절반만, 제외 규칙 적용)
    selection = {direction: a_select(units, table, status, direction, n_layers) for direction in A_DIRECTIONS}
    joint_scores = {l: selection["restore"]["scores"][l] + selection["contaminate"]["scores"][l]
                    for l in range(1, n_layers)}
    joint = min(l for l, s in joint_scores.items() if s == max(joint_scores.values()))
    selection_record = {direction: {k: v for k, v in s.items() if k != "scores"} for direction, s in selection.items()}
    selection_record["joint_descriptive"] = {"layer": joint, "score": joint_scores[joint]}
    selection_record["scores"] = {direction: selection[direction]["scores"] for direction in A_DIRECTIONS}
    if on_selection is not None:
        on_selection(selection_record)

    # (2) 확증 판정
    primary = []
    for name, direction in (("A-H1", "restore"), ("A-H2", "contaminate")):
        layer = selection[direction]["layer"]
        chosen = a_units_for(units, status, "confirmation", direction)
        keys, values = a_values(chosen, table, direction, layer)
        minimum = hypotheses[name]["minimum_effect"]
        excluded = {reason: sum(1 for u in units.values() if u["half"] == "confirmation"
                                and status[(u["unit_id"], direction)] == reason)
                    for reason in ("changed_pair", "changed_none_row", "no_c1", "changed_c1")}
        row = {"hypothesis": name, "direction": direction, "layer": layer, "outcome": "answer change rate difference",
               **summarize(meta, keys, values, PRIMARY_LEVEL, reps, seed), "minimum_effect": minimum,
               "same_set_rate": mean([success(table[(k, direction)][("same", layer)], direction) for k in keys]),
               "c1_rate": mean([success(table[(k, direction)][("c1", layer)], direction) for k in keys]),
               "excluded": excluded}
        row["label"] = decide_rate(row, minimum)
        primary.append(row)

    # (3) 부가 (95%, 판정 없음)
    secondary = []

    def add(label, direction, layer, chosen, outcome="success"):
        keys, values = a_values(chosen, table, direction, layer, outcome)
        secondary.append({"analysis": label, "direction": direction, "layer": layer, "outcome": outcome,
                          **summarize(meta, keys, values, SECONDARY_LEVEL, reps, seed)})

    for direction in A_DIRECTIONS:
        layer = selection[direction]["layer"]
        confirmation = a_units_for(units, status, "confirmation", direction)
        add("all units (95%)", direction, layer, confirmation)
        add("original definition: N-first only", direction, layer, [u for u in confirmation if u["position"] == "first"])
        add("middle positions only", direction, layer, [u for u in confirmation if u["position"] == "middle"])
        for n in (3, 4, 5):
            add(f"{n} images", direction, layer, [u for u in confirmation if u["n_images"] == n])
        add("margin contrast (fp32 variant)", direction, layer, confirmation, "margin")
        add("margin contrast (bf16)", direction, layer, confirmation, "margin_bf16")
        inputs = plan["inputs"]
        # 사전 등록: C1 donor가 있는 단위 전체. 사유 판정에서 '답 바뀜'이 'C1 없음'보다 앞서므로 C1 유무를 따로 본다 (v3)
        add("sensitivity: including units whose no-intervention answer changed", direction, layer,
            a_units_for(units, status, "confirmation", direction, WITH_C1_INCLUDING_CHANGED,
                        keep=lambda u: u[direction]["c1_donor"] is not None))
        add("sensitivity: baseline bf16 margin sign agrees with answer", direction, layer,
            [u for u in confirmation if sign_consistent(inputs[u[direction]["recipient"]])])
    side_effects = a_side_effects(pairs, run["batches"], selection, reps, seed, plan, changed)

    # 기술 통계
    descriptive = {"joint_layer": joint, "joint_confirmation": []}
    for direction in A_DIRECTIONS:
        keys, values = a_values(a_units_for(units, status, "confirmation", direction), table, direction, joint)
        descriptive["joint_confirmation"].append({"direction": direction, "layer": joint,
                                                  **summarize(meta, keys, values, SECONDARY_LEVEL, reps, seed)})
    curves, earliest = [], []
    for half in ("selection", "confirmation"):
        for direction in A_DIRECTIONS:
            chosen = a_units_for(units, status, half, direction)
            everyone = a_units_for(units, status, half, direction, SAME_SET_USABLE)
            for layer in range(n_layers + 1):
                same = mean([success(table[(u["unit_id"], direction)][("same", layer)], direction) for u in everyone])
                c1 = mean([success(table[(u["unit_id"], direction)][("c1", layer)], direction) for u in chosen])
                same_paired = mean([success(table[(u["unit_id"], direction)][("same", layer)], direction)
                                    for u in chosen])
                curves.append({"half": half, "direction": direction, "layer": layer, "same_set_rate": same,
                               "same_set_rate_with_c1": same_paired, "c1_rate": c1,
                               "difference": None if c1 is None else same_paired - c1,
                               "n_units": len(everyone), "n_units_with_c1": len(chosen)})
            if direction == "restore":
                for donor in ("same_set_rate", "c1_rate"):
                    reached = [c["layer"] for c in curves if c["half"] == half and c["direction"] == direction
                               and c["layer"] >= 1 and c[donor] is not None and c[donor] >= 0.5]
                    earliest.append({"half": half, "donor": donor, "earliest_layer_rate_ge_0.5":
                                     min(reached) if reached else "none"})
    descriptive.update(curves=curves, earliest_restore_ge_half=earliest)
    return {"selection": selection_record, "primary": primary, "secondary": secondary,
            "all_positive_side_effects": side_effects, "descriptive": descriptive,
            "counts": a_counts(units, status), "changed_inputs": sorted(changed)}


def a_side_effects(pairs, batches, selection, reps, seed, plan, changed):
    rows_out = []
    meta = {p["pair_id"]: (p["anchor"], p["class"]) for p in pairs.values()}
    for label, layer in (("l*_restore", selection["restore"]["layer"]),
                         ("l*_contaminate", selection["contaminate"]["layer"])):
        keys, flips, deltas = [], [], []
        for pair in pairs.values():
            if pair["half"] != "confirmation":
                continue
            for direction, (recipient, donor) in ALLP_DIRECTIONS.items():
                batch = batches[f"{pair['pair_id']}::{direction}"]
                none = batch["rows"][0]
                recipient_id = pair["inputs"][recipient]
                if (plan["inputs"][recipient_id]["baseline"]["prediction"] != "Yes" or none["answer"] != "Yes"
                        or recipient_id in changed or pair["inputs"][donor] in changed):
                    continue
                row = next(r for r in batch["rows"] if r["donor_kind"] == "same" and r["layer"] == layer)
                keys.append(pair["pair_id"])
                flips.append(int(row["answer"] == "No"))
                deltas.append(row["margin"] - none["margin"])
        for outcome, values in (("yes_to_no_rate", flips), ("delta_margin", deltas)):
            rows_out.append({"layer_rule": label, "layer": layer, "outcome": outcome,
                             "unit": "All-P recipient answering Yes in the baseline and without intervention",
                             **summarize_repeated(meta, keys, values, SECONDARY_LEVEL, reps, seed)})
    return rows_out


def summarize_repeated(meta, keys, values, level, reps, seed):
    """같은 짝이 두 방향으로 두 번 나올 수 있어 단위 키를 (짝, 순번)으로 바꿔 부트스트랩한다."""
    expanded = {}
    unit_keys = []
    for i, key in enumerate(keys):
        unit = f"{key}#{i}"
        expanded[unit] = meta[key]
        unit_keys.append(unit)
    return summarize(expanded, unit_keys, values, level, reps, seed)


def sign_consistent(entry):
    base = entry["baseline"]
    return (base["logit_margin"] > 0 and base["prediction"] == "Yes") or (base["logit_margin"] < 0
                                                                          and base["prediction"] == "No")


def a_counts(units, status):
    counts = defaultdict(int)
    for unit in units.values():
        counts[f"{unit['half']}_units"] += 1
        counts[f"{unit['half']}_{unit['position']}"] += 1
        for direction in A_DIRECTIONS:
            counts[f"{unit['half']}_{direction}_{status[(unit['unit_id'], direction)]}"] += 1
    return dict(sorted(counts.items()))


# ================================================================ 실험 B

def b_index(plan, batches):
    items = {item["item_id"]: item for item in plan["b_items"]}
    table = {}
    for batch in batches.values():
        rows = {"none": batch["rows"][0]}
        for row in batch["rows"][1:]:
            rows[(row["query"], row["kind"], row["target"], tuple(row["band"]))] = row
        table[batch["item_id"]] = rows
    require(set(table) == set(items), "B results do not cover the planned items")
    return table, items


def b_delta(rows, query, kind, target, band, outcome="margin"):
    return rows[(query, kind, target, band)][outcome] - rows["none"][outcome]


def b_valid(plan, table, items):
    """무개입 행의 답이 기준 실행 답과 같은 항목 (사전 등록 v2 제외 규칙)."""
    return {iid for iid, item in items.items()
            if table[iid]["none"]["answer"] == plan["inputs"][item["input_id"]]["baseline"]["prediction"]}


def b_population(items, half, population, valid, keep=None):
    """valid가 None이면 제외 규칙을 적용하지 않는다 (민감도 분석)."""
    return [i for i in items.values() if i["half"] == half and i["population"] == population
            and (valid is None or i["item_id"] in valid) and (keep is None or keep(i))]


def b_band_means(table, chosen, query, kind):
    means = {}
    for band in BANDS:
        values = [b_delta(table[i["item_id"]], query, kind, "N", band) - b_delta(table[i["item_id"]], query, kind, "P", band)
                  for i in chosen]
        means[band] = mean(values)
    return means


def b_select(table, items, population, valid, query="post_image"):
    chosen = b_population(items, "selection", population, valid)
    means = b_band_means(table, chosen, query, population)
    target = max(means.values()) if population == "block" else min(means.values())
    band = next(b for b in BANDS if abs(means[b] - target) <= TIE)
    return {"band": list(band), "mean_contrast": means[band], "n_items": len(chosen),
            "means": {f"{b[0]}-{b[1]}": m for b, m in means.items()}}


def b_arms(table, chosen, query, kind, band, outcome="margin"):
    keys, contrast, n_arm, p_arm = [], [], [], []
    for item in chosen:
        rows = table[item["item_id"]]
        n_delta = b_delta(rows, query, kind, "N", band, outcome)
        p_delta = b_delta(rows, query, kind, "P", band, outcome)
        keys.append(item["item_id"])
        contrast.append(n_delta - p_delta)
        n_arm.append(n_delta)
        p_arm.append(p_delta)
    return keys, contrast, n_arm, p_arm


def analyze_b(plan, run, preregistration, on_selection=None):
    statistics = preregistration["statistics"]
    reps, seed = statistics["reps"], statistics["seed"]
    table, items = b_index(plan, run["batches"])
    meta = {i["item_id"]: (i["anchor"], i["class"]) for i in items.values()}
    hypotheses = preregistration["hypotheses"]
    valid = b_valid(plan, table, items)

    # (1) 선택 (선택 절반만, 제외 규칙 적용)
    selection = {population: b_select(table, items, population, valid) for population in ("block", "boost")}
    selection["last_token_descriptive"] = {population: b_select(table, items, population, valid, "last_token")
                                           for population in ("block", "boost")}
    if on_selection is not None:
        on_selection(selection)

    # (2) 확증 판정
    primary = []
    for name, population, direction in (("B-H1", "block", "positive"), ("B-H2", "boost", "negative")):
        band = tuple(selection[population]["band"])
        chosen = b_population(items, "confirmation", population, valid)
        keys, contrast, n_arm, p_arm = b_arms(table, chosen, "post_image", population, band)
        minimum = hypotheses[name]["minimum_effect"]
        excluded = len(b_population(items, "confirmation", population, None)) - len(chosen)
        c_row = summarize(meta, keys, contrast, PRIMARY_LEVEL, reps, seed)
        n_row = summarize(meta, keys, n_arm, PRIMARY_LEVEL, reps, seed)
        p_row = summarize(meta, keys, p_arm, PRIMARY_LEVEL, reps, seed)
        label, unmet = decide_joint(c_row, n_row, minimum, direction)
        primary.append({"hypothesis": name, "population": population, "band": list(band), "direction": direction,
                        "minimum_effect": minimum, "label": label, "unmet_conditions": unmet,
                        "excluded_changed_no_intervention_answer": excluded,
                        "contrast": c_row, "n_arm": n_row,
                        "p_arm_reported": {**p_row, "sign": None if p_row["estimate"] is None else
                                           ("+" if p_row["estimate"] > 0 else "-" if p_row["estimate"] < 0 else "0")}})

    # (3) 부가 (95%)
    secondary = []

    def add(label, population, query, band, chosen, outcome="margin"):
        keys, contrast, n_arm, p_arm = b_arms(table, chosen, query, population, tuple(band), outcome)
        for arm, values in (("C", contrast), ("delta_N", n_arm), ("delta_P", p_arm)):
            secondary.append({"analysis": label, "population": population, "query": query, "band": list(band),
                              "outcome": outcome, "arm": arm,
                              **summarize(meta, keys, values, SECONDARY_LEVEL, reps, seed)})

    inputs = plan["inputs"]
    for population in ("block", "boost"):
        band = selection[population]["band"]
        own = selection["last_token_descriptive"][population]["band"]
        confirmation = b_population(items, "confirmation", population, valid)
        add("last token only, own selected band", population, "last_token", own, confirmation)
        add("last token only, primary band", population, "last_token", band, confirmation)
        add("primary, bf16 margin", population, "post_image", band, confirmation, "margin_bf16")
        for n in (3, 4, 5):
            add(f"{n} images", population, "post_image", band, [i for i in confirmation if i["n_images"] == n])
        for position in ("first", "middle", "last"):
            add(f"N position {position}", population, "post_image", band,
                [i for i in confirmation if i["position"] == position])
        baseline_answer = {"block": "No", "boost": "Yes"}[population]
        add("sensitivity: including inputs whose no-intervention answer changed", population, "post_image", band,
            b_population(items, "confirmation", population, None))
        add("sensitivity: baseline bf16 margin sign agrees with answer", population, "post_image", band,
            [i for i in confirmation if sign_consistent(inputs[i["input_id"]])])
        flipped = {"block": "Yes", "boost": "No"}[population]
        for target in ("N", "P"):
            keys = [i["item_id"] for i in confirmation]
            values = [int(table[k][("post_image", population, target, tuple(band))]["answer"] == flipped) for k in keys]
            secondary.append({"analysis": f"answer change rate ({baseline_answer} -> {flipped}), target {target}",
                              "population": population, "query": "post_image", "band": list(band),
                              "outcome": "answer_change_rate", "arm": target,
                              **summarize(meta, keys, values, SECONDARY_LEVEL, reps, seed)})
    side_effects, warnings = b_side_effects(table, items, selection, meta, reps, seed, valid)
    curves = []
    for half in ("selection", "confirmation"):
        for population in ("block", "boost"):
            chosen = b_population(items, half, population, valid)
            for query in ("post_image", "last_token"):
                for band in BANDS:
                    _, contrast, n_arm, p_arm = b_arms(table, chosen, query, population, band)
                    curves.append({"half": half, "population": population, "query": query, "band": f"{band[0]}-{band[1]}",
                                   "n": len(chosen), "mean_C": mean(contrast), "mean_delta_N": mean(n_arm),
                                   "mean_delta_P": mean(p_arm)})
    return {"selection": selection, "primary": primary, "secondary": secondary,
            "all_positive_side_effects": side_effects, "warnings": warnings, "descriptive": {"band_curves": curves},
            "counts": {f"{half}_{population}_{kind}": len(b_population(items, half, population, rule))
                       for half in ("selection", "confirmation") for population in ("block", "boost", "all_positive")
                       for kind, rule in (("planned", None), ("analyzed", valid))},
            "changed_inputs": sorted(items[i]["input_id"] for i in set(items) - valid)}


def b_side_effects(table, items, selection, meta, reps, seed, valid):
    rows, warnings = [], []
    chosen = b_population(items, "confirmation", "all_positive", valid)
    for kind in ("block", "boost"):
        selected = tuple(selection[kind]["band"])
        for band in BANDS:
            for target in ("N", "P"):
                keys, flips, deltas = [], [], []
                for item in chosen:
                    item_rows = table[item["item_id"]]
                    if item_rows["none"]["answer"] != "Yes":
                        continue
                    row = item_rows[("post_image", kind, target, band)]
                    keys.append(item["item_id"])
                    flips.append(int(row["answer"] == "No"))
                    deltas.append(row["margin"] - item_rows["none"]["margin"])
                level = SECONDARY_LEVEL
                flip = summarize(meta, keys, flips, level, reps, seed) if band == selected else {
                    "n": len(keys), "estimate": mean(flips)}
                rows.append({"kind": kind, "band": f"{band[0]}-{band[1]}", "selected_band": band == selected,
                             "target": "N slot" if target == "N" else "P control", "outcome": "yes_to_no_rate", **flip})
                rows.append({"kind": kind, "band": f"{band[0]}-{band[1]}", "selected_band": band == selected,
                             "target": "N slot" if target == "N" else "P control", "outcome": "delta_margin",
                             "n": len(keys), "estimate": mean(deltas)})
                if band == selected and flip["estimate"] is not None and flip["estimate"] > 0.05:
                    warnings.append(f"All-P Yes->No rate {flip['estimate']:.3f} > 0.05 ({kind}, band {band}, {target})")
    return rows, warnings
