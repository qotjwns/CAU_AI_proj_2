"""STEP 5-3 조건 목록 (CPU, 실행·검사·평가 공용).

실험 A batch = (단위, 방향): [무개입] + [같은 묶음 donor ℓ = 0..36] + [C1 donor ℓ = 0..36] (C1이 없으면 앞의 둘만).
실험 A All-P batch = (짝, 방향 to_first / to_last): [무개입] + [다른 순서 donor ℓ = 0..36].
실험 B batch = 항목: one-N은 [무개입] + 주 query(이미지 뒤 전체) × 구간 9 × 대상(N, P) + 부가 query(마지막 토큰만)
× 구간 9 × 대상(N, P), 개입 종류는 모집단(차단 = 기준 No, 강화 = 기준 Yes). All-P는 [무개입] + 주 query ×
(차단, 강화) × 구간 9 × (N 자리, P). 연습 실행에는 크기 0 개입 행이 하나 더 붙는다.
"""
import random

BANDS = tuple((start, start + 3) for start in range(1, 37, 4))
A_DIRECTIONS = ("restore", "contaminate")
A_TARGET = {"restore": "No", "contaminate": "Yes"}
ALLP_DIRECTIONS = {"to_first": ("N-first", "N-last"), "to_last": ("N-last", "N-first")}   # (받는 쪽, donor)
B_QUERIES = ("post_image", "last_token")
PILOT_REPRODUCTION_PER_COUNT = 40


def a_batches(units, allp_pairs, n_layers):
    """실험 A batch 목록. 행: {"donor_kind", "donor", "layer"}."""
    batches = []
    for unit in units:
        for direction in A_DIRECTIONS:
            plan = unit[direction]
            rows = [{"donor_kind": "none", "donor": None, "layer": None}]
            rows += [{"donor_kind": "same", "donor": plan["same_donor"], "layer": layer} for layer in range(n_layers + 1)]
            if plan["c1_donor"] is not None:
                rows += [{"donor_kind": "c1", "donor": plan["c1_donor"], "layer": layer}
                         for layer in range(n_layers + 1)]
            batches.append({"batch": f"{unit['unit_id']}::{direction}", "kind": "unit", "unit_id": unit["unit_id"],
                            "direction": direction, "recipient": plan["recipient"], "n_images": unit["n_images"],
                            "half": unit["half"], "rows": rows})
    for pair in allp_pairs:
        for direction, (recipient, donor) in ALLP_DIRECTIONS.items():
            rows = [{"donor_kind": "none", "donor": None, "layer": None}]
            rows += [{"donor_kind": "same", "donor": pair["inputs"][donor], "layer": layer}
                     for layer in range(n_layers + 1)]
            batches.append({"batch": f"{pair['pair_id']}::{direction}", "kind": "allp", "unit_id": pair["pair_id"],
                            "direction": direction, "recipient": pair["inputs"][recipient],
                            "n_images": pair["n_images"], "half": pair["half"], "rows": rows})
    return batches


def b_rows(item, pilot=False):
    """실험 B 항목의 행: {"query", "kind", "target", "band"}."""
    rows = [{"query": "none", "kind": "none", "target": None, "band": None}]
    if item["kind"] == "one_negative":
        kind = item["population"]
        for query in B_QUERIES:
            rows += [{"query": query, "kind": kind, "target": target, "band": list(band)}
                     for band in BANDS for target in ("N", "P")]
    else:
        rows += [{"query": "post_image", "kind": kind, "target": target, "band": list(band)}
                 for kind in ("block", "boost") for band in BANDS for target in ("N", "P")]
    if pilot:
        rows.append({"query": "post_image", "kind": "zero", "target": "N", "band": list(BANDS[0])})
    return rows


def pilot_a(units, allp_pairs, inputs):
    """연습 실행 A: 선택 절반에서 (장수, 위치)마다 첫 단위(양방향 C1 있음), 그 All-P 짝, 재현 표본."""
    chosen = {}
    for unit in sorted(units, key=lambda u: u["unit_id"]):
        key = (unit["n_images"], unit["position"])
        if (unit["half"] == "selection" and key not in chosen and unit["restore"]["c1_donor"]
                and unit["contaminate"]["c1_donor"]):
            chosen[key] = unit
    picked = [chosen[key] for key in sorted(chosen)]
    keys = {(u["series"], u["n_images"]) for u in picked}
    pairs = [p for p in allp_pairs if (p["series"], p["n_images"]) in keys]
    return picked, pairs, reproduction_sample(inputs)


def pilot_b(items, inputs):
    """연습 실행 B: 선택 절반에서 (장수, 위치, 모집단)마다 첫 one-N 항목, 장수마다 첫 All-P 항목, 재현 표본."""
    chosen = {}
    for item in sorted(items, key=lambda i: i["item_id"]):
        key = (item["kind"], item["n_images"], item["position"] if item["kind"] == "one_negative" else "slot",
               item["population"])
        if item["half"] == "selection" and key not in chosen:
            chosen[key] = item
    return [chosen[key] for key in sorted(chosen)], reproduction_sample(inputs)


def reproduction_sample(inputs, seed=53):
    """선택 절반 입력 중 장수별 40개 (seed 추첨): 무개입 빠른 경로가 기준 실행을 재현하는지 보는 표본."""
    rng = random.Random(seed)
    sample = []
    for n in (3, 4, 5):
        pool = sorted(i for i, e in inputs.items() if e["half"] == "selection" and e["case"]["n_images"] == n)
        sample += sorted(rng.sample(pool, min(PILOT_REPRODUCTION_PER_COUNT, len(pool))))
    return sample


def a_work(plan, stage, n_layers):
    if stage == "pilot":
        units, pairs, reproduction = pilot_a(plan["a_units"], plan["a_allp"], plan["inputs"])
    else:
        units, pairs, reproduction = plan["a_units"], plan["a_allp"], []
    batches = a_batches(units, pairs, n_layers)
    needed = {b["recipient"] for b in batches} | {r["donor"] for b in batches for r in b["rows"] if r["donor"]}
    return {"batches": batches, "phase1": sorted(needed | set(reproduction)), "reproduction": reproduction,
            "reference_batches": [b["batch"] for b in batches if b["kind"] == "unit"] if stage == "pilot" else []}


def b_work(plan, stage):
    if stage == "pilot":
        items, reproduction = pilot_b(plan["b_items"], plan["inputs"])
    else:
        items, reproduction = plan["b_items"], []
    batches = [{"batch": item["item_id"], "item_id": item["item_id"], "input_id": item["input_id"],
                "n_images": item["n_images"], "half": item["half"], "rows": b_rows(item, pilot=stage == "pilot")}
               for item in sorted(items, key=lambda i: i["item_id"])]
    return {"batches": batches, "items": {item["item_id"]: item for item in items}, "reproduction": reproduction,
            "reference_batches": [b["batch"] for b in batches] if stage == "pilot" else []}
