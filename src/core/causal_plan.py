"""STEP 5-3 실험 A·B 계획 파일 (CPU, 5-3 출력 전).

5-2b 본 추출(one-N 3~5장)과 5-2 본 추출(All-P)의 기준 답만으로 다음을 만든다. 5-3 출력은 쓰지 않는다.
- 분할: N 이미지 480장을 클래스마다 12 / 12로 선택 절반·확증 절반에 배정 (seed 후보 중 장수별 표본 균형이 가장 좋은 것)
- 실험 A 단위: 같은 계열·장수에서 N-last가 No(Correct-No)이고 끝이 아닌 위치(a < 마지막)가 Yes(False-Yes)인 짝.
  복원(받는 쪽 False-Yes, donor N-last)과 오염(받는 쪽 N-last, donor False-Yes)이 같은 단위를 쓴다.
  C1 donor: 같은 절반·클래스·장수·위치·기준 답, 다른 N 이미지, 전체 토큰 수가 가장 가까운 입력 (동률은 seed 추첨).
- 실험 A All-P: A 단위가 있는 (계열, 장수)의 5-2 All-P 두 순서
- 실험 B 항목: 5-2b one-N 전체 (차단 = 기준 No, 강화 = 기준 Yes)와 5-2 All-P 3~5장 중 기준 Yes. 대조 P 이미지 규칙 적용.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import json
import math
import random

from src.common.data import read_json, read_jsonl, write_json
from src.common.paths import project_path, project_relative_path
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file


N_IMAGES = (3, 4, 5)
HALVES = ("selection", "confirmation")
BASELINE_FIELDS = ("prediction", "raw_response", "yes_logit", "no_logit", "logit_margin", "variant_logit_margin",
                   "fp32_logit_margin", "fp32_variant_logit_margin", "input_token_count", "image_token_count",
                   "query_index", "image_spans")
PLAN_FILES = ("split", "inputs", "a_units", "a_allp", "b_items")


# ---------------------------------------------------------------- 기준 실행 읽기

def load_baseline(run_dir, source):
    """기준 실행 manifest → {case_id: 입력 항목}. 기준 값(답, margin, 토큰 구간)을 함께 담는다."""
    entries = {}
    for record in read_jsonl(project_path(run_dir) / "manifest.jsonl"):
        case = record["case"]
        require(case["case_id"] not in entries, f"Duplicate case {case['case_id']} in {run_dir}")
        baseline = {key: record[key] for key in BASELINE_FIELDS}
        baseline["image_spans"] = [{k: span[k] for k in ("start", "end", "token_count", "image_index", "image_id")}
                                   for span in record["image_spans"]]
        entries[case["case_id"]] = {"input_id": case["case_id"], "source": source, "case": case, "baseline": baseline}
    return entries


def answer(entry):
    return entry["baseline"]["prediction"]


def tokens(entry):
    return entry["baseline"]["input_token_count"]


# ---------------------------------------------------------------- 분할

def one_negative_index(onen):
    """(계열, 장수, a) → 입력 항목."""
    cells = {}
    for entry in onen.values():
        case = entry["case"]
        key = (case["pair_id"], case["n_images"], case["n_before"])
        require(key not in cells, f"Duplicate 5-2b cell {key}")
        cells[key] = entry
    return cells


def a_pairs(cells):
    """A 단위 후보: (계열, 장수, a) — a < 마지막 위치가 Yes이고 같은 계열·장수의 마지막 위치가 No."""
    pairs = []
    for (series, n, a), entry in sorted(cells.items()):
        last = cells[(series, n, n - 1)]
        if a < n - 1 and answer(entry) == "Yes" and answer(last) == "No":
            pairs.append((series, n, a))
    return pairs


def anchor_vectors(onen, cells):
    """N 이미지별 균형 점수 항목 수: (A 단위, B 차단, B 강화) × 장수 3·4·5."""
    vectors = defaultdict(lambda: [0] * 9)
    for series, n, a in a_pairs(cells):
        vectors[cells[(series, n, a)]["case"]["anchor_negative_id"]][N_IMAGES.index(n)] += 1
    for entry in onen.values():
        case, slot = entry["case"], N_IMAGES.index(entry["case"]["n_images"])
        if answer(entry) == "No":
            vectors[case["anchor_negative_id"]][3 + slot] += 1
        elif answer(entry) == "Yes":
            vectors[case["anchor_negative_id"]][6 + slot] += 1
    return vectors


def make_split(onen, seed, candidates):
    cells = one_negative_index(onen)
    by_class = defaultdict(set)
    for entry in onen.values():
        by_class[entry["case"]["class"]].add(entry["case"]["anchor_negative_id"])
    by_class = {name: sorted(anchors) for name, anchors in sorted(by_class.items())}
    require(all(len(anchors) % 2 == 0 for anchors in by_class.values()), "Each class needs an even number of N images")
    vectors = anchor_vectors(onen, cells)
    totals = [sum(vectors[a][i] for anchors in by_class.values() for a in anchors) for i in range(9)]
    require(all(totals), "A balance dimension has no units")
    rng = random.Random(seed)
    best = None
    for index in range(candidates):
        selection = []
        for name, anchors in by_class.items():
            pool = list(anchors)
            rng.shuffle(pool)
            selection += pool[:len(pool) // 2]
        chosen = [sum(vectors[a][i] for a in selection) for i in range(9)]
        score = sum((Fraction(abs(2 * c - t), t) for c, t in zip(chosen, totals)), Fraction(0))
        if best is None or score < best[0]:
            best = (score, index, sorted(selection), chosen)
    score, index, selection, chosen = best
    selected = set(selection)
    halves = {anchor: ("selection" if anchor in selected else "confirmation")
              for anchors in by_class.values() for anchor in anchors}
    names = [f"{kind}_{n}" for kind in ("a_units", "b_block", "b_boost") for n in N_IMAGES]
    return {
        "seed": seed, "candidates": candidates, "selected_candidate": index, "score": float(score),
        "score_exact": f"{score.numerator}/{score.denominator}",
        "rule": "smallest score; ties -> smallest candidate index (exact rational comparison)",
        "halves": {half: sorted(a for a, h in halves.items() if h == half) for half in HALVES},
        "by_class": {name: {half: sorted(a for a in anchors if halves[a] == half) for half in HALVES}
                     for name, anchors in by_class.items()},
        "balance": {name: {"selection": c, "confirmation": t - c} for name, c, t in zip(names, chosen, totals)},
    }


def half_lookup(split):
    return {anchor: half for half in HALVES for anchor in split["halves"][half]}


# ---------------------------------------------------------------- 실험 A

def closest(candidates, target_tokens, rng):
    """(input_id, 토큰 수) 후보 중 토큰 수가 가장 가까운 것. 동률은 seed 추첨 (input_id 정렬 후)."""
    if not candidates:
        return None
    best = min(abs(count - target_tokens) for _, count in candidates)
    tied = sorted(input_id for input_id, count in candidates if abs(count - target_tokens) == best)
    return tied[0] if len(tied) == 1 else rng.choice(tied)


def build_a_units(onen, split, seed):
    cells = one_negative_index(onen)
    halves = half_lookup(split)
    pool_last_no, pool_yes = defaultdict(list), defaultdict(list)
    for (series, n, a), entry in cells.items():
        case = entry["case"]
        half = halves[case["anchor_negative_id"]]
        if a == n - 1 and answer(entry) == "No":
            pool_last_no[(half, case["class"], n)].append(entry)
        if a < n - 1 and answer(entry) == "Yes":
            pool_yes[(half, case["class"], n, a)].append(entry)
    rng = random.Random(seed)
    units = []
    for series, n, a in a_pairs(cells):
        yes_entry, no_entry = cells[(series, n, a)], cells[(series, n, n - 1)]
        case = yes_entry["case"]
        anchor, half = case["anchor_negative_id"], halves[case["anchor_negative_id"]]
        restore_pool = [(e["input_id"], tokens(e)) for e in pool_last_no[(half, case["class"], n)]
                        if e["case"]["anchor_negative_id"] != anchor]
        contaminate_pool = [(e["input_id"], tokens(e)) for e in pool_yes[(half, case["class"], n, a)]
                            if e["case"]["anchor_negative_id"] != anchor]
        units.append({
            "unit_id": f"A::{series}::n{n}::a{a}", "series": series, "anchor": anchor, "class": case["class"],
            "strength": case["strength"], "n_images": n, "a": a, "position": "first" if a == 0 else "middle",
            "half": half, "false_yes_input": yes_entry["input_id"], "correct_no_input": no_entry["input_id"],
            "restore": {"recipient": yes_entry["input_id"], "same_donor": no_entry["input_id"],
                        "c1_donor": closest(restore_pool, tokens(yes_entry), rng)},
            "contaminate": {"recipient": no_entry["input_id"], "same_donor": yes_entry["input_id"],
                            "c1_donor": closest(contaminate_pool, tokens(no_entry), rng)},
        })
    return units


def build_a_allp(units, allp, split):
    """A 단위가 있는 (계열, 장수)마다 5-2 All-P 두 순서 (N 자리 첫 / 끝)."""
    halves = half_lookup(split)
    index = {}
    for entry in allp.values():
        case = entry["case"]
        index[(case["pair_id"], case["n_images"], case["placement"])] = entry
    pairs = []
    for series, n in sorted({(u["series"], u["n_images"]) for u in units}):
        first, last = index[(series, n, "N-first")], index[(series, n, "N-last")]
        case = first["case"]
        pairs.append({"pair_id": f"A-AllP::{series}::n{n}", "series": series, "anchor": case["anchor_negative_id"],
                      "class": case["class"], "strength": case["strength"], "n_images": n,
                      "half": halves[case["anchor_negative_id"]],
                      "inputs": {"N-first": first["input_id"], "N-last": last["input_id"]}})
    return pairs


# ---------------------------------------------------------------- 실험 B

def control_p(spans, labels, target):
    """대조 P: 이미지 토큰 수가 대상과 가장 가까운 P, 동률이면 위치가 가까운 것, 그래도 같으면 앞쪽."""
    reference = spans[target]["token_count"]
    candidates = [i for i, label in enumerate(labels) if label == "P" and i != target]
    require(candidates, "No P image for the control")
    return min(candidates, key=lambda i: (abs(spans[i]["token_count"] - reference), abs(i - target), i))


def b_item(entry, half, kind, population, target):
    case, baseline = entry["case"], entry["baseline"]
    spans = baseline["image_spans"]
    n = case["n_images"]
    position = "first" if target == 0 else ("last" if target == n - 1 else "middle")
    suffix_start = spans[-1]["end"] + 1
    return {
        "item_id": f"B::{entry['input_id']}", "input_id": entry["input_id"], "kind": kind, "population": population,
        "series": case["pair_id"], "anchor": case["anchor_negative_id"], "class": case["class"],
        "strength": case["strength"], "n_images": n, "target_index": target, "position": position, "half": half,
        "control_index": control_p(spans, case["image_labels"], target),
        "suffix_start": suffix_start, "last_index": baseline["query_index"],
    }


def build_b_items(onen, allp, split):
    halves = half_lookup(split)
    items = []
    for entry in sorted(onen.values(), key=lambda e: e["input_id"]):
        case = entry["case"]
        population = {"No": "block", "Yes": "boost"}.get(answer(entry))
        require(population is not None, f"{entry['input_id']}: unparsed baseline answer")
        items.append(b_item(entry, halves[case["anchor_negative_id"]], "one_negative", population, case["n_before"]))
    for entry in sorted(allp.values(), key=lambda e: e["input_id"]):
        case = entry["case"]
        if case["n_images"] not in N_IMAGES or answer(entry) != "Yes":
            continue
        slot = 0 if case["placement"] == "N-first" else case["n_images"] - 1
        items.append(b_item(entry, halves[case["anchor_negative_id"]], "all_positive", "all_positive", slot))
    return items


# ---------------------------------------------------------------- 검사와 저장

def check_plan(split, onen, allp, units, allp_pairs, items, inputs):
    checks = {}

    def record(name, passed, detail=None):
        checks[name] = {"passed": bool(passed), "detail": detail}

    halves = half_lookup(split)
    per_class = {name: (len(v["selection"]), len(v["confirmation"])) for name, v in split["by_class"].items()}
    record("split_480_classes_12_12", len(halves) == 480 and all(v == (12, 12) for v in per_class.values()),
           {"anchors": len(halves), "classes": len(per_class)})
    record("split_disjoint", not set(split["halves"]["selection"]) & set(split["halves"]["confirmation"]))
    cells = one_negative_index(onen)
    record("one_negative_grid_complete", len(cells) == 11520 and all(
        (s, n, a) in cells for (s, n, _) in cells for a in range(n)), {"records": len(cells)})
    bad_units = []
    for unit in units:
        yes_entry, no_entry = inputs[unit["false_yes_input"]], inputs[unit["correct_no_input"]]
        ok = (answer(yes_entry) == "Yes" and answer(no_entry) == "No"
              and yes_entry["case"]["pair_id"] == no_entry["case"]["pair_id"] == unit["series"]
              and yes_entry["case"]["n_images"] == no_entry["case"]["n_images"] == unit["n_images"]
              and yes_entry["case"]["n_before"] == unit["a"] < unit["n_images"] - 1
              and no_entry["case"]["n_before"] == unit["n_images"] - 1
              and sorted(yes_entry["case"]["image_ids"]) == sorted(no_entry["case"]["image_ids"]))
        for direction, same_answer in (("restore", "No"), ("contaminate", "Yes")):
            plan = unit[direction]
            c1 = plan["c1_donor"]
            if c1 is None:
                continue
            donor, same = inputs[c1], inputs[plan["same_donor"]]
            ok &= (answer(donor) == same_answer and donor["case"]["class"] == unit["class"]
                   and donor["case"]["n_images"] == unit["n_images"]
                   and donor["case"]["n_before"] == same["case"]["n_before"]
                   and donor["case"]["anchor_negative_id"] != unit["anchor"]
                   and halves[donor["case"]["anchor_negative_id"]] == unit["half"]
                   and donor["case"]["prompt"] == same["case"]["prompt"])
        if not ok:
            bad_units.append(unit["unit_id"])
    record("a_units_valid", not bad_units, {"invalid": bad_units[:5]})
    record("a_units_count", len(units) == 1186, {"units": len(units)})
    missing = {d: Counter(u["half"] for u in units if u[d]["c1_donor"] is None) for d in ("restore", "contaminate")}
    record("a_c1_missing_reported", True, {d: dict(v) for d, v in missing.items()})
    record("a_allp_pairs", all(inputs[p["inputs"]["N-first"]]["case"]["condition"] == "all_positive"
                               and inputs[p["inputs"]["N-last"]]["case"]["condition"] == "all_positive"
                               for p in allp_pairs) and len(allp_pairs) == len({(u["series"], u["n_images"])
                                                                                for u in units}),
           {"pairs": len(allp_pairs)})
    bad_items = []
    for item in items:
        entry = inputs[item["input_id"]]
        labels, spans = entry["case"]["image_labels"], entry["baseline"]["image_spans"]
        expected = "N" if item["kind"] == "one_negative" else "P"
        ok = (labels[item["target_index"]] == expected and labels[item["control_index"]] == "P"
              and item["control_index"] != item["target_index"] and len(spans) == item["n_images"]
              and 20 <= item["last_index"] - item["suffix_start"] + 1 <= 40
              and spans[-1]["end"] < item["suffix_start"])
        if item["kind"] == "one_negative":
            ok &= item["population"] == {"No": "block", "Yes": "boost"}[answer(entry)]
        else:
            ok &= answer(entry) == "Yes"
        if not ok:
            bad_items.append(item["item_id"])
    populations = Counter(item["population"] for item in items)
    record("b_items_valid", not bad_items, {"invalid": bad_items[:5]})
    record("b_items_count", populations == Counter({"block": 6499, "boost": 5021, "all_positive": 3994}),
           dict(populations))
    token_counts = defaultdict(set)
    for entry in inputs.values():
        for span in entry["baseline"]["image_spans"]:
            token_counts[span["image_id"]].add(span["token_count"])
    record("image_token_counts_consistent", all(len(v) == 1 for v in token_counts.values()),
           {"images": len(token_counts)})
    suffix = Counter(item["last_index"] - item["suffix_start"] + 1 for item in items)
    record("post_image_suffix_lengths", True, dict(sorted(suffix.items())))
    return checks


def build_plan(split_config, patching_config):
    """분할·A·B 계획을 만들고 data/step5_causal/에 쓴다. 이미 있으면 덮어쓰지 않는다."""
    output_dir = project_path(split_config["output_dir"])
    paths = {name: output_dir / filename for name, filename in split_config["files"].items()}
    require(not any(path.exists() for path in paths.values()), f"Plan files already exist in {output_dir}")
    onen = load_baseline(patching_config["baseline"]["one_negative_run"], "5-2b")
    every_52 = load_baseline(patching_config["baseline"]["all_positive_run"], "5-2")
    allp = {k: v for k, v in every_52.items() if v["case"]["condition"] == "all_positive"}
    split = make_split(onen, split_config["seed"], split_config["balance"]["candidates"])
    units = build_a_units(onen, split, split_config["seed"])
    allp_pairs = build_a_allp(units, allp, split)
    items = build_b_items(onen, allp, split)
    used = set(onen) | {p["inputs"][k] for p in allp_pairs for k in ("N-first", "N-last")}
    used |= {item["input_id"] for item in items}
    halves = half_lookup(split)
    inputs = {}
    for input_id in sorted(used):
        entry = onen.get(input_id) or allp[input_id]
        inputs[input_id] = {**entry, "half": halves[entry["case"]["anchor_negative_id"]]}
    checks = check_plan(split, onen, allp, units, allp_pairs, items, inputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(split, paths["split"], allow_nan=False)
    for name, rows in (("inputs", inputs.values()), ("a_units", units), ("a_allp", allp_pairs), ("b_items", items)):
        with paths[name].open("x", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    failed = [name for name, check in checks.items() if not check["passed"]]
    report = {
        "status": "passed" if not failed else "failed", "failed": failed, "checks": checks,
        "note": "Built from baseline answers only (5-2b and 5-2 main runs); no 5-3 output existed.",
        "baseline_sha256": {project_relative_path(project_path(run) / name): sha256_file(project_path(run) / name)
                            for run in (patching_config["baseline"]["one_negative_run"],
                                        patching_config["baseline"]["all_positive_run"])
                            for name in ("manifest.jsonl", "run.json")},
        "files_sha256": {project_relative_path(path): sha256_file(path) for path in paths.values()},
        "counts": {"inputs": len(inputs), "a_units": len(units), "a_allp_pairs": len(allp_pairs), "b_items": len(items)},
    }
    write_json(report, output_dir / split_config["check_file"], allow_nan=False)
    return report


def load_plan(split_config):
    """계획 파일과 검사 결과를 읽는다 (검사 통과·해시 일치 확인)."""
    output_dir = project_path(split_config["output_dir"])
    check = read_json(output_dir / split_config["check_file"])
    require(check["status"] == "passed", "The STEP 5-3 plan check has not passed")
    paths = {name: output_dir / filename for name, filename in split_config["files"].items()}
    for path in paths.values():
        require(check["files_sha256"][project_relative_path(path)] == sha256_file(path), f"Plan file changed: {path}")
    return {
        "check": check, "split": read_json(paths["split"]),
        "inputs": {row["input_id"]: row for row in read_jsonl(paths["inputs"])},
        "a_units": list(read_jsonl(paths["a_units"])), "a_allp": list(read_jsonl(paths["a_allp"])),
        "b_items": list(read_jsonl(paths["b_items"])),
    }


# ---------------------------------------------------------------- freeze 전 표본 수·예상 CI 반폭 보고

Z_975 = 2.241403            # 양측 97.5% 정규 분위수


def cluster_sizes(keys):
    """군집별 단위 수 → (군집 수, 가중 평균 크기 Σm²/Σm)."""
    counts = Counter(keys)
    sizes = list(counts.values())
    return len(sizes), (sum(m * m for m in sizes) / sum(sizes) if sizes else 0.0)


def half_widths(n, weighted_size, sd, rhos):
    """설계 효과 1 + (m − 1)ρ를 적용한 평균의 97.5% 정규 근사 반폭."""
    return {f"rho_{rho}": Z_975 * sd * math.sqrt((1 + (weighted_size - 1) * rho) / n) if n else None for rho in rhos}


def sample_report(plan, variances=(0.30, 0.48), sds=(1.21, 2.0), rhos=(0.0, 0.3, 1.0), class_rhos=(0.05, 0.1)):
    units, items = plan["a_units"], plan["b_items"]
    report = {"note": ("Sample sizes on the actual split and projected 97.5% CI half-widths (normal approximation with "
                       "a design effect; scenario values, not data). A: SD of the paired difference = sqrt(variance). "
                       "B: SD of C or Delta m(N) proxied by the baseline margin SD 1.21 (and 2.0)."),
              "A": {}, "B": {}}
    for half in HALVES:
        chosen = [u for u in units if u["half"] == half]
        entry = {"units": len(chosen), "by_n": dict(Counter(u["n_images"] for u in chosen)),
                 "by_position": dict(Counter(u["position"] for u in chosen)),
                 "n_images_with_units": len({u["anchor"] for u in chosen}),
                 "classes_with_units": len({u["class"] for u in chosen}), "directions": {}}
        for direction in ("restore", "contaminate"):
            usable = [u for u in chosen if u[direction]["c1_donor"] is not None]
            anchors, weighted = cluster_sizes([u["anchor"] for u in usable])
            classes, class_weighted = cluster_sizes([u["class"] for u in usable])
            entry["directions"][direction] = {
                "units_with_c1": len(usable), "without_c1": len(chosen) - len(usable),
                "n_images": anchors, "units_per_n_image_weighted": weighted, "classes": classes,
                "units_per_class_weighted": class_weighted,
                "half_width_n_image": {f"var_{v}": half_widths(len(usable), weighted, math.sqrt(v), rhos)
                                       for v in variances},
                "half_width_class": {f"var_{v}": half_widths(len(usable), class_weighted, math.sqrt(v), class_rhos)
                                     for v in variances}}
        report["A"][half] = entry
    for half in HALVES:
        entry = {}
        for population in ("block", "boost", "all_positive"):
            chosen = [i for i in items if i["half"] == half and i["population"] == population]
            anchors, weighted = cluster_sizes([i["anchor"] for i in chosen])
            classes, class_weighted = cluster_sizes([i["class"] for i in chosen])
            entry[population] = {
                "items": len(chosen), "by_n": dict(Counter(i["n_images"] for i in chosen)),
                "by_position": dict(Counter(i["position"] for i in chosen)), "n_images": anchors,
                "items_per_n_image_weighted": weighted, "classes": classes,
                "half_width_n_image": {f"sd_{s}": half_widths(len(chosen), weighted, s, rhos) for s in sds}
                if population != "all_positive" else None,
                "half_width_class": {f"sd_{s}": half_widths(len(chosen), class_weighted, s, class_rhos) for s in sds}
                if population != "all_positive" else None}
        report["B"][half] = entry
    return report
