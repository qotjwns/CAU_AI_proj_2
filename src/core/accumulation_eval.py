"""STEP 5-2 평가 (CPU, confirmatory).

사전 등록 v2(고정 manifest)의 가설·결과 변수·대비·판정 규칙을 그대로 적용한다. 판정에 쓰는 값(최소 효과, 판정
규칙, 부트스트랩 반복·seed)은 설정 파일이 아니라 고정 manifest에서 읽는다.

- 계열(series): N 이미지 1장 × 강도(High/Low)의 P 5장 순서. 배치(N-first/N-last)마다 one-N 입력 4개(P = 1..4)와
  같은 장수의 All-P 입력 4개가 있다.
- 대비는 같은 계열(H2는 같은 N 이미지) 안에서 짝지어 계산한 뒤 평균한다. High·Low 계열은 같은 가중치.
- 신뢰구간: N 이미지 ID 군집 부트스트랩(그 N의 계열·입력이 함께 뽑힘), 견고성 표기용 클래스 군집 부트스트랩.
- 민감도 분석의 제외: 대비에 쓰이는 입력 중 하나라도 제외 대상이면 그 대비 단위(짝) 전체를 뺀다.
"""
from collections import Counter, defaultdict
import math

import torch
from tqdm import tqdm

from src.common.config import read_yaml
from src.common.data import read_json, read_jsonl
from src.common.paths import project_path, project_relative_path
from src.core.accumulation_freeze import verify_accumulation_freeze
from src.core.feature_validation import inspect_run, load_validated_feature, require
from src.core.probe_cases import ImageBootstrap
from src.core.probe_freeze import sha256_file, verify_frozen


PLACEMENTS = ("N-first", "N-last")
STRENGTHS = ("High", "Low")
STEPS = (1, 2, 3, 4)
CONDITIONS = ("one_negative", "all_positive")
SAME_MODEL_KEYS = ("image_processor", "model_config", "chat_template", "dtype", "attention_implementation", "pooling")
SUPPORTED = ("robust support", "support (cross-class generalization uncertain)")
OUTCOME_NAMES = {"margin": "margin_fp32_variant", "margin_bf16": "margin_bf16"}
DEFINITIONS = {
    "series": "one N image x one strength (High/Low): five P images in a fixed order; P = k uses the first k",
    "P4_minus_P1": "one-N input with 4 P images minus the one with 1 P image, same series and placement; mean over series",
    "all_P_5_minus_2_images": "All-P control: 5-image minus 2-image All-P input of the same series and placement",
    "slope_P1_to_P4": "least-squares slope over P = 1..4 within a series (x centred at 2.5), mean over series",
    "difference_in_differences": "(one-N margin P4 - P1) - (All-P margin 5 - 2 images), within series and placement",
    "H2_high_minus_low": "High minus Low one-N input of the same N image, placement and P count; mean over N images",
    "H2_slope_of_difference": "least-squares slope over P = 1..4 of (High - Low) within an N image",
    "series_weighting": "High and Low series weigh equally; every N image contributes one series of each strength",
    "false_yes": "one-N input with parsed answer Yes; unparsed answers count as not Yes and are reported separately",
    "margin_fp32_variant": "fp32_variant_logit_margin (primary margin, preregistered)",
    "margin_bf16": "logit_margin: bf16 generation logits of 'Yes' and 'No' (sensitivity)",
    "bootstrap": "95% percentile cluster bootstrap; replicate estimate = draw-count-weighted mean over contrast units",
    "exclusion_rule": "a contrast unit is dropped if any of its inputs is excluded (pairs stay intact)",
    "sign_consistency": ("an input is sign-consistent if the bf16 margin is > 0 with answer Yes or < 0 with answer No; "
                         "ties and unparsed answers are excluded"),
}
EXPLORATORY_NOTE = (
    "Exploratory, not part of any decision. z_N: frozen STEP 4 z_i probe (per class) on the N image's pooled features "
    "in one-N inputs. h_q: frozen STEP 4 readouts for positions 1-3, trained on 3-image inputs; inputs with 2, 4 or 5 "
    "images are off-distribution, so h_q scores are descriptive only.")


# ---------------------------------------------------------------- 입력 표와 계열 설계

def outcome_table(records):
    """레코드 → 입력 행 (계열·배치·단계·조건과 결과 변수)."""
    rows = []
    for item in records:
        case = item["case"]
        prediction, bf16, primary = item["prediction"], item["logit_margin"], item["fp32_variant_logit_margin"]
        require(math.isfinite(bf16) and math.isfinite(primary), f"{case['case_id']}: non-finite margin")
        rows.append({
            "case_id": case["case_id"], "series": case["pair_id"], "anchor": case["anchor_negative_id"],
            "class": case["class"], "strength": case["strength"], "placement": case["placement"],
            "step": case["series_step"], "n_images": case["n_images"], "condition": case["condition"],
            "prediction": prediction, "yes": int(prediction == "Yes"), "unparsed": prediction not in ("Yes", "No"),
            "margin": float(primary), "margin_bf16": float(bf16),
            "sign_consistent": (bf16 > 0 and prediction == "Yes") or (bf16 < 0 and prediction == "No"),
        })
    return rows


def build_design(rows, conditions=CONDITIONS):
    """cells[(condition, series, placement, step)] = 입력 행. 빠진 칸·중복·한 N의 High/Low 누락이 있으면 실패."""
    cells, series = {}, {}
    for row in rows:
        key = (row["condition"], row["series"], row["placement"], row["step"])
        require(key not in cells, f"Duplicate input for {key}")
        cells[key] = row
        meta = {"anchor": row["anchor"], "class": row["class"], "strength": row["strength"]}
        require(series.setdefault(row["series"], meta) == meta, f"Inconsistent metadata in series {row['series']}")
    missing = [(c, s, p, k) for s in series for c in conditions for p in PLACEMENTS for k in STEPS
               if (c, s, p, k) not in cells]
    require(not missing, f"{len(missing)} design cells are missing, e.g. {missing[:3]}")
    require(len(cells) == len(series) * len(conditions) * len(PLACEMENTS) * len(STEPS), "Unexpected extra inputs")
    anchors = defaultdict(dict)
    for name, meta in series.items():
        require(meta["strength"] not in anchors[meta["anchor"]],
                f"N image {meta['anchor']} has two {meta['strength']} series")
        anchors[meta["anchor"]][meta["strength"]] = name
    incomplete = sorted(anchor for anchor, by_strength in anchors.items() if set(by_strength) != set(STRENGTHS))
    require(not incomplete, f"N images without both High and Low series: {incomplete[:5]}")
    return {"cells": cells, "series": series, "anchors": dict(anchors)}


def subset_design(design, names):
    """같은 셀 표에서 일부 계열만 쓰는 설계 (비전 fp32 부분집합과 짝짓기용)."""
    series = {name: design["series"][name] for name in names}
    anchors = defaultdict(dict)
    for name, meta in series.items():
        anchors[meta["anchor"]][meta["strength"]] = name
    return {"cells": design["cells"], "series": series, "anchors": dict(anchors)}


# ---------------------------------------------------------------- 대비 단위

def change(values):
    """마지막 − 처음 (P4 − P1, 또는 Pk − P1)."""
    return values[-1] - values[0]


def slope(values):
    """P = 1..4 최소제곱 기울기: Σ(k − 2.5)·y / Σ(k − 2.5)² (= 5)."""
    return math.fsum((k - 2.5) * value for k, value in zip(STEPS, values)) / 5


def first(values):
    return values[0]


def field(name):
    return lambda row: row[name]


def series_units(design, condition, placement, value, steps, statistic, keep=None):
    """계열마다 하나의 단위: statistic(steps 단계의 값들). keep(meta)로 계열을 거를 수 있다."""
    units = []
    for name, meta in sorted(design["series"].items()):
        if keep is not None and not keep(meta):
            continue
        rows = [design["cells"][(condition, name, placement, k)] for k in steps]
        units.append({"anchor": meta["anchor"], "class": meta["class"],
                      "value": statistic([value(row) for row in rows]), "members": tuple(r["case_id"] for r in rows)})
    return units


def did_units(design, placement, value):
    units = []
    for name, meta in sorted(design["series"].items()):
        one = [design["cells"][("one_negative", name, placement, k)] for k in (1, 4)]
        allp = [design["cells"][("all_positive", name, placement, k)] for k in (1, 4)]
        units.append({"anchor": meta["anchor"], "class": meta["class"],
                      "value": (value(one[1]) - value(one[0])) - (value(allp[1]) - value(allp[0])),
                      "members": tuple(row["case_id"] for row in one + allp)})
    return units


def h2_units(design, placement, value, statistic):
    """같은 N 이미지의 High − Low (one-N). statistic: P 장수(1..4), 'mean'(1..4 평균) 또는 'slope'."""
    units = []
    for anchor, by_strength in sorted(design["anchors"].items()):
        high, low = ([design["cells"][("one_negative", by_strength[s], placement, k)] for k in STEPS] for s in STRENGTHS)
        differences = [value(h) - value(l) for h, l in zip(high, low)]
        if statistic == "mean":
            result, members = math.fsum(differences) / len(differences), high + low
        elif statistic == "slope":
            result, members = slope(differences), high + low
        else:
            index = STEPS.index(statistic)
            result, members = differences[index], [high[index], low[index]]
        units.append({"anchor": anchor, "class": design["series"][by_strength["High"]]["class"], "value": result,
                      "members": tuple(row["case_id"] for row in members)})
    return units


# ---------------------------------------------------------------- 요약과 판정

def cluster_interval(bootstrap, keys, values):
    """군집 추출 횟수로 가중한 평균의 95% 백분위 구간. 군집이 하나도 뽑히지 않은 반복은 버린다."""
    weights = bootstrap.unit_weights(keys)
    totals = weights.sum(dim=1)
    valid = totals > 0
    if not bool(valid.any()):
        return None, None, 0
    estimates = weights[valid] @ torch.tensor(values, dtype=torch.float64) / totals[valid]
    low, high = torch.quantile(estimates, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(low), float(high), int(valid.sum())


def summarize(units, bootstraps, exclude=frozenset()):
    """bootstraps: {열 접두어: (ImageBootstrap, 'anchor' | 'class')}."""
    kept = [unit for unit in units if not exclude.intersection(unit["members"])]
    values = [unit["value"] for unit in kept]
    row = {"n_units": len(kept), "excluded_units": len(units) - len(kept),
           "n_anchors": len({unit["anchor"] for unit in kept}), "n_classes": len({unit["class"] for unit in kept}),
           "estimate": math.fsum(values) / len(values) if values else None}
    for prefix, (bootstrap, key) in bootstraps.items():
        low, high, valid = cluster_interval(bootstrap, [unit[key] for unit in kept], values) if values else (None, None, 0)
        row.update({f"{prefix}ci_low": low, f"{prefix}ci_high": high, f"{prefix}ci_valid": valid})
    return row


def outcome_name(outcome, condition="one_negative"):
    if outcome == "yes":
        return "false_yes_rate" if condition == "one_negative" else "yes_rate"
    return OUTCOME_NAMES[outcome]


def decide(row, minimum):
    """사전 등록 판정 규칙 (배치별)."""
    if row["estimate"] is None or row["ci_low"] is None:
        return "not estimable"
    if row["estimate"] >= minimum and row["ci_low"] > 0:
        robust = row.get("class_ci_low") is not None and row["class_ci_low"] > 0
        return SUPPORTED[0] if robust else SUPPORTED[1]
    if row["ci_high"] < minimum:
        return "no meaningful increase"
    return "inconclusive"


def overall_verdict(labels, decision):
    """두 배치의 판정을 H1 결론으로 묶는다 (사전 등록 decision 문구를 함께 기록)."""
    supported = [placement for placement in PLACEMENTS if labels[placement] in SUPPORTED]
    if len(supported) == len(PLACEMENTS):
        return {"verdict": "H1 supported (both placements)", "rule": decision["H1_supported"]}
    if supported:
        return {"verdict": f"supported in {supported[0]} only", "rule": decision["one_placement_only"]}
    if all(labels[placement] == "no meaningful increase" for placement in PLACEMENTS):
        return {"verdict": "no meaningful increase in either placement", "rule": decision["no_increase"]}
    return {"verdict": "H1 not supported (inconclusive in at least one placement)", "rule": decision["otherwise"]}


def thresholds(preregistration):
    effect = preregistration["minimum_effect"]
    return {"false_yes_rate": effect["false_yes_rate"], "margin_fp32_variant": effect["margin"],
            "margin_bf16": effect["margin"]}


# ---------------------------------------------------------------- 분석

def contrasts(design, bootstraps, outcomes, exclude=frozenset()):
    """주 대비(P4 − P1), All-P 대조, 기울기, 차이의 차이(margin), H2(장수별·평균·기울기)."""
    rows = []

    def add(analysis, condition, placement, outcome, units, **labels):
        rows.append({"analysis": analysis, "condition": condition, "placement": placement,
                     "outcome": outcome_name(outcome, condition), "p_count": labels.get("p_count", ""),
                     **summarize(units, bootstraps, exclude)})

    for placement in PLACEMENTS:
        for outcome in outcomes:
            value = field(outcome)
            for condition in CONDITIONS:
                name = "P4_minus_P1" if condition == "one_negative" else "all_P_5_minus_2_images"
                add(name, condition, placement, outcome,
                    series_units(design, condition, placement, value, (1, 4), change))
                add("slope_P1_to_P4", condition, placement, outcome,
                    series_units(design, condition, placement, value, STEPS, slope))
            if outcome != "yes":
                add("difference_in_differences", "one_negative_minus_all_positive", placement, outcome,
                    did_units(design, placement, value))
            for k in STEPS:
                add("H2_high_minus_low", "one_negative", placement, outcome, h2_units(design, placement, value, k),
                    p_count=k)
            add("H2_high_minus_low", "one_negative", placement, outcome, h2_units(design, placement, value, "mean"),
                p_count="mean_1_to_4")
            add("H2_slope_of_difference", "one_negative", placement, outcome,
                h2_units(design, placement, value, "slope"))
    return rows


def levels(design, bootstraps, outcomes):
    """조건 × 배치 × 강도(합침/High/Low) × 단계별 평균 (그림·표용, 판정에 쓰지 않음)."""
    rows = []
    for condition in CONDITIONS:
        for placement in PLACEMENTS:
            for outcome in outcomes:
                for strength in ("pooled", *STRENGTHS):
                    keep = None if strength == "pooled" else (lambda meta, s=strength: meta["strength"] == s)
                    for k in STEPS:
                        units = series_units(design, condition, placement, field(outcome), (k,), first, keep)
                        rows.append({"condition": condition, "placement": placement,
                                     "outcome": outcome_name(outcome, condition), "strength": strength,
                                     "series_step": k, "n_images": k + 1,
                                     "p_count": k if condition == "one_negative" else k + 1,
                                     **summarize(units, bootstraps)})
    return rows


def label_rows(rows, preregistration):
    """주 대비 행에 같은 판정 규칙의 표기를 붙인다 (민감도 분석에서는 참고용)."""
    limits = thresholds(preregistration)
    for row in rows:
        if row["analysis"] == "P4_minus_P1" and row["outcome"] in limits:
            row["minimum_effect"] = limits[row["outcome"]]
            row["label"] = decide(row, limits[row["outcome"]])
    return rows


def sensitivity(design, bootstraps, rows, exceeding, preregistration):
    inconsistent = frozenset(row["case_id"] for row in rows if not row["sign_consistent"])
    variants = (
        ("bf16_margin", ("margin_bf16",), frozenset(),
         "Primary, trend, difference-in-differences and H2 margin contrasts with the bf16 'Yes'/'No' margin."),
        ("sign_consistent", ("yes", "margin"), inconsistent,
         "Contrasts without inputs whose bf16 margin sign disagrees with the answer (ties and unparsed included)."),
        ("integrity_exclusion", ("yes", "margin"), frozenset(exceeding),
         "Contrasts without N-first inputs above the cosine-distance criterion, with the pairs they belong to."),
    )
    output, summary = [], []
    for name, outcomes, exclude, description in variants:
        result = label_rows(contrasts(design, bootstraps, outcomes, exclude), preregistration)
        output += [{"variant": name, **row} for row in result]
        one_negative = [row for row in rows if row["condition"] == "one_negative"]
        summary.append({"variant": name, "description": description, "excluded_inputs": len(exclude),
                        "excluded_one_negative_inputs": sum(row["case_id"] in exclude for row in one_negative)})
    return output, summary


def vision_comparison(design, vision_rows, statistics):
    """비전 fp32 부분집합: 같은 계열의 P4 − P1을 본 실행(bf16 비전)과 비교하고 짝 차이를 낸다."""
    vision = build_design(vision_rows, conditions=("one_negative",))
    for key, row in vision["cells"].items():
        require(design["cells"][key]["case_id"] == row["case_id"], f"Vision subset input differs from main run: {key}")
    main = subset_design(design, vision["series"])
    bootstrap = ImageBootstrap(sorted(vision["anchors"]), statistics["reps"], statistics["seed"])
    bootstraps = {"": (bootstrap, "anchor")}
    rows = []
    for placement in PLACEMENTS:
        for outcome in ("yes", "margin"):
            value = field(outcome)
            before = series_units(main, "one_negative", placement, value, (1, 4), change)
            after = series_units(vision, "one_negative", placement, value, (1, 4), change)
            require([unit["members"] for unit in before] == [unit["members"] for unit in after], "Series order differs")
            paired = [{**unit, "value": other["value"] - unit["value"]} for unit, other in zip(before, after)]
            for run, units in (("main_run_bf16_vision", before), ("vision_fp32", after), ("fp32_minus_bf16", paired)):
                rows.append({"analysis": "P4_minus_P1", "run": run, "placement": placement,
                             "outcome": outcome_name(outcome), **summarize(units, bootstraps)})
    same = sum(design["cells"][key]["prediction"] == row["prediction"] for key, row in vision["cells"].items())
    counts = {"inputs": len(vision["cells"]), "series": len(vision["series"]), "n_images_anchor": len(vision["anchors"]),
              "same_parsed_answer": same, "different_parsed_answer": len(vision["cells"]) - same}
    return rows, counts


def linear_score(model, x):
    """고정 로지스틱 Probe/readout의 logit (predict와 같은 표준화)."""
    return float(((x.double() - model["mean"]) / model["scale"]) @ model["weight"] + model["bias"])


def scales(logits):
    return {"logit": logits, "p": torch.sigmoid(torch.tensor(logits, dtype=torch.float64)).tolist()}


def evidence_scores(run_dir, metadata, records, tolerance, zi, hq):
    """one-N 입력의 N 이미지 z_i 점수(레이어별)와, N이 위치 1~3이면 그 위치의 h_q readout 점수."""
    layers = zi["layers"]
    require(hq["layers"] == layers, "z_i and h_q layers differ")
    probes = {(probe["class"], probe["layer"]): probe for probe in zi["probes"]}
    readouts = {(r["scope"], r["class"], r["layer"], r["position"]): r for r in hq["readouts"]}
    targets = [item for item in records if item["case"]["condition"] == "one_negative"]
    scores = {}
    for item in tqdm(targets, desc="Score N images", unit="record"):
        case = item["case"]
        payload, _ = load_validated_feature(run_dir, item, metadata, tolerance["atol"], tolerance["rtol"],
                                            expected_images=None)
        position = case["negative_positions"][0]
        require(position + 1 == n_position(case["placement"], case["series_step"]),
                f"{case['case_id']}: N position does not follow the placement")
        image = payload["image_features"][:, position]
        entry = {"position": position + 1, "h_q": {},
                 "z_N": scales([linear_score(probes[(case["class"], layer)], image[i]) for i, layer in enumerate(layers)])}
        if position < 3:
            for scope in hq["scopes"]:
                group = case["class"] if scope == "per_class" else "all"
                entry["h_q"][scope] = scales([
                    linear_score(readouts[(scope, group, layer, position + 1)], payload["query_features"][i])
                    for i, layer in enumerate(layers)])
        scores[case["case_id"]] = entry
        del payload
    return scores


def n_position(placement, k):
    """N 이미지의 위치 (1부터): N-first는 1, N-last는 P 장수 + 1."""
    return 1 if placement == "N-first" else k + 1


def exploratory(design, bootstraps, scores, layers, hq_scopes, per_class=True):
    n_image = {"": bootstraps[""]}
    classes = sorted({meta["class"] for meta in design["series"].values()})
    class_rows, z_rows, h_rows = [], [], []
    if per_class:
        for name in classes:
            for placement in PLACEMENTS:
                for outcome in ("yes", "margin"):
                    units = series_units(design, "one_negative", placement, field(outcome), (1, 4), change,
                                         lambda meta, c=name: meta["class"] == c)
                    class_rows.append({"class": name, "placement": placement, "outcome": outcome_name(outcome),
                                       "analysis": "P4_minus_P1", **summarize(units, n_image)})
    if scores is None:
        return {"note": EXPLORATORY_NOTE, "per_class": class_rows, "z_n": [], "h_q": []}
    for placement in PLACEMENTS:
        for i, layer in enumerate(layers):
            for scale in ("logit", "p"):
                value = lambda row, s=scale, j=i: scores[row["case_id"]]["z_N"][s][j]
                for k in STEPS:
                    base = {"placement": placement, "layer": layer, "scale": scale, "p_count": k, "n_images": k + 1}
                    z_rows.append({**base, "analysis": "level", **summarize(
                        series_units(design, "one_negative", placement, value, (k,), first), n_image)})
                    if k > 1:
                        z_rows.append({**base, "analysis": "change_vs_two_image_input", **summarize(
                            series_units(design, "one_negative", placement, value, (1, k), change), n_image)})
                for scope in hq_scopes:
                    h_value = lambda row, s=scale, j=i, c=scope: scores[row["case_id"]]["h_q"][c][s][j]
                    available = [k for k in STEPS if n_position(placement, k) <= 3]
                    for k in available:
                        base = {"placement": placement, "scope": scope, "layer": layer, "scale": scale, "p_count": k,
                                "n_images": k + 1, "n_position": n_position(placement, k)}
                        h_rows.append({**base, "analysis": "level", **summarize(
                            series_units(design, "one_negative", placement, h_value, (k,), first), n_image)})
                        if k > 1 and 1 in available:
                            h_rows.append({**base, "analysis": "change_vs_two_image_input", **summarize(
                                series_units(design, "one_negative", placement, h_value, (1, k), change), n_image)})
    return {"note": EXPLORATORY_NOTE, "per_class": class_rows, "z_n": z_rows, "h_q": h_rows}


# ---------------------------------------------------------------- 입력 확인과 실행

def load_inputs(eval_config):
    """고정 해시·사전 등록 일치·무결성 검사 통과·전체 실행·케이스 순서·비전 fp32 연결을 확인하고 입력을 읽는다."""
    config_path = project_path(eval_config["accumulation_config"])
    config = read_yaml(config_path)
    frozen_dir = project_path(config["frozen_dir"])
    manifest = read_json(frozen_dir / "manifest.json")
    require(manifest["status"] == "frozen", "STEP 5-2 manifest is not marked as frozen")
    changed = verify_accumulation_freeze(manifest)
    require(not changed, f"Files changed after the STEP 5-2 freeze: {changed}")
    require(project_relative_path(config_path) in manifest["files_sha256"], "The accumulation config is not frozen")
    require(config["preregistration"] == manifest["preregistration"], "Config preregistration differs from the frozen one")
    zi_dir = project_path(config["zi_frozen_dir"])
    zi_manifest = read_json(zi_dir / "manifest.json")
    zi_changed = verify_frozen(zi_dir, zi_manifest)
    require(not zi_changed, f"STEP 4 frozen probe files changed: {zi_changed}")
    require(manifest["zi_frozen"] == {"dir": project_relative_path(zi_dir), "version": zi_manifest["version"]},
            "STEP 4 frozen probes differ from the ones recorded in the STEP 5-2 freeze")

    run_dir = project_path(config["run_dir"]).resolve()
    main_check = read_json(run_dir / "main_check.json")
    require(main_check["status"] == "passed" and not main_check["failed"], "The main integrity check did not pass")
    require(main_check["frozen_version"] == manifest["version"], "The main check used another frozen version")
    metadata, records, structure, _ = inspect_run(run_dir)
    require(all(structure.values()), f"Invalid extraction folder: {[k for k, v in structure.items() if not v]}")
    require(metadata["status"] == "completed" and metadata["limit"] is None and metadata["category"] is None,
            "The main extraction is incomplete or a subset")
    require({s["path"]: s["sha256"] for s in metadata["sources"]} == {config["cases"]: manifest["files_sha256"][config["cases"]]},
            "The main extraction used cases that differ from the frozen cases")
    require([item["case"] for item in records] == list(read_jsonl(project_path(config["cases"]))),
            "Main extraction records differ from the frozen cases (content or order)")
    require(len(records) == manifest["counts"]["records"], "Record count differs from the freeze")
    exceeding = [row["case_id"] for row in read_json(run_dir / "n_first_exceeding_inputs.json")["inputs"]
                 if row.get("against", "series_two_image_input") == "series_two_image_input"]

    vision_config = config["vision_fp32"]
    vision_dir = project_path(vision_config["run_dir"]).resolve()
    vision_meta, vision_records, vision_structure, _ = inspect_run(vision_dir)
    require(all(vision_structure.values()), "Invalid vision fp32 extraction folder")
    require(vision_meta["status"] == "completed" and vision_meta["limit"] is None and vision_meta["category"] is None,
            "The vision fp32 extraction is incomplete or a subset")
    require(vision_meta["vision_dtype"] == "torch.float32", "The vision fp32 run did not use a float32 vision encoder")
    require(vision_meta["config"].get("answer_variants") is True and vision_meta["config"].get("fp32_margin") is True,
            "The vision fp32 run lacks the primary margin")
    differences = [key for key in SAME_MODEL_KEYS if vision_meta[key] != metadata[key]]
    require(not differences, f"Vision fp32 run differs from the main run in {differences}")
    require({s["path"]: s["sha256"] for s in vision_meta["sources"]}
            == {vision_config["cases"]: manifest["files_sha256"][vision_config["cases"]]},
            "The vision fp32 extraction used cases that differ from the frozen subset")
    require([item["case"] for item in vision_records] == list(read_jsonl(project_path(vision_config["cases"]))),
            "Vision fp32 records differ from the frozen subset (content or order)")
    return {
        "config_path": config_path, "config": config, "frozen_dir": frozen_dir, "manifest": manifest,
        "zi_dir": zi_dir, "zi_manifest": zi_manifest, "run_dir": run_dir, "main_check": main_check,
        "metadata": metadata, "records": records, "exceeding": exceeding,
        "vision_dir": vision_dir, "vision_records": vision_records,
    }


def structure_summary(inputs):
    """평가 지표 없이 구조만 요약한다 (--check-only)."""
    rows = outcome_table(inputs["records"])
    design = build_design(rows)
    vision = build_design(outcome_table(inputs["vision_records"]), conditions=("one_negative",))
    for key, row in vision["cells"].items():
        require(design["cells"][key]["case_id"] == row["case_id"], f"Vision subset input differs from main run: {key}")
    inputs_by_key = Counter((record["case"]["prompt"], tuple(record["case"]["image_ids"])) for record in inputs["records"])
    return {
        "note": "Structure only: no False-Yes rate, margin or contrast is computed.",
        "frozen_version": inputs["manifest"]["version"], "zi_frozen_version": inputs["zi_manifest"]["version"],
        "records": len(rows), "series": len(design["series"]), "n_images_anchor": len(design["anchors"]),
        "classes": len({meta["class"] for meta in design["series"].values()}),
        "unique_inputs": len(inputs_by_key), "records_sharing_an_input": sum(c - 1 for c in inputs_by_key.values() if c > 1),
        "cells_per_series": len(CONDITIONS) * len(PLACEMENTS) * len(STEPS),
        "integrity_exceeding_inputs": len(inputs["exceeding"]),
        "vision_fp32": {"records": len(vision["cells"]), "series": len(vision["series"]),
                        "n_images_anchor": len(vision["anchors"])},
    }


def analyze(rows, vision_rows, exceeding, preregistration, scores=None, layers=None, hq_scopes=(), settings=None):
    """판정·대비·민감도·탐색 계산 (파일을 읽지 않는 순수 계산; 가짜 데이터 테스트도 이 함수를 쓴다)."""
    settings = settings or {}
    statistics = preregistration["statistics"]
    design = build_design(rows)
    anchor_boot = ImageBootstrap(sorted(design["anchors"]), statistics["reps"], statistics["seed"])
    class_boot = ImageBootstrap(sorted({meta["class"] for meta in design["series"].values()}),
                                statistics["reps"], statistics["seed"])
    bootstraps = {"": (anchor_boot, "anchor"), "class_": (class_boot, "class")}

    main = label_rows(contrasts(design, bootstraps, ("yes", "margin")), preregistration)
    primary = [row for row in main if "label" in row]
    labels = {outcome: {row["placement"]: row["label"] for row in primary if row["outcome"] == outcome}
              for outcome in ("false_yes_rate", "margin_fp32_variant")}
    decision = {
        "rule": {key: preregistration["decision"][key] for key in
                 ("per_placement_supported", "robust_label", "per_placement_no_meaningful_increase", "otherwise")},
        "false_yes": {"by_placement": labels["false_yes_rate"],
                      **overall_verdict(labels["false_yes_rate"], preregistration["decision"])},
        "margin_co_primary": {"by_placement": labels["margin_fp32_variant"],
                              **overall_verdict(labels["margin_fp32_variant"], preregistration["decision"])},
        "note": "H1 is decided by the False-Yes rule; the margin is reported with the same rule as co-primary.",
    }
    sensitivity_rows, sensitivity_summary = sensitivity(design, bootstraps, rows, exceeding, preregistration)
    vision_result, vision_counts = vision_comparison(design, vision_rows, statistics)
    explored = exploratory(design, bootstraps, scores, layers, hq_scopes, settings.get("per_class", True))
    if not settings.get("z_n", True):
        explored["z_n"] = []
    unparsed = Counter((row["condition"], row["placement"], row["n_images"]) for row in rows if row["unparsed"])
    return {
        "decision": decision, "primary": primary, "contrasts": main,
        "levels": levels(design, bootstraps, ("yes", "margin")),
        "sensitivity": {"summary": sensitivity_summary, "rows": sensitivity_rows,
                        "vision_fp32": {"counts": vision_counts, "rows": vision_result}},
        "exploratory": explored,
        "unparsed": {"total": sum(unparsed.values()),
                     "by_cell": [{"condition": c, "placement": p, "n_images": n, "count": count}
                                 for (c, p, n), count in sorted(unparsed.items())]},
    }


def run_accumulation_eval(eval_config, inputs, with_features=True):
    config, manifest = inputs["config"], inputs["manifest"]
    preregistration = manifest["preregistration"]
    rows = outcome_table(inputs["records"])
    scores, layers, scopes = None, None, []
    settings = eval_config.get("exploratory", {})
    if with_features and (settings.get("z_n", True) or settings.get("h_q", True)):
        zi = torch.load(inputs["zi_dir"] / "zi_probes.pt", map_location="cpu", weights_only=True)
        hq = torch.load(inputs["zi_dir"] / "hq_readouts.pt", map_location="cpu", weights_only=True)
        scores = evidence_scores(inputs["run_dir"], inputs["metadata"], inputs["records"], config["verification"], zi, hq)
        layers, scopes = zi["layers"], (hq["scopes"] if settings.get("h_q", True) else [])
    sections = analyze(rows, outcome_table(inputs["vision_records"]), inputs["exceeding"], preregistration,
                       scores, layers, scopes, settings)
    identity = inputs["main_check"]["checks"]["n_first_within_bf16_tolerance"]["detail"]
    report = {
        "status": config["status"], "step": "5-2 accumulation", "definitions": DEFINITIONS,
        "preregistration": {
            "frozen_dir": project_relative_path(inputs["frozen_dir"]), "version": manifest["version"],
            "manifest_sha256": sha256_file(inputs["frozen_dir"] / "manifest.json"),
            "created_at": manifest["created_at"], "hypotheses": preregistration["hypotheses"],
            "primary": preregistration["primary"], "co_primary_margin": preregistration["co_primary_margin"],
            "minimum_effect": preregistration["minimum_effect"], "statistics": preregistration["statistics"],
            "model_predictions": preregistration["model_predictions"],
            "zi_frozen": manifest["zi_frozen"],
        },
        "inputs": {"run_dir": project_relative_path(inputs["run_dir"]),
                   "vision_run_dir": project_relative_path(inputs["vision_dir"]),
                   "accumulation_config": project_relative_path(inputs["config_path"])},
        "counts": structure_summary(inputs),
        "integrity": {"n_first_identity": identity,
                      "margin_sign_agreement": inputs["main_check"]["checks"]["margin_sign_agreement_reported"]["detail"]},
        **sections,
    }
    report["counts"].pop("note", None)
    return report, rows, scores
