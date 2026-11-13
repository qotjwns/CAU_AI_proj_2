"""STEP 5-2b 평가 (CPU, confirmatory).

사전 등록 v1(고정 manifest)의 가설·대비·판정 규칙을 그대로 적용한다. 판정에 쓰는 값(최소 효과, 판정 규칙, 반복·seed)은
설정 파일이 아니라 고정 manifest에서 읽는다. 평가 결과를 본 뒤 주 대비와 판정 기준을 바꾸지 않는다.

- 계열(series): N 이미지 1장 × 강도(High/Low). P 장수 k = 2, 3, 4(전체 3~5장)마다 N 위치 a = 0..k (a = N 앞의 P 수).
- H1b: 가운데 위치(a = 1..k-1) 평균 False-Yes − 끝 위치(a = k) False-Yes > 0
- H2b: 가운데 위치 평균 False-Yes − 첫 위치(a = 0) False-Yes > 0
- 계열마다 장수별 대비를 구하고 3·4·5장을 같은 가중치로 평균한 뒤 계열 평균 (High·Low 같은 가중치).
- 주 판정: 97.5% N 이미지 군집 부트스트랩 (Bonferroni), robust 표기: 97.5% 클래스 군집. 부가·탐색: 95%.
- 확증 대비는 모두 5-2b 한 실행 안의 값. 5-2 값은 실행 간 비교(exploratory)에만 쓴다.
"""
from collections import defaultdict
import math
import statistics

import torch
from tqdm import tqdm

from src.common.config import read_yaml
from src.common.data import read_json, read_jsonl
from src.common.paths import project_path, project_relative_path
from src.core.accumulation_explore import Clusters, cosine_distance, index as step52_index, load_run as load_step52_run
from src.core.accumulation_explore import probe_logits, sigmoid, z_n_by_a as step52_z_n_by_a
from src.core.accumulation_freeze import verify_accumulation_freeze
from src.core.feature_validation import inspect_run, load_validated_feature, require
from src.core.probe_freeze import sha256_file, verify_frozen


LAYERS = (9, 18, 27, 36)
COUNTS = (2, 3, 4)
HYPOTHESES = (("H1b", "last"), ("H2b", "first"))
PRIMARY_LEVEL = 0.975
SECONDARY_LEVEL = 0.95
LABELS = {"robust": "robust support", "support": "support (cross-class generalization uncertain)",
          "none": "no meaningful difference", "other": "inconclusive", "missing": "not estimable"}
DEFINITIONS = {
    "series": "one N image x one strength (High/Low); P order P1..P5 from STEP 5-2",
    "position": "a = P images before N, b = P images after N, k = a + b P images (3-5 images in total)",
    "middle_mean": "3 images: a = 1; 4 images: mean over a = 1, 2; 5 images: mean over a = 1, 2, 3",
    "contrast": ("per series and image count: middle_mean minus the end (a = k for H1b, a = 0 for H2b); "
                 "equal-weight mean over 3, 4 and 5 images; mean over series (High and Low equal weight)"),
    "false_yes": "one-N input with parsed answer Yes; unparsed answers count as not Yes and are reported separately",
    "margin_fp32_variant": "fp32_variant_logit_margin (primary margin definition, estimates only)",
    "margin_bf16": "logit_margin: bf16 'Yes' / 'No' generation logits (estimates only)",
    "bootstrap": "percentile cluster bootstrap; replicate estimate = draw-count-weighted mean over series",
    "exclusion_rule": "a series is dropped from a contrast if any input it uses is excluded",
    "sign_consistency": "bf16 margin > 0 with answer Yes or < 0 with answer No; ties and unparsed answers excluded",
    "within_run": "every confirmatory contrast uses values from the STEP 5-2b run only",
}
CROSS_RUN_NOTE = ("Cross-run comparison (exploratory): STEP 5-2 values come from a different extraction run of the "
                  "same model and environment; not part of any decision.")


# ---------------------------------------------------------------- 입력 표와 설계

def outcome_table(records):
    rows = []
    for item in records:
        case = item["case"]
        prediction, bf16, primary = item["prediction"], item["logit_margin"], item["fp32_variant_logit_margin"]
        require(math.isfinite(bf16) and math.isfinite(primary), f"{case['case_id']}: non-finite margin")
        rows.append({
            "case_id": case["case_id"], "series": case["pair_id"], "anchor": case["anchor_negative_id"],
            "class": case["class"], "strength": case["strength"], "k": case["series_step"],
            "n_images": case["n_images"], "a": case["n_before"], "b": case["n_after"], "role": case["role"],
            "source_cases": case["source_cases"], "prediction": prediction, "yes": int(prediction == "Yes"),
            "unparsed": prediction not in ("Yes", "No"), "margin": float(primary), "margin_bf16": float(bf16),
            "sign_consistent": (bf16 > 0 and prediction == "Yes") or (bf16 < 0 and prediction == "No"),
        })
    return rows


def build_design(rows):
    cells, meta = {}, {}
    for row in rows:
        key = (row["series"], row["k"], row["a"])
        require(key not in cells, f"Duplicate input for {key}")
        cells[key] = row
        info = (row["anchor"], row["class"], row["strength"])
        require(meta.setdefault(row["series"], info) == info, f"Inconsistent metadata in {row['series']}")
    missing = [(s, k, a) for s in meta for k in COUNTS for a in range(k + 1) if (s, k, a) not in cells]
    require(not missing, f"{len(missing)} position cells are missing, e.g. {missing[:3]}")
    require(len(cells) == len(meta) * sum(k + 1 for k in COUNTS), "Unexpected extra inputs")
    return {"cells": cells, "meta": meta}


def field(name):
    return lambda row: row[name]


def end_position(end, k):
    return 0 if end == "first" else k


def hypothesis_units(design, end, value, counts=COUNTS):
    """계열마다 (계열, 값, 사용한 입력들): 장수별 (가운데 평균 − 끝)을 같은 가중치로 평균."""
    units = []
    for series in sorted(design["meta"]):
        parts, members = [], []
        for k in counts:
            middle = [design["cells"][(series, k, a)] for a in range(1, k)]
            terminal = design["cells"][(series, k, end_position(end, k))]
            parts.append(math.fsum(value(row) for row in middle) / len(middle) - value(terminal))
            members += [row["case_id"] for row in middle] + [terminal["case_id"]]
        units.append((series, math.fsum(parts) / len(parts), tuple(members)))
    return units


def summarize(units, clusters, level, exclude=frozenset()):
    kept = [(series, value) for series, value, members in units if not exclude.intersection(members)]
    row = clusters.summary([s for s, _ in kept], [v for _, v in kept], level=level)
    row.pop("n", None)
    return {"level": level, "n_series": len(kept), "excluded_series": len(units) - len(kept),
            "n_anchors": len({clusters.meta[s][0] for s, _ in kept}), **row}


def decide(row, minimum):
    """사전 등록 판정 규칙 (가설별)."""
    if row.get("estimate") is None or row.get("ci_low") is None:
        return LABELS["missing"]
    if row["estimate"] >= minimum and row["ci_low"] > 0:
        return LABELS["robust"] if row.get("class_ci_low") is not None and row["class_ci_low"] > 0 else LABELS["support"]
    if row["ci_high"] < minimum:
        return LABELS["none"]
    return LABELS["other"]


# ---------------------------------------------------------------- 확증·부가 분석

def primary(design, clusters, preregistration, exclude=frozenset()):
    minimum = preregistration["minimum_effect"]["false_yes_rate"]
    rows = []
    for name, end in HYPOTHESES:
        row = {"hypothesis": name, "statement": preregistration["hypotheses"][name],
               "contrast": f"middle mean - {end} position", "outcome": "false_yes_rate",
               **summarize(hypothesis_units(design, end, field("yes")), clusters, PRIMARY_LEVEL, exclude)}
        row.update(minimum_effect=minimum, label=decide(row, minimum))
        rows.append(row)
    return rows


def secondary_contrasts(design, clusters):
    """장수별 대비 (False-Yes, 두 margin)와 3~5장 통합 margin 대비. 판정 없음, 95%."""
    rows = []
    for name, end in HYPOTHESES:
        for outcome, label in (("yes", "false_yes_rate"), ("margin", "margin_fp32_variant"),
                               ("margin_bf16", "margin_bf16")):
            scopes = [("3-5 images (equal weight)", COUNTS)] if outcome != "yes" else []
            scopes += [(f"{k + 1} images", (k,)) for k in COUNTS]
            for scope, counts in scopes:
                rows.append({"hypothesis": name, "contrast": f"middle mean - {end} position", "outcome": label,
                             "scope": scope, **summarize(hypothesis_units(design, end, field(outcome), counts),
                                                         clusters, SECONDARY_LEVEL)})
    return rows


def step4_comparison(design, clusters, preregistration):
    """3장: NPP(a=0), PNP(a=1), PPN(a=2) False-Yes 비율과 PNP − NPP (H2b의 3장 부분), STEP 4 값과 나란히."""
    names = sorted(design["meta"])
    reference = preregistration["observed_before"]["step4_three_image_rates"]
    rows = []
    for a, composition in enumerate(("NPP", "PNP", "PPN")):
        values = [float(design["cells"][(s, 2, a)]["yes"]) for s in names]
        rows.append({"composition": composition, "measure": "false_yes_rate", "step4_value": reference[composition],
                     **clusters.summary(names, values, level=SECONDARY_LEVEL)})
    for label, first, second in (("PNP - NPP (part of H2b)", 1, 0), ("PNP - PPN (part of H1b)", 1, 2)):
        values = [float(design["cells"][(s, 2, first)]["yes"] - design["cells"][(s, 2, second)]["yes"]) for s in names]
        rows.append({"composition": label, "measure": "difference",
                     "step4_value": reference["PNP"] - reference["NPP" if second == 0 else "PPN"],
                     **clusters.summary(names, values, level=SECONDARY_LEVEL)})
    note = "Follow-up check on the same image pool (STEP 4 images, different P combinations), not an independent replication."
    return rows, note


def sign_consistency(design, clusters, rows, preregistration):
    excluded = frozenset(row["case_id"] for row in rows if not row["sign_consistent"])
    result = [{"variant": "sign_consistent", **row} for row in primary(design, clusters, preregistration, excluded)]
    return result, {"excluded_inputs": len(excluded),
                    "description": "H1b and H2b without inputs whose bf16 margin sign disagrees with the answer (ties included)"}


def cross_run(rows, step52_records, z_pairs=None):
    """5-2와 5-2b의 같은 입력(첫·끝 위치): 답 일치, margin 차이, (특징이 있으면) z_N 코사인 거리."""
    pairs = []
    for row in rows:
        if row["role"] == "middle":
            continue
        source = step52_records[next(iter(row["source_cases"].values()))]
        pairs.append((row, source))
    same = sum(row["prediction"] == source["prediction"] for row, source in pairs)
    report = {"note": CROSS_RUN_NOTE, "identical_inputs": len(pairs), "same_parsed_answer": same,
              "answer_agreement": same / len(pairs) if pairs else None}
    for name, key in (("margin_fp32_variant", "margin"), ("margin_bf16", "margin_bf16")):
        differences = [row[key] - source[key] for row, source in pairs]
        report[name] = {"mean_difference": statistics.fmean(differences),
                        "mean_abs_difference": statistics.fmean(abs(d) for d in differences),
                        "max_abs_difference": max(abs(d) for d in differences),
                        "exactly_equal": sum(d == 0 for d in differences)}
    if z_pairs:
        distances = [max(cosine_distance(first.double(), second.double())) for first, second in z_pairs]
        report["z_n_max_cosine_distance"] = {"n": len(distances), "max": max(distances),
                                             "median": statistics.median(distances),
                                             "exactly_identical": sum(d == 0 for d in distances)}
    return report


# ---------------------------------------------------------------- 탐색 분석

def grid_levels(design, clusters):
    rows = []
    for k in COUNTS:
        for a in range(k + 1):
            for strength in ("pooled", "High", "Low"):
                names = sorted(s for s, m in design["meta"].items() if strength == "pooled" or m[2] == strength)
                for outcome, label in (("yes", "false_yes_rate"), ("margin", "margin_fp32_variant")):
                    values = [float(design["cells"][(s, k, a)][outcome]) for s in names]
                    rows.append({"n_images": k + 1, "a": a, "b": k - a, "strength": strength, "outcome": label,
                                 **clusters.summary(names, values, level=SECONDARY_LEVEL)})
    return rows


def ab_slopes(design, clusters):
    """H3 (탐색): a 고정 시 b에 따른, b 고정 시 a에 따른 계열 안 최소제곱 기울기 (장수도 함께 바뀜)."""
    names = sorted(design["meta"])
    cells_by = {"a": defaultdict(list), "b": defaultdict(list)}
    for k in COUNTS:
        for a in range(k + 1):
            cells_by["a"][a].append((k - a, k, a))
            cells_by["b"][k - a].append((a, k, a))
    rows = []
    for fixed, other in (("a", "b"), ("b", "a")):
        for level_value, points in sorted(cells_by[fixed].items()):
            if len(points) < 2:
                continue
            xs = [x for x, _, _ in points]
            centre = statistics.fmean(xs)
            denominator = math.fsum((x - centre) ** 2 for x in xs)
            for outcome, label in (("yes", "false_yes_rate"), ("margin", "margin_fp32_variant")):
                values = [math.fsum((x - centre) * design["cells"][(s, k, a)][outcome] for x, k, a in points) / denominator
                          for s in names]
                rows.append({"fixed": f"{fixed} = {level_value}", "slope_over": other,
                             "cells": [f"{k + 1} images, a={a}" for _, k, a in points], "outcome": label,
                             **clusters.summary(names, values, level=SECONDARY_LEVEL)})
    return rows


def position_influence(design, clusters, allp):
    """실행 간 비교 (exploratory): 같은 계열·장수의 5-2 All-P Yes 비율(두 순서 평균) − 5-2b one-N (a, b) 칸."""
    names = sorted(design["meta"])
    rows = []
    for k in COUNTS:
        for a in range(k + 1):
            for outcome, label in (("yes", "yes_rate"), ("margin", "margin_fp32_variant")):
                values = []
                for s in names:
                    reference = allp[(s, k)]
                    values.append(statistics.fmean(float(r[outcome]) for r in reference)
                                  - float(design["cells"][(s, k, a)][outcome]))
                rows.append({"n_images": k + 1, "a": a, "b": k - a, "outcome": label,
                             "measure": "STEP 5-2 All-P (same series and count) minus STEP 5-2b one-N",
                             "note": CROSS_RUN_NOTE, **clusters.summary(names, values, level=SECONDARY_LEVEL)})
    return rows


def first_input_with(design, series, a):
    """z_N은 a에만 의존 (prefix 성질): 그 a를 가진 가장 짧은 입력."""
    k = max(a, COUNTS[0])
    return design["cells"][(series, k, a)]


def z_n_by_a(design, clusters, features):
    names = sorted(design["meta"])
    rows = []
    for i, layer in enumerate(LAYERS):
        noise = [cosine_distance(features[design["cells"][(s, k, 0)]["case_id"]]["z_n"][i].double(),
                                 features[design["cells"][(s, COUNTS[0], 0)]["case_id"]]["z_n"][i].double())
                 for s in names for k in COUNTS[1:]]
        rows.append({"run": "step5_2b", "layer": layer, "a": 0, "kind": "within-run noise floor (a = 0, k = 3, 4 vs 2)",
                     "cos_median": statistics.median(noise), "cos_max": max(noise)})
        for a in range(1, max(COUNTS) + 1):
            reference = [features[first_input_with(design, s, 0)["case_id"]] for s in names]
            current = [features[first_input_with(design, s, a)["case_id"]] for s in names]
            distance = [cosine_distance(c["z_n"][i].double(), r["z_n"][i].double()) for c, r in zip(current, reference)]
            logit = [c["z_n_logit"][i] - r["z_n_logit"][i] for c, r in zip(current, reference)]
            prob = [sigmoid(c["z_n_logit"][i]) - sigmoid(r["z_n_logit"][i]) for c, r in zip(current, reference)]
            row = {"run": "step5_2b", "layer": layer, "a": a, "kind": "z_N with a P images before vs a = 0",
                   "cos_median": statistics.median(distance), "cos_mean": statistics.fmean(distance),
                   "cos_max": max(distance)}
            row.update({f"logit_{k}": v for k, v in clusters.summary(names, logit, level=SECONDARY_LEVEL).items() if k != "n"})
            row.update({f"p_{k}": v for k, v in clusters.summary(names, prob, level=SECONDARY_LEVEL).items() if k != "n"})
            rows.append(row)
    return rows


def z_n_margin_relation(design, clusters, features):
    """계열·장수 안에서 a > 0 위치의 (z_N logit 변화, margin 변화)를 a = 0 기준으로 모아 기울기와 상관."""
    names = sorted(design["meta"])
    rows = []
    for i, layer in enumerate(LAYERS):
        x, y, keys = [], [], []
        for s in names:
            for k in COUNTS:
                base = design["cells"][(s, k, 0)]
                for a in range(1, k + 1):
                    cell = design["cells"][(s, k, a)]
                    x.append(features[cell["case_id"]]["z_n_logit"][i] - features[base["case_id"]]["z_n_logit"][i])
                    y.append(cell["margin"] - base["margin"])
                    keys.append(clusters.meta[s][0])
        xt, yt = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
        weights = clusters.anchor.unit_weights(keys)
        total = weights.sum(1, keepdim=True)
        mx, my = (weights * xt).sum(1, keepdim=True) / total, (weights * yt).sum(1, keepdim=True) / total
        slopes = (weights * (xt - mx) * (yt - my)).sum(1) / (weights * (xt - mx) ** 2).sum(1)
        low, high = torch.quantile(slopes, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
        cx, cy = xt - xt.mean(), yt - yt.mean()
        rows.append({"layer": layer, "points": len(x), "slope_margin_per_logit": float((cx * cy).sum() / (cx * cx).sum()),
                     "ci_low": low, "ci_high": high, "level": SECONDARY_LEVEL,
                     "pearson_r": float((cx * cy).sum() / (cx.norm() * cy.norm()))})
    return rows


def h_q_three_images(design, clusters, features, scopes):
    """3장 입력(a = 0, 1, 2)에서만: N 위치 readout p(P)와 N 위치 정확도(argmin p(P) = N 위치)."""
    names = sorted(design["meta"])
    rows = []
    for scope in scopes:
        for i, layer in enumerate(LAYERS):
            for a in range(3):
                entries = [features[design["cells"][(s, 2, a)]["case_id"]]["hq"][scope][i] for s in names]
                p_n = [float(e[a]) for e in entries]
                located = [float(int(torch.argmin(e)) == a) for e in entries]
                for measure, values in (("p(P) at N position", p_n), ("N localization (argmin p(P) = N)", located)):
                    rows.append({"scope": scope, "layer": layer, "a": a, "composition": ("NPP", "PNP", "PPN")[a],
                                 "measure": measure, **clusters.summary(names, values, level=SECONDARY_LEVEL)})
    return rows


def position_accuracy(design, features):
    """위치별 고정 z_i Probe 정확도 (P: logit >= 0, N: logit < 0), 입력 길이에 따른 분포 이동 점검."""
    counts = defaultdict(lambda: [0, 0])
    for (s, k, a), row in design["cells"].items():
        logits = features[row["case_id"]]["logits"]
        for i, layer in enumerate(LAYERS):
            for position in range(k + 1):
                label = "N" if position == a else "P"
                hit = (logits[i, position].item() >= 0) == (label == "P")
                key = (layer, k + 1, position + 1, label)
                counts[key][0] += hit
                counts[key][1] += 1
    return [{"layer": layer, "n_images": n, "position": p, "label": label, "images": total, "accuracy": hit / total}
            for (layer, n, p, label), (hit, total) in sorted(counts.items())]


def analyze(rows, step52, preregistration, features=None, settings=None):
    """파일을 읽지 않는 순수 계산. step52: {"records": {case_id: 행}, "allp": {(계열, k): [행, 행]},
    "z_pairs": [(z_5_2b, z_5_2)] 또는 None, "z_n_by_a": 5-2 N-last 같은 형식 행 또는 []}."""
    settings = settings or {}
    statistics_ = preregistration["statistics"]
    design = build_design(rows)
    clusters = Clusters(design["meta"], statistics_["reps"], statistics_["seed"])
    main = primary(design, clusters, preregistration)
    comparison, comparison_note = step4_comparison(design, clusters, preregistration)
    sensitivity, sensitivity_summary = sign_consistency(design, clusters, rows, preregistration)
    unparsed = sum(row["unparsed"] for row in rows)
    report = {
        "decision": {
            "rule": preregistration["decision"],
            "by_hypothesis": {row["hypothesis"]: row["label"] for row in main},
            "note": "H1b and H2b are decided separately; neither changes the STEP 5-2 hypotheses or decisions.",
        },
        "primary": main,
        "secondary": {"contrasts": secondary_contrasts(design, clusters),
                      "step4_comparison": {"note": comparison_note, "rows": comparison},
                      "sign_consistency": {**sensitivity_summary, "rows": sensitivity},
                      "cross_run": cross_run(rows, step52["records"], step52.get("z_pairs"))},
        "unparsed": {"total": unparsed},
        "exploratory": {
            "note": "Exploratory, 95% intervals, not part of any decision.",
            "grid": grid_levels(design, clusters), "ab_slopes": ab_slopes(design, clusters),
            "position_influence": position_influence(design, clusters, step52["allp"]),
        },
    }
    if features is not None:
        report["exploratory"].update(
            z_n_by_a=z_n_by_a(design, clusters, features) + list(step52.get("z_n_by_a", [])),
            z_n_margin_relation=z_n_margin_relation(design, clusters, features),
            h_q_three_images=h_q_three_images(design, clusters, features, settings.get("hq_scopes", ["pooled", "per_class"])),
            position_accuracy=position_accuracy(design, features),
            representation_note=("a changes both the preceding images and N's absolute token position (not separated); "
                                 "h_q readouts are applied to 3-image inputs only (trained on 3 images)."))
    return report


# ---------------------------------------------------------------- 입력 확인과 실행

def load_inputs(eval_config):
    config_path = project_path(eval_config["position_config"])
    config = read_yaml(config_path)
    frozen_dir = project_path(config["frozen_dir"])
    manifest = read_json(frozen_dir / "manifest.json")
    require(manifest["status"] == "frozen", "STEP 5-2b manifest is not marked as frozen")
    changed = verify_accumulation_freeze(manifest)
    require(not changed, f"Files changed after the STEP 5-2b freeze: {changed}")
    require(project_relative_path(config_path) in manifest["files_sha256"], "The position config is not frozen")
    require(config["preregistration"] == manifest["preregistration"], "Config preregistration differs from the frozen one")
    zi_dir = project_path(config["zi_frozen_dir"])
    zi_manifest = read_json(zi_dir / "manifest.json")
    require(not verify_frozen(zi_dir, zi_manifest), "STEP 4 frozen probe files changed")
    step52_dir = project_path(config["accumulation"]["frozen_dir"])
    require(not verify_accumulation_freeze(read_json(step52_dir / "manifest.json")), "STEP 5-2 frozen files changed")

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

    step52_run = project_path(config["accumulation"]["run_dir"]).resolve()
    step52_meta, step52_records, step52_structure, _ = inspect_run(step52_run)
    require(all(step52_structure.values()), "STEP 5-2 extraction folder is invalid")
    return {"config_path": config_path, "config": config, "frozen_dir": frozen_dir, "manifest": manifest,
            "zi_dir": zi_dir, "zi_manifest": zi_manifest, "run_dir": run_dir, "main_check": main_check,
            "metadata": metadata, "records": records, "step52_run": step52_run, "step52_meta": step52_meta,
            "step52_records": step52_records}


def step52_tables(records):
    """5-2 레코드 → 답·margin 표 (case_id), All-P (계열, k) 두 순서."""
    table, allp = {}, defaultdict(list)
    for item in records:
        case = item["case"]
        row = {"prediction": item["prediction"], "yes": int(item["prediction"] == "Yes"),
               "margin": float(item["fp32_variant_logit_margin"]), "margin_bf16": float(item["logit_margin"])}
        table[case["case_id"]] = row
        if case["condition"] == "all_positive":
            allp[(case["pair_id"], case["series_step"])].append(row)
    return table, dict(allp)


def structure_summary(inputs):
    rows = outcome_table(inputs["records"])
    design = build_design(rows)
    table, allp = step52_tables(inputs["step52_records"])
    missing_sources = sum(next(iter(r["source_cases"].values())) not in table for r in rows if r["role"] != "middle")
    return {
        "note": "Structure only: no False-Yes rate, margin or contrast is computed.",
        "frozen_version": inputs["manifest"]["version"], "records": len(rows), "series": len(design["meta"]),
        "n_images_anchor": len({m[0] for m in design["meta"].values()}),
        "classes": len({m[1] for m in design["meta"].values()}), "cells_per_series": sum(k + 1 for k in COUNTS),
        "identical_5_2_inputs_found": sum(1 for r in rows if r["role"] != "middle") - missing_sources,
        "all_positive_pairs": sum(len(v) == 2 for v in allp.values()),
        "environment": {"step5_2b": inputs["metadata"].get("environment", {}).get("gpus"),
                        "step5_2_versions": inputs["step52_meta"].get("versions")},
    }


def load_position_features(inputs, zi, hq):
    probes = {(p["class"], p["layer"]): p for p in zi["probes"]}
    readouts = {(r["scope"], r["class"], r["layer"], r["position"]): r for r in hq["readouts"]}
    tolerance = inputs["config"]["verification"]
    features = {}
    for item in tqdm(inputs["records"], desc="Load 5-2b features", unit="record"):
        case = item["case"]
        payload, _ = load_validated_feature(inputs["run_dir"], item, inputs["metadata"], tolerance["atol"],
                                            tolerance["rtol"], expected_images=None)
        logits = probe_logits(probes, case["class"], payload["image_features"])
        entry = {"z_n": payload["image_features"][:, case["n_before"]].float().clone(), "logits": logits,
                 "z_n_logit": logits[:, case["n_before"]].tolist(), "hq": {}}
        if case["n_images"] == 3:
            for scope in hq["scopes"]:
                group = case["class"] if scope == "per_class" else "all"
                values = torch.empty(len(LAYERS), 3, dtype=torch.float64)
                for i, layer in enumerate(LAYERS):
                    for position in range(3):
                        readout = readouts[(scope, group, layer, position + 1)]
                        query = payload["query_features"][i].double()
                        values[i, position] = torch.sigmoid(((query - readout["mean"]) / readout["scale"])
                                                            @ readout["weight"] + readout["bias"])
                entry["hq"][scope] = values
        features[case["case_id"]] = entry
        del payload
    return features


def run_position_eval(eval_config, inputs, with_features=True):
    config, manifest = inputs["config"], inputs["manifest"]
    preregistration = manifest["preregistration"]
    rows = outcome_table(inputs["records"])
    table, allp = step52_tables(inputs["step52_records"])
    step52 = {"records": table, "allp": allp, "z_pairs": None, "z_n_by_a": []}
    features, settings = None, dict(eval_config.get("exploratory", {}))
    if with_features and settings.get("features", True):
        zi = torch.load(inputs["zi_dir"] / "zi_probes.pt", map_location="cpu", weights_only=True)
        hq = torch.load(inputs["zi_dir"] / "hq_readouts.pt", map_location="cpu", weights_only=True)
        settings["hq_scopes"] = hq["scopes"]
        features = load_position_features(inputs, zi, hq)
        probes = {(p["class"], p["layer"]): p for p in zi["probes"]}
        data52 = load_step52_run(inputs["step52_run"], probes, config["verification"])
        step52["z_pairs"] = [(features[r["case_id"]]["z_n"], data52[next(iter(r["source_cases"].values()))]["z_n"])
                             for r in rows if r["role"] != "middle"]
        cells52, meta52 = step52_index(data52)
        step52["z_n_by_a"] = [{**row, "run": "step5_2 N-last (same format, a = k)"} for row in
                              step52_z_n_by_a(cells52, meta52, Clusters(meta52, preregistration["statistics"]["reps"],
                                                                       preregistration["statistics"]["seed"]), "step5_2")]
    sections = analyze(rows, step52, preregistration, features, settings)
    report = {
        "status": config["status"], "step": "5-2b position grid", "definitions": DEFINITIONS,
        "preregistration": {
            "frozen_dir": project_relative_path(inputs["frozen_dir"]), "version": manifest["version"],
            "manifest_sha256": sha256_file(inputs["frozen_dir"] / "manifest.json"), "created_at": manifest["created_at"],
            "hypotheses": preregistration["hypotheses"], "minimum_effect": preregistration["minimum_effect"],
            "decision": preregistration["decision"], "statistics": preregistration["statistics"],
            "observed_before": preregistration["observed_before"], "scope": preregistration["scope"],
        },
        "inputs": {"run_dir": project_relative_path(inputs["run_dir"]),
                   "step5_2_run_dir": project_relative_path(inputs["step52_run"]),
                   "position_config": project_relative_path(inputs["config_path"])},
        "counts": {k: v for k, v in structure_summary(inputs).items() if k != "note"},
        "integrity": {name: inputs["main_check"]["checks"][name].get("detail") for name in
                      ("prefix_identity_within_run", "prefix_identity_across_runs_reported", "environment_recorded",
                       "margin_sign_agreement_reported")},
        **sections,
    }
    return report, rows, features
