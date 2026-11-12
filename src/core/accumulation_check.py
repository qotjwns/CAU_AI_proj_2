"""STEP 5-2 추출 무결성 검사 (CPU): 파일럿과 본 실행에 공통.

구조·연결·설정·토큰 수·N-first 표현 동일성(bf16 수치 오차 기준)·답 파싱·margin 부호 일치·시간만 확인한다.
False-Yes 비율, margin 비교 등 평가 지표는 계산하지도 출력하지도 않는다.
"""
from collections import defaultdict
import os
import statistics

import torch

from src.common.data import read_json, read_jsonl
from src.common.paths import project_path, project_relative_path
from src.core.accumulation_freeze import verify_accumulation_freeze
from src.core.feature_validation import inspect_run, load_validated_feature
from src.core.probe import predict


NOTE = ("Integrity check only: structure, linkage, settings, token counts, N-first representation identity within "
        "bf16 numerical error, answer parsing, margin sign agreement and timing. No False-Yes rate, margin contrast "
        "or other evaluation metric is computed.")
MARGINS = ("logit_margin", "variant_logit_margin", "fp32_logit_margin", "fp32_variant_logit_margin")
STEP4_RUN = "results/step4/qwen2_5_vl_3b/extraction"


def representation_gap(values, reference):
    """레이어별 코사인 거리와 상대 L2 (values, reference: [레이어, 차원])."""
    cosine = torch.nn.functional.cosine_similarity(values.double(), reference.double(), dim=-1)
    relative = (values.double() - reference.double()).norm(dim=-1) / reference.double().norm(dim=-1)
    return (1 - cosine).tolist(), relative.tolist()


def n_first_identity(records, features, config, probes, layers):
    """N-first one-N 입력에서 N(위치 1) 표현을 같은 계열의 2장 입력, 그리고 STEP 4 위치 1 벡터와 비교한다."""
    tolerance = config["integrity"]["n_first_max_cosine_distance"]
    step4_dir = project_path(STEP4_RUN)
    step4_first = {}
    for item in read_jsonl(step4_dir / "manifest.jsonl"):
        case = item["case"]
        if case["image_labels"][0] == "N":
            step4_first.setdefault((case["class"], case["image_ids"][0]), item)
    series = defaultdict(dict)
    for item in records:
        case = item["case"]
        if case["condition"] == "one_negative" and case["placement"] == "N-first":
            series[case["pair_id"]][case["series_step"]] = (item, features[item["index"]])
    rows, exceeding = [], []
    for pair_id, by_step in sorted(series.items()):
        anchor = next(iter(by_step.values()))[0]["case"]
        step4_item = step4_first[(anchor["class"], anchor["anchor_negative_id"])]
        step4_values = torch.load(step4_dir / step4_item["feature_file"], map_location="cpu",
                                  weights_only=True)["image_features"][:, 0]
        references = {"series_two_image_input": by_step[min(by_step)][1], "step4_position1": step4_values}
        for step, (item, values) in sorted(by_step.items()):
            for against, reference in references.items():
                if against == "series_two_image_input" and step == min(by_step):
                    continue
                cosine, relative = representation_gap(values, reference)
                delta = [float(predict(probes[(anchor["class"], layer)], values[i:i + 1])
                               - predict(probes[(anchor["class"], layer)], reference[i:i + 1]))
                         for i, layer in enumerate(layers)]
                row = {"case_id": item["case"]["case_id"], "series": pair_id, "series_step": step,
                       "n_images": item["case"]["n_images"], "against": against,
                       "max_cosine_distance": max(cosine), "max_relative_l2": max(relative),
                       "max_abs_delta_p": max(abs(value) for value in delta),
                       **{f"cosine_distance_L{layer}": value for layer, value in zip(layers, cosine)},
                       **{f"delta_p_L{layer}": value for layer, value in zip(layers, delta)}}
                rows.append(row)
                if row["max_cosine_distance"] > tolerance:
                    exceeding.append(row)
    summary = {}
    for against in ("series_two_image_input", "step4_position1"):
        selected = [row for row in rows if row["against"] == against]
        if selected:
            summary[against] = {
                "n": len(selected), "exceeding": sum(row["max_cosine_distance"] > tolerance for row in selected),
                **{f"max_{key}": max(row[key] for row in selected)
                   for key in ("max_cosine_distance", "max_relative_l2", "max_abs_delta_p")},
                **{f"median_{key}": statistics.median(row[key] for row in selected)
                   for key in ("max_cosine_distance", "max_relative_l2", "max_abs_delta_p")}}
    return rows, exceeding, summary


def check_run(run_dir, config, kind):
    frozen_dir = project_path(config["frozen_dir"])
    manifest = read_json(frozen_dir / "manifest.json")
    tolerance = config["verification"]
    metadata, records, checks, inventory = inspect_run(run_dir)
    step4_meta = read_json(project_path(STEP4_RUN) / "run.json")
    results, problems = {}, []

    def record(name, passed, detail=None):
        results[name] = {"passed": bool(passed), **({"detail": detail} if detail is not None else {})}
        if not passed:
            problems.append(name)

    record("run_structure", all(checks.values()), {k: v for k, v in checks.items() if not v} or inventory)
    if kind == "pilot":
        record("extraction_verification_flags", metadata["verify_repeat"] and metadata["verify_reference"],
               "run the pilot with --verify-repeat --verify-reference")
        record("extraction_verification_passed", metadata.get("verification_passed") is True)
        record("pilot_limit_matches_frozen", metadata["limit"] == manifest["pilot"]["limit"],
               {"run": metadata["limit"], "frozen": manifest["pilot"]["limit"]})
    settings = metadata["config"]
    record("step5_options", settings.get("expected_images") is None and settings.get("answer_variants") is True
           and settings.get("fp32_margin") is True and settings["layers"] == [9, 18, 27, 36]
           and settings.get("vision_dtype") is None)
    same = {key: metadata[key] == step4_meta[key] for key in (
        "image_processor", "model_config", "chat_template", "dtype", "attention_implementation", "pooling")}
    record("same_model_resolution_precision_as_step4", all(same.values()), same)
    sources = {source["path"]: source["sha256"] for source in metadata["sources"]}
    record("cases_file_hash_matches_freeze",
           sources.get(config["cases"]) == manifest["files_sha256"][config["cases"]], sources)

    frozen_cases = list(read_jsonl(project_path(config["cases"])))
    by_id = {case["case_id"]: case for case in frozen_cases}
    if kind == "main":
        record("all_frozen_cases_in_order", metadata["limit"] is None
               and [item["case"] for item in records] == frozen_cases, {"records": len(records), "frozen": len(frozen_cases)})
    link, files, order, query, load = [], [], [], [], []
    zi = torch.load(project_path(config["zi_frozen_dir"]) / "zi_probes.pt", map_location="cpu", weights_only=False)
    probes = {(probe["class"], probe["layer"]): probe for probe in zi["probes"]}
    layers = settings["layers"]
    features = {}
    for item in records:
        case = item["case"]
        if by_id.get(case["case_id"]) != case:
            link.append(case["case_id"])
        for name, image_id in zip(case["image_order"], case["image_ids"]):
            if name != f"{image_id:012d}.jpg" or not (project_path(case["image_dir"]) / name).exists():
                files.append(case["case_id"])
        if [span["image_id"] for span in item["image_spans"]] != case["image_ids"]:
            order.append(case["case_id"])
        if item["query_index"] != item["input_token_count"] - 1:
            query.append(case["case_id"])
        try:
            payload, _ = load_validated_feature(run_dir, item, metadata, tolerance["atol"], tolerance["rtol"],
                                                expected_images=None)
            if case["condition"] == "one_negative" and case["placement"] == "N-first":
                features[item["index"]] = payload["image_features"][:, 0].clone()
            del payload
        except (ValueError, KeyError, RuntimeError, TypeError) as error:
            load.append(f"{case['case_id']}: {error}")
    record("cases_identical_to_frozen_cases_file", not link, link[:20] or None)
    record("image_files_match_ids", not files, files[:20] or None)
    record("image_token_spans_follow_image_order", not order, order[:20] or None)
    record("h_q_is_last_valid_input_token", not query, query[:20] or None)
    record("features_recomputed_and_linked", not load, load[:20] or None)

    tokens, step4_tokens = defaultdict(set), defaultdict(set)
    for item in records:
        for span in item["image_spans"]:
            tokens[span["image_id"]].add(span["token_count"])
    for item in read_jsonl(project_path(STEP4_RUN) / "manifest.jsonl"):
        for span in item["image_spans"]:
            step4_tokens[span["image_id"]].add(span["token_count"])
    varying = {i: sorted(c) for i, c in tokens.items() if len(c) > 1}
    differs = {i: [sorted(c), sorted(step4_tokens[i])] for i, c in tokens.items() if c != step4_tokens[i]}
    record("image_token_count_identical_across_inputs", not varying, varying or None)
    record("image_token_count_identical_to_step4", not differs, differs or None)

    identity_rows, exceeding, identity = ([], [], {}) if load else n_first_identity(records, features, config, probes, layers)
    within = [row for row in exceeding if row["against"] == "series_two_image_input"]
    record("n_first_within_bf16_tolerance", not within,
           {"criterion": f"cosine distance <= {config['integrity']['n_first_max_cosine_distance']} at every layer",
            **identity, "exceeding_inputs": [row["case_id"] for row in within][:50]})

    parsed = sum(item["prediction"] in ("Yes", "No") for item in records)
    agreement = {margin: {"agree": sum((item[margin] > 0 and item["prediction"] == "Yes")
                                       or (item[margin] < 0 and item["prediction"] == "No") for item in records),
                          "ties": sum(item[margin] == 0 for item in records), "n": len(records)} for margin in MARGINS}
    record("answers_parsed_reported", True, {"parsed": parsed, "unparsed": len(records) - parsed, "n": len(records)})
    record("margin_sign_agreement_reported", True, agreement)
    gap = max(abs(item["fp32_logit_margin"] - item["logit_margin"]) for item in records)
    record("fp32_margin_close_to_bf16", gap < 0.5, {"max_abs_difference": gap})

    times = sorted((os.path.getmtime(run_dir / item["feature_file"]), item["case"]["n_images"]) for item in records)
    per_count = defaultdict(list)
    for (start, _), (end, count) in zip(times, times[1:]):
        per_count[count].append(end - start)
    timing = {count: statistics.median(values) for count, values in sorted(per_count.items())}
    changed = verify_accumulation_freeze(manifest)
    record("frozen_files_unchanged", not changed, changed or None)
    report = {
        "check": f"step5_accumulation_{kind}_integrity", "note": NOTE,
        "status": "passed" if not problems else "failed", "run_dir": project_relative_path(run_dir),
        "frozen_dir": project_relative_path(frozen_dir), "frozen_version": manifest["version"],
        "records": len(records), "n_images": sorted({item["case"]["n_images"] for item in records}),
        "conditions": sorted({item["case"]["condition"] for item in records}),
        "placements": sorted({item["case"]["placement"] for item in records}),
        "strengths": sorted({item["case"]["strength"] for item in records}),
        "peak_gpu_memory_gb": metadata.get("peak_gpu_memory_bytes", 0) / 1e9,
        "median_seconds_per_case": timing, "checks": results, "failed": problems,
    }
    if kind == "pilot":
        report["estimated_main_run_hours"] = sum(timing.get(case["n_images"], 0) for case in frozen_cases) / 2 / 3600
        report["estimate_note"] = "pilot time includes a repeated pass and reference hidden states; halved as a rough estimate"
    return report, identity_rows, exceeding
