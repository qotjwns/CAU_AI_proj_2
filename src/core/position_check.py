"""STEP 5-2b 추출 무결성 검사 (CPU): 파일럿과 본 실행에 공통.

구조·연결·설정·실행 환경 기록·토큰 수·prefix 동일성·답 파싱·margin 부호 일치·시간만 확인한다.
False-Yes 비율, 위치 대비 등 평가 지표는 계산하지도 출력하지도 않는다.

prefix 동일성: 이미지 토큰이 질문보다 앞에 있으므로 N의 표현 z_N은 N 앞의 이미지(P1..Pa)에만 의존한다.
- 같은 실행 안: 같은 계열·같은 a의 입력끼리 z_N이 같아야 한다 (코사인 거리 기준, 판정을 막는 검사).
- 실행 간: 5-2의 같은 prefix 입력(a = 0: N-first, a ≥ 1: N-last k = a)과 비교해 보고한다. 기준을 넘으면 실행 환경
  차이로 표시하고 평가는 막지 않는다 (확증 대비는 5-2b 한 실행 안에서 계산).
"""
from collections import defaultdict
import os
import statistics

from src.common.data import read_json, read_jsonl
from src.common.paths import project_path, project_relative_path
from src.core.accumulation_check import MARGINS, representation_gap
from src.core.accumulation_freeze import verify_accumulation_freeze
from src.core.feature_validation import inspect_run, load_validated_feature
from src.core.probe_freeze import verify_frozen


NOTE = ("Integrity check only: structure, linkage, settings, recorded environment, token counts, prefix identity of "
        "z_N, answer parsing, margin sign agreement and timing. No False-Yes rate, position contrast or other "
        "evaluation metric is computed.")
SAME_KEYS = ("image_processor", "model_config", "chat_template", "dtype", "attention_implementation", "pooling")
STEP4_RUN = "results/step4/qwen2_5_vl_3b/extraction"


def summarize_gaps(rows, tolerance):
    if not rows:
        return {"n": 0}
    return {"n": len(rows), "exceeding": sum(row["max_cosine_distance"] > tolerance for row in rows),
            "max_cosine_distance": max(row["max_cosine_distance"] for row in rows),
            "median_cosine_distance": statistics.median(row["max_cosine_distance"] for row in rows),
            "max_relative_l2": max(row["max_relative_l2"] for row in rows)}


def gap_row(kind, item, reference_id, values, reference, layers):
    cosine, relative = representation_gap(values, reference)
    case = item["case"]
    return {"kind": kind, "case_id": case["case_id"], "reference": reference_id, "series": case["pair_id"],
            "n_before": case["n_before"], "n_images": case["n_images"], "role": case["role"],
            "max_cosine_distance": max(cosine), "max_relative_l2": max(relative),
            **{f"cosine_distance_L{layer}": value for layer, value in zip(layers, cosine)}}


class Step52References:
    """5-2 본 추출에서 같은 prefix를 가진 입력의 z_N (a = 0: N-first k = 1, a ≥ 1: N-last k = a)."""

    def __init__(self, run_dir, tolerance):
        self.run_dir = project_path(run_dir).resolve()
        self.metadata, records, checks, _ = inspect_run(self.run_dir)
        if not all(checks.values()):
            raise ValueError("STEP 5-2 extraction folder is invalid")
        self.items = {}
        for item in records:
            case = item["case"]
            if case["condition"] != "one_negative":
                continue
            if case["placement"] == "N-first" and case["series_step"] == 1:
                self.items[(case["pair_id"], 0)] = item
            elif case["placement"] == "N-last":
                self.items[(case["pair_id"], case["series_step"])] = item
        self.tolerance = tolerance

    def z_n(self, series, a):
        """(5-2 case_id, z_N [레이어, 차원]). 해당 prefix 입력이 5-2에 없으면 KeyError."""
        item = self.items[(series, a)]
        payload, _ = load_validated_feature(self.run_dir, item, self.metadata, self.tolerance["atol"],
                                            self.tolerance["rtol"], expected_images=None)
        return item["case"]["case_id"], payload["image_features"][:, item["case"]["negative_positions"][0]].clone()


def check_run(run_dir, config, kind):
    frozen_dir = project_path(config["frozen_dir"])
    manifest = read_json(frozen_dir / "manifest.json")
    tolerance = config["verification"]
    limit = config["integrity"]["prefix_max_cosine_distance"]
    metadata, records, checks, inventory = inspect_run(run_dir)
    accumulation = config["accumulation"]
    step52_meta = read_json(project_path(accumulation["run_dir"]) / "run.json")
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
    same = {key: metadata[key] == step52_meta[key] for key in SAME_KEYS}
    record("same_model_resolution_precision_as_5_2", all(same.values()), same)
    environment = metadata.get("environment") or {}
    record("environment_recorded", bool(environment.get("gpus")) and environment.get("nvidia_smi") is not None
           and environment.get("same_gpu_at_end") is True,
           {"gpus": [gpu.get("name") for gpu in environment.get("gpus", [])],
            "drivers": sorted({row.get("driver_version") for row in environment.get("nvidia_smi") or []}),
            "torch": environment.get("torch"), "transformers": (environment.get("packages") or {}).get("transformers"),
            "versions_5_2": step52_meta.get("versions")})
    sources = {source["path"]: source["sha256"] for source in metadata["sources"]}
    record("cases_file_hash_matches_freeze",
           sources.get(config["cases"]) == manifest["files_sha256"][config["cases"]], sources)
    frozen_cases = list(read_jsonl(project_path(config["cases"])))
    by_id = {case["case_id"]: case for case in frozen_cases}
    if kind == "main":
        record("all_frozen_cases_in_order", metadata["limit"] is None
               and [item["case"] for item in records] == frozen_cases, {"records": len(records), "frozen": len(frozen_cases)})

    layers = settings["layers"]
    references = Step52References(accumulation["run_dir"], tolerance)
    link, files, order, query, load, missing_reference = [], [], [], [], [], []
    within_rows, cross_rows = [], []
    series = defaultdict(list)
    for item in records:
        series[item["case"]["pair_id"]].append(item)
    for name, items in series.items():
        z_n = {}
        for item in items:
            case = item["case"]
            if by_id.get(case["case_id"]) != case:
                link.append(case["case_id"])
            for file_name, image_id in zip(case["image_order"], case["image_ids"]):
                if file_name != f"{image_id:012d}.jpg" or not (project_path(case["image_dir"]) / file_name).exists():
                    files.append(case["case_id"])
            if [span["image_id"] for span in item["image_spans"]] != case["image_ids"]:
                order.append(case["case_id"])
            if item["query_index"] != item["input_token_count"] - 1:
                query.append(case["case_id"])
            try:
                payload, _ = load_validated_feature(run_dir, item, metadata, tolerance["atol"], tolerance["rtol"],
                                                    expected_images=None)
            except (ValueError, KeyError, RuntimeError, TypeError) as error:
                load.append(f"{case['case_id']}: {error}")
                continue
            z_n[case["case_id"]] = payload["image_features"][:, case["n_before"]].clone()
            del payload
        groups = defaultdict(list)
        for item in items:
            if item["case"]["case_id"] in z_n:
                groups[item["case"]["n_before"]].append(item)
        for a, members in groups.items():
            reference_item = members[0]
            reference = z_n[reference_item["case"]["case_id"]]
            for item in members[1:]:
                within_rows.append(gap_row("within_run_same_prefix", item, reference_item["case"]["case_id"],
                                           z_n[item["case"]["case_id"]], reference, layers))
            try:
                reference_id, step52 = references.z_n(name, a)
            except KeyError:
                missing_reference.append(f"{name} a={a}")
                continue
            for item in members:
                cross_rows.append(gap_row("cross_run_vs_5_2", item, reference_id, z_n[item["case"]["case_id"]],
                                          step52, layers))
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

    within = summarize_gaps(within_rows, limit)
    record("prefix_identity_within_run", not load and within.get("exceeding", 0) == 0,
           {"criterion": f"cosine distance <= {limit} at every layer", **within,
            "exceeding_inputs": [r["case_id"] for r in within_rows if r["max_cosine_distance"] > limit][:50]})
    cross = summarize_gaps(cross_rows, limit)
    record("prefix_identity_across_runs_reported", True, {
        "criterion": f"cosine distance <= {limit} at every layer (reported; does not block evaluation)", **cross,
        "environment_difference": cross.get("exceeding", 0) > 0, "missing_5_2_references": missing_reference[:20]})

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
    step52_changed = verify_accumulation_freeze(read_json(project_path(accumulation["frozen_dir"]) / "manifest.json"))
    zi_dir = project_path(config["zi_frozen_dir"])
    zi_changed = verify_frozen(zi_dir, read_json(zi_dir / "manifest.json"))
    record("frozen_files_unchanged", not (changed or step52_changed or zi_changed),
           {"step5_2b": changed, "step5_2": step52_changed, "step4": zi_changed})
    report = {
        "check": f"step5_2b_position_{kind}_integrity", "note": NOTE,
        "status": "passed" if not problems else "failed", "run_dir": project_relative_path(run_dir),
        "frozen_dir": project_relative_path(frozen_dir), "frozen_version": manifest["version"],
        "records": len(records), "n_images": sorted({item["case"]["n_images"] for item in records}),
        "roles": sorted({item["case"]["role"] for item in records}),
        "positions": sorted({(item["case"]["n_images"], item["case"]["n_before"]) for item in records}),
        "peak_gpu_memory_gb": metadata.get("peak_gpu_memory_bytes", 0) / 1e9,
        "median_seconds_per_case": timing, "checks": results, "failed": problems,
    }
    if kind == "pilot":
        report["estimated_main_run_hours"] = sum(timing.get(case["n_images"], 0) for case in frozen_cases) / 2 / 3600
        report["estimate_note"] = "pilot time includes a repeated pass and reference hidden states; halved as a rough estimate"
    return report, within_rows + cross_rows
