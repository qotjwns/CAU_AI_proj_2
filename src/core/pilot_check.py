"""STEP 4 GPU 파일럿 무결성 검사 (CPU).

프로그램이 실제 데이터에서 올바르게 동작하는지만 확인한다. 정답률, Probe 성능 등 평가 지표는
계산하지도 출력하지도 않는다 (독립 검증 결과를 미리 보지 않기 위해).
"""
from pathlib import Path

import torch

from src.common.data import read_jsonl
from src.common.paths import project_path, project_relative_path
from src.core.feature_analysis import extraction_signature
from src.core.feature_validation import inspect_run, load_validated_feature
from src.core.hq_probe import predict_hq
from src.core.probe import predict
from src.core.probe_analysis import case_table, unique_cases
from src.core.probe_apply import check_extraction_source, load_frozen, signature_differences
from src.core.probe_data import build_vectors, load_slots
from src.core.probe_freeze import verify_frozen


NOTE = ("Pilot integrity check only: structure, linkage and settings. No accuracy, probe performance or other "
        "evaluation metric is computed or reported.")


def check_pilot(run_dir, config):
    frozen_dir = project_path(config["frozen_dir"])
    manifest, zi, hq = load_frozen(frozen_dir)
    metadata, records, checks, inventory = inspect_run(run_dir)
    results, problems = {}, []

    def record(name, passed, detail=None):
        results[name] = {"passed": bool(passed), **({"detail": detail} if detail is not None else {})}
        if not passed:
            problems.append(name)

    record("run_structure", all(checks.values()), {name: ok for name, ok in checks.items() if not ok} or None)
    record("extraction_verification_flags", metadata["verify_repeat"] and metadata["verify_reference"],
           "run the pilot with --verify-repeat --verify-reference")
    record("extraction_verification_passed", metadata.get("verification_passed") is True)
    try:
        check_extraction_source(metadata, manifest, full_run=False)
        record("cases_match_frozen_evaluation_cases", True)
    except ValueError as error:
        record("cases_match_frozen_evaluation_cases", False, str(error))
    differences = signature_differences(extraction_signature(metadata), zi["extraction_signature"])
    record("extraction_settings_match_frozen_probes", not differences, differences or None)

    frozen_cases = {case["case_id"]: case for case in read_jsonl(project_path(manifest["evaluation"]["cases"]))}
    image_dir_errors, link_errors, order_errors, query_errors, load_errors = [], [], [], [], []
    for item in records:
        case = item["case"]
        if frozen_cases.get(case["case_id"]) != case:
            link_errors.append(case["case_id"])
        for name, image_id in zip(case["image_order"], case["image_ids"]):
            if name != f"{image_id:012d}.jpg" or not (project_path(case["image_dir"]) / name).exists():
                image_dir_errors.append(case["case_id"])
        if [span["image_id"] for span in item["image_spans"]] != case["image_ids"]:
            order_errors.append(case["case_id"])
        if item["query_index"] != item["input_token_count"] - 1:
            query_errors.append(case["case_id"])
        try:
            load_validated_feature(run_dir, item, metadata,
                                   config["verification"]["atol"], config["verification"]["rtol"])
        except (ValueError, KeyError, RuntimeError, TypeError) as error:
            load_errors.append(f"{case['case_id']}: {error}")
    record("cases_identical_to_frozen_cases_file", not link_errors, link_errors[:20] or None)
    record("image_files_match_ids", not image_dir_errors, image_dir_errors[:20] or None)
    record("image_token_spans_follow_image_order", not order_errors, order_errors[:20] or None)
    record("h_q_is_last_valid_input_token", not query_errors, query_errors[:20] or None)
    record("features_recomputed_and_linked", not load_errors, load_errors[:20] or None)
    record("vlm_answers_parsed", all(item["prediction"] in ("Yes", "No") for item in records),
           "count only; correctness is not checked")

    if not load_errors:
        _, _, features, slots, queries = load_slots(run_dir, config["verification"]["atol"],
                                                    config["verification"]["rtol"], with_query=True)
        vectors, matrix = build_vectors(slots, features, config["duplicate_cosine_tolerance"])
        probes = {(probe["class"], probe["layer"]): probe for probe in zi["probes"]}
        finite = True
        for name in sorted({vector["class"] for vector in vectors}):
            members = [vector["vector"] for vector in vectors if vector["class"] == name]
            for column, layer in enumerate(zi["layers"]):
                values = predict(probes[(name, layer)], matrix[members, column])
                finite &= bool(torch.isfinite(values).all() and ((values >= 0) & (values <= 1)).all())
        unique, _ = unique_cases(case_table(records, slots))
        for scope in hq["scopes"]:
            values = predict_hq(unique, queries[[case["record"] for case in unique]], hq["readouts"], hq["layers"], scope)
            finite &= bool(torch.isfinite(values).all())
        record("frozen_probes_produce_finite_probabilities", finite, "values are not inspected")
    changed = verify_frozen(frozen_dir, manifest)
    record("frozen_files_unchanged", not changed, changed or None)
    return {
        "check": "step4_pilot_integrity", "note": NOTE, "status": "passed" if not problems else "failed",
        "run_dir": project_relative_path(run_dir), "frozen_dir": project_relative_path(frozen_dir),
        "frozen_version": manifest["version"], "records": len(records),
        "modes": sorted({item["case"]["mode"] for item in records}),
        "compositions": sorted({"".join(item["case"]["image_labels"]) for item in records}),
        "checks": results, "failed": problems,
    }
