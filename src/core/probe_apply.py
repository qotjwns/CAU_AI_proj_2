"""STEP 4: 고정된 z_i Probe와 h_q readout을 독립 데이터의 STEP 2 추출 결과에 적용해 평가한다.

재학습·재표준화·임계값 조정 없음. 사전 지정 내용과 임계값은 설정 파일이 아니라 고정 파일에서 읽는다.
"""
from datetime import datetime, timezone

import torch

from src.common.data import read_json
from src.common.paths import project_path, project_relative_path
from src.core.feature_analysis import extraction_signature
from src.core.feature_validation import require
from src.core.hq_probe import SHORTCUT_NOTE, predict_hq, readability_rows
from src.core.probe import predict
from src.core.probe_analysis import DEFINITIONS, evaluate_predictions
from src.core.probe_cases import (
    ImageBootstrap, constant_baselines, evaluate_layers, group_comparison, localization, paired_feature_comparison,
)
from src.core.probe_data import build_vectors, load_slots
from src.core.probe_freeze import signature_hash, verify_frozen


INDEPENDENCE_SCOPE = (
    "Every STEP 4 image is disjoint from all images used to train the frozen z_i probes and h_q readouts (checked). "
    "Weights, standardization, threshold, layers, positions and scopes were fixed before STEP 4 extraction and are "
    "not tuned on STEP 4 data. z_i and h_q are evaluated on the same inputs."
)
COMPARED_METRICS = (
    "image_accuracy", "fy_negative_detection_rate", "strict_discrepancy_rate", "fy_correction_rate",
    "ppp_retention_rate", "probe_and_accuracy", "vlm_accuracy",
)


def load_frozen(frozen_dir):
    manifest = read_json(frozen_dir / "manifest.json")
    require(manifest["status"] == "frozen", "Probe manifest is not marked as frozen")
    changed = verify_frozen(frozen_dir, manifest)
    require(not changed, f"{changed} differ from the frozen manifest (sha256 mismatch)")
    zi = torch.load(frozen_dir / "zi_probes.pt", map_location="cpu", weights_only=True)
    hq = torch.load(frozen_dir / "hq_readouts.pt", map_location="cpu", weights_only=True)
    require(zi["threshold"] == hq["threshold"] == manifest["threshold"], "Frozen thresholds differ")
    require(zi["extraction_signature"] == hq["extraction_signature"], "Frozen extraction signatures differ")
    return manifest, zi, hq


def check_extraction_source(metadata, manifest, full_run=True):
    """추출에 쓴 케이스 파일이 고정 시점의 평가 케이스와 같은지, (본 평가라면) 전체 실행인지 확인한다."""
    evaluation = manifest["evaluation"]
    sources = metadata["sources"]
    require(len(sources) == 1 and sources[0]["sha256"] == evaluation["cases_sha256"],
            "Extraction used cases that differ from the frozen evaluation cases")
    if full_run:
        require(metadata["limit"] is None and metadata["category"] is None,
                "This extraction is a pilot or subset (--limit/--category); only the full run can be evaluated")
        require(metadata["selected_cases"] == metadata["written_cases"] == evaluation["records"],
                "Extraction does not cover every frozen evaluation case")


def signature_differences(current, frozen):
    keys = sorted(set(current) | set(frozen))
    differences = [key for key in keys if key != "model_config" and current.get(key) != frozen.get(key)]
    model_keys = sorted(set(current["model_config"]) | set(frozen["model_config"]))
    differences += [f"model_config.{key}" for key in model_keys
                    if current["model_config"].get(key) != frozen["model_config"].get(key)]
    return differences


def step3_comparison(step3_report, key_rows, group_rows, layers):
    """STEP 3(OOF, 탐색적)과 STEP 4(고정 Probe, 독립 이미지)의 z_i 핵심 지표를 나란히 둔다."""
    if not step3_report:
        return []
    old_key = {row["layer"]: row for row in step3_report["key_table"] if row["level"] == "unique_input"}
    new_key = {row["layer"]: row for row in key_rows if row["level"] == "unique_input"}
    rows = []
    for layer in layers:
        for metric in COMPARED_METRICS:
            new = new_key[layer]
            rows.append({
                "layer": layer, "metric": metric, "step3": old_key.get(layer, {}).get(metric),
                "step4": new.get(metric), "step4_ci_low": new.get(f"{metric}_ci_low"),
                "step4_ci_high": new.get(f"{metric}_ci_high"),
            })
        for analysis, metric in (("matched_class_composition", "strict"), ("paired_negative_image", "strict"),
                                 ("paired_negative_image", "negative_detection")):
            def find(source):
                return next((row for row in source if row["level"] == "unique_input" and row["layer"] == layer
                             and row["analysis"] == analysis and row["metric"] == metric), {})
            old, new = find(step3_report.get("group_comparison", [])), find(group_rows)
            rows.append({
                "layer": layer, "metric": f"{analysis}:{metric}:difference", "step3": old.get("difference"),
                "step4": new.get("difference"), "step4_ci_low": new.get("ci_low"), "step4_ci_high": new.get("ci_high"),
            })
    return rows


def hq_evaluation(unique, predictions, queries, hq, layers, threshold, bootstrap, preregistration):
    """같은 고유 입력에서 z_i와 h_q(통합·클래스별)를 평가한다. 슬롯 index는 3 × 입력 + 위치."""
    cases = [{**case, "vectors": [3 * i, 3 * i + 1, 3 * i + 2]} for i, case in enumerate(unique)]
    slot_index = torch.tensor([index for case in unique for index in case["vectors"]])
    features = {"z_i": predictions[slot_index]}
    for scope in hq["scopes"]:
        features[f"h_q_{scope}"] = predict_hq(cases, queries, hq["readouts"], layers, scope)
    readability = readability_rows(cases, None, features, layers, threshold)
    groups, located = [], []
    for feature, values in features.items():
        groups += [{"feature": feature, **row}
                   for row in group_comparison({"unique_input": cases}, values, layers, threshold, bootstrap)]
        located += [{"feature": feature, **row} for row in localization(cases, values, layers, bootstrap)]
    column = {layer: i for i, layer in enumerate(layers)}
    zi_layer = preregistration["z_i"]["primary_layer"]
    hq_layer = preregistration["h_q"]["primary_layer"]
    primary_scope = f"h_q_{preregistration['h_q']['primary_scope']}"
    pairs = [("primary", ("z_i", zi_layer), (primary_scope, hq_layer))]
    pairs += [("same_layer", ("z_i", layer), (primary_scope, layer)) for layer in layers]
    pairs += [("h_q_scope", (primary_scope, hq_layer), (f"h_q_{scope}", hq_layer))
              for scope in preregistration["h_q"]["secondary_scopes"]]
    paired = []
    for kind, (first, first_layer), (second, second_layer) in pairs:
        rows = paired_feature_comparison(
            cases, (first, features[first], column[first_layer], first_layer),
            (second, features[second], column[second_layer], second_layer), threshold, bootstrap)
        paired += [{"comparison": kind, **row} for row in rows]
    return {
        "shortcut_note": SHORTCUT_NOTE, "readability": readability, "group_comparison": groups,
        "localization": located, "paired": paired,
    }, cases, features


def apply_frozen_probe(run_dir, config, step3_report=None):
    frozen_dir = project_path(config["frozen_dir"])
    manifest, zi, hq = load_frozen(frozen_dir)
    preregistration = manifest["preregistration"]
    require(config["preregistration"] == preregistration,
            "configs preregistration differs from the frozen manifest; STEP 4 must use the frozen preregistration")
    verification = config["verification"]
    metadata, records, features, slots, queries = load_slots(
        run_dir, verification["atol"], verification["rtol"], with_query=True)
    check_extraction_source(metadata, manifest)
    signature = extraction_signature(metadata)
    differences = signature_differences(signature, zi["extraction_signature"])
    require(not differences, f"Extraction settings differ from the frozen probes' training data: {differences}")
    layers = signature["layers"]
    require(layers == zi["layers"] == hq["layers"], "Layer selection differs from the frozen probes")
    overlap = {slot["image_id"] for slot in slots} & set(zi["training_image_ids"])
    require(not overlap, f"{len(overlap)} evaluation images were used to train the frozen probes")
    threshold = manifest["threshold"]
    statistics = preregistration["statistics"]
    settings = {"bootstrap_reps": statistics["reps"], "seed": statistics["seed"],
                "examples_per_category": config["examples_per_category"]}

    vectors, matrix = build_vectors(slots, features, config["duplicate_cosine_tolerance"])
    del features
    for vector in vectors:
        vector.update(fold=None, context="pure")
    for slot in slots:
        slot.update(fold=None, context="pure", prediction=records[slot["record"]]["prediction"])
    probes = {(probe["class"], probe["layer"]): probe for probe in zi["probes"]}
    missing = sorted({vector["class"] for vector in vectors} - {name for name, _ in probes})
    require(not missing, f"No frozen z_i probe for classes: {missing}")
    predictions = torch.full((len(vectors), len(layers)), float("nan"), dtype=torch.float64)
    for name in sorted({vector["class"] for vector in vectors}):
        members = [vector["vector"] for vector in vectors if vector["class"] == name]
        for column, layer in enumerate(layers):
            predictions[members, column] = predict(probes[(name, layer)], matrix[members, column])
    require(not bool(torch.isnan(predictions).any()), "Some context vectors received no prediction")

    evaluation = evaluate_predictions(records, slots, vectors, predictions, layers, threshold, settings)
    unique = evaluation["unique"]
    observed = evaluate_layers(unique, vectors, predictions, layers, threshold)
    bootstrap = ImageBootstrap([vector["image_id"] for vector in vectors], statistics["reps"], statistics["seed"])
    hq_report, hq_cases, hq_features = hq_evaluation(
        unique, predictions, queries[[case["record"] for case in unique]], hq, layers, threshold, bootstrap,
        preregistration)
    sections = evaluation["sections"]
    extracted_at = datetime.fromtimestamp((run_dir / "run.json").stat().st_mtime, timezone.utc).isoformat(
        timespec="seconds")
    report = {
        "status": config["status"], "independence_scope": INDEPENDENCE_SCOPE,
        "definitions": {**DEFINITIONS, "context_pure": "STEP 4: every image in the input is unseen by the frozen probes."},
        "preregistration": {
            **preregistration, "primary_layer": preregistration["z_i"]["primary_layer"],
            "threshold": threshold, "frozen_dir": project_relative_path(frozen_dir),
            "frozen_version": manifest["version"], "frozen_files_verified_before_evaluation": True,
            "frozen_files_sha256": manifest["files"], "frozen_configs_sha256": manifest["configs"],
            "evaluation_cases": manifest["evaluation"],
            "frozen_created_at": manifest["created_at"], "extraction_run_json_modified_at": extracted_at,
            "frozen_before_extraction_finished": manifest["created_at"] < extracted_at,
            "extraction_signature_sha256": signature_hash(signature),
        },
        "run_dir": project_relative_path(run_dir), "extraction_settings": signature,
        "source": {"versions": metadata["versions"], "device": metadata["device"],
                   "attention_implementation": metadata["attention_implementation"]},
        "settings": {**settings, "duplicate_cosine_tolerance": config["duplicate_cosine_tolerance"],
                     "threshold": threshold, "probe_settings": manifest["probe_settings"]},
        "counts": {
            "records": len(records), "slots": len(slots), "context_vectors": len(vectors),
            "images": len({vector["image_id"] for vector in vectors}),
            "classes": len({vector["class"] for vector in vectors}),
            "unique_inputs": len(unique),
            "unique_inputs_with_inconsistent_prediction": evaluation["inconsistent"],
            "training_image_overlap": 0,
            "max_duplicate_cosine_distance": max(vector["max_duplicate_cosine_distance"] for vector in vectors),
        },
        **sections,
        "baselines": constant_baselines(observed, unique, vectors, layers),
        "step3_comparison": step3_comparison(step3_report, sections["key_table"], sections["group_comparison"], layers),
        "hq": hq_report,
    }
    return {
        "report": report, "manifest": manifest, "frozen_dir": frozen_dir,
        "vectors": vectors, "slots": slots, "predictions": predictions, "layers": layers,
        "strict_cases": evaluation["strict_cases"], "examples": evaluation["examples"],
        "hq_cases": hq_cases, "hq_features": hq_features,
    }
