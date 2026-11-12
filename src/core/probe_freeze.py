"""STEP 4 GPU 추출 전에 z_i Probe와 h_q readout을 학습·고정한다 (CPU)."""
from datetime import datetime, timezone
import hashlib
import json

import torch

from src.common.data import read_json, read_jsonl, write_csv, write_json
from src.common.paths import project_path, project_relative_path
from src.core.feature_analysis import extraction_signature
from src.core.feature_validation import require
from src.core.hq_probe import FEATURE_DEFINITION, POSITION_DEFINITION, fit_final_hq
from src.core.probe import fit_final
from src.core.probe_analysis import case_table, unique_cases
from src.core.probe_data import build_vectors, load_slots


RULE = "Do not retrain, re-standardize, change thresholds, layers, positions or scopes after STEP 4 data is inspected."


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def signature_hash(signature):
    return sha256_bytes(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode())


ZI_ROLE = "Frozen per-class image-feature probes for H1-A (observational)."
HQ_ROLE = ("Frozen h_q readouts for H1-B (observational diagnostic). Not the STEP 5 intervention decision readout "
           "s_i = f(h_q, z_i); readability here does not validate any intervention direction.")


def verify_frozen(frozen_dir, manifest):
    """manifest에 기록된 고정 파일들의 SHA-256을 다시 계산해 비교한다. 다른 파일 이름 목록을 반환."""
    return [name for name, digest in manifest["files"].items() if sha256_file(frozen_dir / name) != digest]


def training_file_hashes(run_dir, records):
    """학습 데이터 원본(run.json, manifest.jsonl, 모든 특징 .pt)의 상대경로와 SHA-256."""
    names = ["run.json", "manifest.jsonl", *[record["feature_file"] for record in records]]
    return [(sha256_file(run_dir / name), name) for name in names]


def previous_versions(output_dir):
    siblings = sorted(path for path in output_dir.parent.glob("*/manifest.json") if path.parent != output_dir)
    return [{"frozen_dir": project_relative_path(path.parent), "manifest_sha256": sha256_file(path)} for path in siblings]


def evaluation_lock(config):
    """평가 케이스 파일 해시와 Annotation Consistency Check 통과 여부를 확인해 고정한다."""
    cases_path = project_path(config["evaluation_cases"])
    report_path = project_path(config["consistency_check"])
    require(report_path.exists(), f"Run the annotation consistency check first: {report_path}")
    report = read_json(report_path)
    digest = sha256_file(cases_path)
    require(report["status"] == "passed", "Annotation consistency check did not pass")
    require(report["cases_sha256"] == digest, "Evaluation cases changed after the consistency check")
    return {
        "cases": project_relative_path(cases_path), "cases_sha256": digest,
        "records": sum(1 for line in cases_path.open(encoding="utf-8") if line.strip()),
        "groups": report["groups"], "groups_sha256": report["groups_sha256"],
        "consistency_check": project_relative_path(report_path), "consistency_check_sha256": sha256_file(report_path),
        "consistency_check_note": report["note"],
    }


def freeze_probes(config_path, config, step3_config_path, step3_config, reason=None):
    """config: step4.yaml, step3_config: step3.yaml(Probe 설정·임계값·중복 허용치).

    이미 고정된 버전은 수정하지 않는다. 다른 버전이 있으면 reason(변경 이유)이 있어야 새 버전을 만든다.
    """
    output_dir = project_path(config["frozen_dir"])
    require(not (output_dir / "manifest.json").exists(),
            f"Frozen probes already exist: {output_dir}. They are not overwritten; freeze a new version instead.")
    earlier = previous_versions(output_dir)
    require(not earlier or reason, "Earlier frozen versions exist; pass --reason to record why a new version is needed")
    evaluation = evaluation_lock(config)
    training = config["training"]
    run_dir = project_path(training["run_dir"]).resolve()
    verification = step3_config["verification"]
    metadata, records, features, slots, queries = load_slots(
        run_dir, verification["atol"], verification["rtol"], with_query=True, progress=True)
    signature = extraction_signature(metadata)
    layers = signature["layers"]
    threshold = step3_config["threshold"]
    settings = step3_config["probe"]

    vectors, matrix = build_vectors(slots, features, step3_config["duplicate_cosine_tolerance"])
    del features
    training_images = sorted({vector["image_id"] for vector in vectors})
    step4_images = {image_id for case in read_jsonl(project_path(config["evaluation_cases"])) for image_id in case["image_ids"]}
    require(not step4_images & set(training_images), "Evaluation cases share images with the probe training data")
    zi_probes, zi_rows = fit_final(vectors, matrix, layers, settings, threshold, progress=True)
    unique, _ = unique_cases(case_table(records, slots))
    hq_queries = queries[[case["record"] for case in unique]]
    del queries
    scopes = [config["preregistration"]["h_q"]["primary_scope"], *config["preregistration"]["h_q"]["secondary_scopes"]]
    hq_readouts, hq_rows = fit_final_hq(unique, hq_queries, layers, settings, threshold, scopes, progress=True)

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {
        "status": "frozen", "created_at": created, "threshold": threshold, "layers": layers,
        "probe_settings": settings, "extraction_signature": signature,
        "training_run_dir": project_relative_path(run_dir), "training_image_ids": training_images,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    zi_path, hq_path = output_dir / "zi_probes.pt", output_dir / "hq_readouts.pt"
    torch.save({**common, "kind": "z_i", "role": ZI_ROLE,
                "feature_definition": "Mean of the image's visual tokens (STEP 2 pooling).",
                "training_unit": "context vector (class, ordered prefix image_ids)", "probes": zi_probes}, zi_path)
    torch.save({**common, "kind": "h_q", "role": HQ_ROLE, "feature_definition": FEATURE_DEFINITION,
                "position_definition": POSITION_DEFINITION, "scopes": scopes,
                "training_unit": "unique model input (prompt, ordered image_ids)", "readouts": hq_readouts}, hq_path)
    write_json(training_images, output_dir / "training_image_ids.json")
    hashes = training_file_hashes(run_dir, records)
    (output_dir / "training_files.sha256").write_text("".join(f"{digest}  {name}\n" for digest, name in hashes))
    write_csv(zi_rows, output_dir / "training_zi.csv", list(zi_rows[0]))
    write_csv(hq_rows, output_dir / "training_hq.csv", list(hq_rows[0]))
    files = {name: sha256_file(output_dir / name)
             for name in ("zi_probes.pt", "hq_readouts.pt", "training_image_ids.json", "training_files.sha256")}
    manifest = {
        "status": "frozen", "version": output_dir.name, "created_at": created, "rule": RULE,
        "reason": reason or "initial version", "previous_versions": earlier,
        "files": files,
        "configs": {project_relative_path(config_path): sha256_file(project_path(config_path)),
                    project_relative_path(step3_config_path): sha256_file(project_path(step3_config_path))},
        "training_data": {
            "run_dir": project_relative_path(run_dir), "files_listed": len(hashes),
            "hash_list": "training_files.sha256", "run_json_sha256": hashes[0][0], "manifest_sha256": hashes[1][0],
        },
        "evaluation": evaluation,
        "extraction_signature_sha256": signature_hash(signature),
        "model_name": signature["model_name"], "layers": layers, "threshold": threshold, "probe_settings": settings,
        "training_images": len(training_images), "training_context_vectors": len(vectors),
        "training_unique_inputs": len(unique),
        "zi_probes": len(zi_probes), "zi_not_converged": sum(not row["converged"] for row in zi_rows),
        "hq_readouts": len(hq_readouts), "hq_not_converged": sum(not row["converged"] for row in hq_rows),
        "roles": {"z_i": ZI_ROLE, "h_q": HQ_ROLE},
        "hq_scopes": scopes, "position_definition": POSITION_DEFINITION, "hq_feature_definition": FEATURE_DEFINITION,
        "preregistration": config["preregistration"],
    }
    write_json(manifest, output_dir / "manifest.json", allow_nan=False)
    return output_dir, manifest
