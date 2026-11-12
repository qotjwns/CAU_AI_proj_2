"""STEP 5-2 사전 등록 고정 (CPU).

케이스·설계·일관성 검사·설정·추출 코드의 해시와 사전 등록 내용을 고정한다. 본 추출 전에 실행하며, 이미 고정된
버전은 덮어쓰지 않는다. 이전 버전이 있으면 설정의 revision(버전, 이유, 변경 항목)이 있어야 새 버전을 만든다.
"""
from datetime import datetime, timezone

from src.common.data import read_json, read_jsonl, write_json
from src.common.paths import project_path, project_relative_path
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file, verify_frozen


def verify_accumulation_freeze(manifest):
    """고정 이후 바뀐 파일 목록 (프로젝트 기준 경로)."""
    return [path for path, digest in manifest["files_sha256"].items() if sha256_file(project_path(path)) != digest]


def earlier_versions(frozen_dir):
    return sorted(path for path in frozen_dir.parent.glob("*/manifest.json") if path.parent != frozen_dir)


def freeze_accumulation(config_path, config):
    frozen_dir = project_path(config["frozen_dir"])
    require(not (frozen_dir / "manifest.json").exists(),
            f"Already frozen: {frozen_dir}. Frozen versions are never overwritten.")
    preregistration = config["preregistration"]
    require(preregistration["minimum_effect"]["confirmed"] is True,
            "The minimum meaningful effect has not been confirmed by the user")
    require(not (project_path(config["run_dir"]) / "run.json").exists(),
            "The main extraction already has model output; freeze must come before it")
    earlier = earlier_versions(frozen_dir)
    revision = config.get("revision")
    if earlier:
        require(revision and revision.get("reason") and revision.get("changes") and revision.get("version") == frozen_dir.name
                and revision.get("made_before_main_extraction") is True,
                "Earlier frozen versions exist; the config needs a revision block (version, reason, changes)")
    else:
        require(not (project_path(config["pilot_dir"]) / "run.json").exists(),
                "Pilot output exists without an earlier frozen version; freeze must come before the pilot")
    check = read_json(project_path(config["consistency_check"]))
    require(check["status"] == "passed", "The case consistency check has not passed")
    zi_dir = project_path(config["zi_frozen_dir"])
    zi_manifest = read_json(zi_dir / "manifest.json")
    changed = verify_frozen(zi_dir, zi_manifest)
    require(not changed, f"STEP 4 frozen probe files changed: {changed}")

    paths = [config[key] for key in ("cases", "consistency_check", "design", "used_images", "case_config",
                                     "extract_config")]
    if "vision_fp32" in config:
        paths += [config["vision_fp32"]["cases"], config["vision_fp32"]["extract_config"]]
    paths += [project_relative_path(config_path), *config["code_files"],
              project_relative_path(zi_dir / "manifest.json")]
    cases = list(read_jsonl(project_path(config["cases"])))
    pilot_run = project_path(config["pilot_dir"]) / "run.json"
    manifest = {
        "status": "frozen", "version": frozen_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Frozen before the main STEP 5-2 extraction. The evaluation must use this preregistration; a change "
                 "requires a new version with a recorded reason."),
        "revision": revision if earlier else None,
        "previous_versions": [{"frozen_dir": project_relative_path(path.parent), "manifest_sha256": sha256_file(path)}
                              for path in earlier],
        "preregistration": preregistration,
        "integrity": config.get("integrity"),
        "files_sha256": {path: sha256_file(project_path(path)) for path in dict.fromkeys(paths)},
        "counts": {"records": len(cases), "unique_inputs": len({(c["prompt"], tuple(c["image_ids"])) for c in cases}),
                   "negatives": len({(c["class"], c["anchor_negative_id"]) for c in cases}),
                   "series": len({c["pair_id"] for c in cases})},
        "zi_frozen": {"dir": project_relative_path(zi_dir), "version": zi_manifest["version"]},
        "pilot": {"dir": config["pilot_dir"], "limit": config["pilot_limit"],
                  "existed_at_freeze": pilot_run.exists(),
                  "run_json_sha256": sha256_file(pilot_run) if pilot_run.exists() else None,
                  "policy": (revision or {}).get("pilot_policy",
                                                 "Integrity check only; not part of any analysis (as in STEP 4).")},
    }
    write_json(manifest, frozen_dir / "manifest.json", allow_nan=False)
    return frozen_dir, manifest
