"""STEP 5-2b 사전 등록 고정 (CPU).

케이스·일관성 검사·설정·추출/검사 코드·재사용하는 5-2 기록의 해시와 사전 등록 내용을 고정한다. 파일럿·본 추출 전에
실행하며, 이미 고정된 버전은 덮어쓰지 않는다. 이전 버전이 있으면 설정의 revision(버전, 이유, 변경 항목)이 있어야 한다.
"""
from datetime import datetime, timezone

from src.common.data import read_json, read_jsonl, write_json
from src.common.paths import project_path, project_relative_path
from src.core.accumulation_freeze import earlier_versions, verify_accumulation_freeze
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file, verify_frozen


def freeze_position(config_path, config):
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
    cases_path = project_path(config["cases"])
    check = read_json(project_path(config["consistency_check"]))
    require(check["status"] == "passed", "The case consistency check has not passed")
    require(check["cases_sha256"] == sha256_file(cases_path), "Cases changed after the consistency check")
    zi_dir = project_path(config["zi_frozen_dir"])
    zi_manifest = read_json(zi_dir / "manifest.json")
    zi_changed = verify_frozen(zi_dir, zi_manifest)
    require(not zi_changed, f"STEP 4 frozen probe files changed: {zi_changed}")
    accumulation = config["accumulation"]
    step52_dir = project_path(accumulation["frozen_dir"])
    step52_manifest = read_json(step52_dir / "manifest.json")
    step52_changed = verify_accumulation_freeze(step52_manifest)
    require(not step52_changed, f"STEP 5-2 frozen files changed: {step52_changed}")
    step52_run = project_path(accumulation["run_dir"])
    step52_records = [step52_run / "manifest.jsonl", step52_run / "run.json", step52_run / "main_check.json",
                      project_path(accumulation["evaluation"])]

    paths = [config[key] for key in ("cases", "consistency_check", "case_config", "extract_config")]
    paths += [project_relative_path(config_path), *config["code_files"], project_relative_path(zi_dir / "manifest.json"),
              project_relative_path(step52_dir / "manifest.json"), *[project_relative_path(p) for p in step52_records]]
    cases = list(read_jsonl(cases_path))
    roles = {}
    for case in cases:
        roles[case["role"]] = roles.get(case["role"], 0) + 1
    manifest = {
        "status": "frozen", "version": frozen_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Frozen before the STEP 5-2b pilot and main extraction. The evaluation must use this "
                 "preregistration; a change requires a new version with a recorded reason."),
        "revision": revision if earlier else None,
        "previous_versions": [{"frozen_dir": project_relative_path(path.parent), "manifest_sha256": sha256_file(path)}
                              for path in earlier],
        "preregistration": preregistration,
        "integrity": config.get("integrity"),
        "files_sha256": {path: sha256_file(project_path(path)) for path in dict.fromkeys(paths)},
        "counts": {"records": len(cases), "unique_inputs": len({(c["prompt"], tuple(c["image_ids"])) for c in cases}),
                   "series": len({c["pair_id"] for c in cases}),
                   "n_images_anchor": len({c["anchor_negative_id"] for c in cases}), "by_role": roles},
        "zi_frozen": {"dir": project_relative_path(zi_dir), "version": zi_manifest["version"]},
        "accumulation": {"frozen_dir": project_relative_path(step52_dir), "version": step52_manifest["version"],
                         "observed_before": preregistration["observed_before"]["note"]},
        "pilot": {"dir": config["pilot_dir"], "limit": config["pilot_limit"], "existed_at_freeze": False,
                  "policy": "Integrity check only; not part of any analysis (as in STEP 4 and 5-2)."},
    }
    write_json(manifest, frozen_dir / "manifest.json", allow_nan=False)
    return frozen_dir, manifest
