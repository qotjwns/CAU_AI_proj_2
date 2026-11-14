"""STEP 6-2 사전 등록 고정 (CPU, STEP 6-2 GPU 출력 전). 고정본은 덮어쓰지 않는다.

python -m experiments.step6_contribution.freeze
→ results/step6/qwen2_5_vl_3b/contribution/frozen/v1/manifest.json

files_sha256: 서버에도 있어야 하는 파일 (설정, 코드, STEP 6 계획 파일, STEP 3 풀 케이스, STEP 5-3 분할·계획 파일).
  GPU 실행 전에 서버에서 다시 확인한다.
local_sha256: 로컬 평가에서만 다시 확인하는 파일 (겹침 검사에 쓴 5-2·5-2b 케이스, 나란히 비교할 5-3B 본 실행 결과).
"""
from datetime import datetime, timezone

from src.common.config import read_yaml
from src.common.data import read_json, write_json
from src.common.paths import project_path, project_relative_path
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file


CONFIG = "configs/experiments/step6_contribution.yaml"


def load_config():
    return read_yaml(project_path(CONFIG))


def changed_files(hashes):
    return [path for path, digest in hashes.items()
            if not project_path(path).exists() or sha256_file(project_path(path)) != digest]


def gpu_outputs(config):
    paths = [project_path(config["validation_output"])]
    paths += [project_path(config["dirs"][stage]) / "run.json" for stage in ("basis", "pilot", "main")]
    return [project_relative_path(path) for path in paths if path.exists()]


def freeze():
    config = load_config()
    frozen_dir = project_path(config["frozen_dir"])
    require(not (frozen_dir / "manifest.json").exists(),
            f"Already frozen: {frozen_dir}. Frozen versions are never overwritten.")
    earlier = sorted(path for path in frozen_dir.parent.glob("*/manifest.json") if path.parent != frozen_dir)
    revision = config.get("revision")
    if earlier:
        require(revision and revision.get("version") == frozen_dir.name and revision.get("reason")
                and revision.get("changes"), "Earlier frozen versions exist; the config needs a revision block")
    require(config["status"] == "confirmatory", "Only a confirmatory config can be frozen")
    preregistration = config["preregistration"]
    require(preregistration["minimum_effect"]["confirmed"] is True, "The minimum effect has not been confirmed")
    from experiments.step6_contribution.conditions import design
    require(config["design"] == design(), "Config design differs from the code constants (conditions.design)")
    existing = gpu_outputs(config)
    require(not existing, f"STEP 6-2 GPU output already exists ({existing}); freeze must come first")
    plan_check_path = project_path(config["plan_dir"]) / "plan_check.json"
    check = read_json(plan_check_path)
    require(check["status"] == "passed", "The STEP 6 plan check has not passed")
    require(not changed_files(check["files_sha256"]), "STEP 6 plan files changed after the plan check")
    require(not changed_files(check["sources_sha256"]), "Plan sources changed after the plan check")
    server = [CONFIG, *config["code_files"], project_relative_path(plan_check_path), *check["files_sha256"],
              *config["pool_cases"], config["split_config"],
              *[path for path in check["sources_sha256"] if path.startswith("data/step5_causal/")]]
    local = [path for path in check["sources_sha256"] if path not in server]
    comparison = config["comparison_5_3b"]
    local += [f"{comparison['main_dir']}/{name}" for name in comparison["files"]]
    manifest = {
        "status": "frozen", "experiment": "step6_contribution", "version": frozen_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Frozen before any STEP 6-2 GPU output (real 3B validation, basis, pilot, main). The evaluation must "
                 "use this preregistration; a change requires a new version with a recorded reason."),
        "revision": revision if earlier else None,
        "previous_versions": [{"frozen_dir": project_relative_path(path.parent), "manifest_sha256": sha256_file(path)}
                              for path in earlier],
        "preregistration": preregistration, "verification": config["verification"],
        "stop_conditions": config["stop_conditions"], "design": config["design"],
        "files_sha256": {path: sha256_file(project_path(path)) for path in dict.fromkeys(server)},
        "local_sha256": {path: sha256_file(project_path(path)) for path in dict.fromkeys(local)},
        "plan_counts": check["counts"], "plan_overlap": check["overlap"], "plan_sufficiency": check["sufficiency"],
    }
    write_json(manifest, frozen_dir / "manifest.json", allow_nan=False)
    return frozen_dir, manifest


def verify_freeze(config, with_local=False):
    """고정 manifest를 읽고 파일 해시를 다시 확인한다. 바뀐 파일이 있으면 중단."""
    manifest = read_json(project_path(config["frozen_dir"]) / "manifest.json")
    require(manifest["status"] == "frozen" and manifest["experiment"] == "step6_contribution", "Invalid manifest")
    changed = changed_files(manifest["files_sha256"])
    require(not changed, f"Files changed after the STEP 6-2 freeze: {changed}")
    for key in ("preregistration", "verification", "stop_conditions", "design"):
        require(config[key] == manifest[key], f"Config {key} differs from the frozen one")
    if with_local:
        changed = changed_files(manifest["local_sha256"])
        require(not changed, f"Local files changed after the freeze: {changed}")
    return manifest


def main():
    frozen_dir, manifest = freeze()
    print(f"고정: {project_relative_path(frozen_dir)} ({len(manifest['files_sha256'])} 서버 파일, "
          f"{len(manifest['local_sha256'])} 로컬 파일)")
    print(f"manifest sha256 {sha256_file(frozen_dir / 'manifest.json')}")


if __name__ == "__main__":
    main()
