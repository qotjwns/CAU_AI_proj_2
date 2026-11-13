"""STEP 5-3 실험 A·B 사전 등록 고정 (CPU, 연습 실행·본 실행 전).

files_sha256: 설정·분할·계획 파일·코드 (GPU 서버에도 있는 파일, 실행 전에 서버에서 다시 확인)
baseline_sha256: 기준 실행 manifest·run.json과 5-2b·5-2 고정 manifest (로컬 평가 때 다시 확인)
이미 고정된 버전은 덮어쓰지 않는다. 이전 버전이 있으면 설정의 revision(버전, 이유, 변경 항목, 5-3 GPU 출력 전
수정 여부)이 있어야 새 버전을 만들고, 이전 manifest의 해시를 기록한다 (이전 버전 파일은 그대로 둔다).
"""
from datetime import datetime, timezone

from src.common.config import read_yaml
from src.common.data import read_json, write_json
from src.common.paths import project_path, project_relative_path
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file


EXPERIMENTS = {"A": "configs/experiments/step5_patching.yaml", "B": "configs/experiments/step5_attention.yaml"}


def experiment_config(experiment):
    path = EXPERIMENTS[experiment]
    return path, read_yaml(project_path(path))


def changed_files(hashes):
    return [path for path, digest in hashes.items()
            if not project_path(path).exists() or sha256_file(project_path(path)) != digest]


def freeze_causal(experiment, frozen_dir=None, check_outputs=True):
    config_path, config = experiment_config(experiment)
    frozen_dir = project_path(frozen_dir or config["frozen_dir"])
    require(not (frozen_dir / "manifest.json").exists(),
            f"Already frozen: {frozen_dir}. Frozen versions are never overwritten.")
    earlier = sorted(path for path in frozen_dir.parent.glob("*/manifest.json") if path.parent != frozen_dir)
    revision = config.get("revision")
    if earlier:
        require(revision and revision.get("version") == frozen_dir.name and revision.get("reason")
                and revision.get("changes") and revision.get("made_before_any_step5_3_gpu_output") is True,
                "Earlier frozen versions exist; the config needs a revision block (version, reason, changes)")
    preregistration = config["preregistration"]
    require(preregistration["minimum_effect"]["confirmed"] is True, "The minimum effect has not been confirmed")
    require(config["status"] == "confirmatory", "Only a confirmatory config can be frozen")
    if check_outputs:
        for key in ("pilot_dir", "main_dir"):
            require(not (project_path(config[key]) / "run.json").exists(),
                    f"{config[key]} already has model output; freeze must come before any 5-3 GPU run")
    split_path = config["split_config"]
    split_config = read_yaml(project_path(split_path))
    plan_dir = project_path(split_config["output_dir"])
    check = read_json(plan_dir / split_config["check_file"])
    require(check["status"] == "passed", "The plan check has not passed")
    require(not changed_files(check["files_sha256"]), "Plan files changed after the plan check")
    require(not changed_files(check["baseline_sha256"]), "Baseline runs changed after the plan was built")
    report = plan_dir / split_config["report_file"]
    require(report.exists(), "Write the pre-freeze sample-size report first (build_step5_causal_plan --report)")
    paths = [project_relative_path(config_path), split_path, *config["code_files"],
             project_relative_path(plan_dir / split_config["check_file"]), project_relative_path(report),
             *check["files_sha256"]]
    baseline = dict(check["baseline_sha256"])
    for key in ("one_negative_frozen", "all_positive_frozen"):
        manifest_path = project_path(config["baseline"][key]) / "manifest.json"
        baseline[project_relative_path(manifest_path)] = sha256_file(manifest_path)
    manifest = {
        "status": "frozen", "experiment": experiment, "version": frozen_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Frozen before any STEP 5-3 pilot or main run. The evaluation must use this preregistration; "
                 "a change requires a new version with a recorded reason."),
        "revision": revision if earlier else None,
        "previous_versions": [{"frozen_dir": project_relative_path(path.parent), "manifest_sha256": sha256_file(path)}
                              for path in earlier],
        "preregistration": preregistration, "verification": config["verification"],
        "stop_conditions": config["stop_conditions"],
        "files_sha256": {path: sha256_file(project_path(path)) for path in dict.fromkeys(paths)},
        "baseline_sha256": baseline, "plan_counts": check["counts"],
    }
    write_json(manifest, frozen_dir / "manifest.json", allow_nan=False)
    return frozen_dir, manifest


def verify_causal_freeze(experiment, config, with_baseline=False):
    """고정 manifest를 읽고 파일 해시를 다시 확인한다. 바뀐 파일이 있으면 중단."""
    manifest = read_json(project_path(config["frozen_dir"]) / "manifest.json")
    require(manifest["status"] == "frozen" and manifest["experiment"] == experiment, "Invalid frozen manifest")
    changed = changed_files(manifest["files_sha256"])
    require(not changed, f"Files changed after the STEP 5-3 freeze: {changed}")
    require(config["preregistration"] == manifest["preregistration"], "Config preregistration differs from the frozen one")
    if with_baseline:
        changed = changed_files(manifest["baseline_sha256"])
        require(not changed, f"Baseline files changed after the freeze: {changed}")
    return manifest
