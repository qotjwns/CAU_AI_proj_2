"""STEP 5-3 실험 A·B 사전 등록 고정 (CPU, 연습 실행·본 실행 전).

files_sha256: 설정·분할·계획 파일·코드 (GPU 서버에도 있는 파일, 실행 전에 서버에서 다시 확인)
baseline_sha256: 기준 실행 manifest·run.json과 5-2b·5-2 고정 manifest (로컬 평가 때 다시 확인)
이미 고정된 버전은 덮어쓰지 않는다. 이전 버전이 있으면 설정의 revision(버전, 이유, 변경 항목, 5-3 GPU 출력 전
수정 여부)이 있어야 새 버전을 만들고, 이전 manifest의 해시를 기록한다 (이전 버전 파일은 그대로 둔다).
GPU 실행 뒤의 수정(v3)은 확증 값을 보기 전이고(made_before_any_confirmation_value) 실행이 쓴 고정본
(run_frozen_version)을 밝힐 때만 허용한다. 평가 때 run_compatibility가 평가 쪽 파일만 바뀌었는지 다시 확인한다.
"""
from datetime import datetime, timezone

from src.common.config import read_yaml
from src.common.data import read_json, write_json
from src.common.paths import project_path, project_relative_path
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file


EXPERIMENTS = {"A": "configs/experiments/step5_patching.yaml", "B": "configs/experiments/step5_attention.yaml"}
# GPU 실행이 쓰지 않는 평가 쪽 파일 (실행 뒤 고정본에서 바뀌어도 되는 파일, 실험 설정 파일은 따로 더한다)
EVALUATION_SIDE_FILES = ("src/core/causal_eval.py", "src/core/causal_freeze.py",
                         "experiments/step5_causal/evaluate.py", "experiments/step5_causal/selftest_eval.py")
# 실행 고정본과 평가 고정본에서 같아야 하는 내용
RUN_INVARIANT_KEYS = ("preregistration", "verification", "stop_conditions", "baseline_sha256", "plan_counts")


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
    post_run = False
    if earlier:
        require(revision and revision.get("version") == frozen_dir.name and revision.get("reason")
                and revision.get("changes"),
                "Earlier frozen versions exist; the config needs a revision block (version, reason, changes)")
        post_run = revision.get("made_before_any_step5_3_gpu_output") is not True
        require(not post_run or (revision.get("made_before_any_confirmation_value") is True
                                 and revision.get("run_frozen_version")),
                "A revision after the 5-3 GPU run must be made before any confirmation value and name the run's version")
    preregistration = config["preregistration"]
    require(preregistration["minimum_effect"]["confirmed"] is True, "The minimum effect has not been confirmed")
    require(config["status"] == "confirmatory", "Only a confirmatory config can be frozen")
    run_frozen = None
    if post_run:
        run = read_json(project_path(config["main_dir"]) / "run.json")
        run_frozen = run["frozen"]
        require(run["status"] == "completed" and run_frozen["version"] == revision["run_frozen_version"],
                f"The main run was not made under {revision['run_frozen_version']}")
        require(not (project_path(config["evaluation_dir"]) / "analysis.json").exists(),
                "An evaluation result already exists; a post-run revision must come before any confirmation value")
    elif check_outputs:
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
        "note": (("Frozen after the STEP 5-3 GPU run and before any confirmation value; the run was made under "
                  f"{run_frozen['version']} and only evaluation-side files may differ from it (see revision). "
                  if post_run else "Frozen before any STEP 5-3 pilot or main run. ")
                 + "The evaluation must use this preregistration; a change requires a new version with a recorded reason."),
        "revision": revision if earlier else None,
        "run_frozen": run_frozen,
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


def run_compatibility(experiment, manifest, run_frozen):
    """본 실행이 쓴 고정본(run.json의 frozen)과 평가 고정본이 다르면, 평가 쪽 파일만 바뀌었는지 확인한다."""
    if run_frozen["version"] == manifest["version"]:
        return {"run_frozen_version": run_frozen["version"], "same_version": True}
    revision = manifest.get("revision") or {}
    require(revision.get("run_frozen_version") == run_frozen["version"] and manifest.get("run_frozen") == run_frozen,
            f"The main run used {run_frozen['version']}, which this frozen version does not declare")
    path = project_path(run_frozen["dir"]) / "manifest.json"
    require(sha256_file(path) == run_frozen["manifest_sha256"], f"The run's frozen manifest changed: {path}")
    run_manifest = read_json(path)
    for key in RUN_INVARIANT_KEYS:
        require(run_manifest[key] == manifest[key], f"{key} differs from the run's frozen version")
    old, new = run_manifest["files_sha256"], manifest["files_sha256"]
    differing = sorted(p for p in set(old) | set(new) if old.get(p) != new.get(p))
    allowed = set(EVALUATION_SIDE_FILES) | {EXPERIMENTS[experiment]}
    require(set(differing) <= allowed, f"Files used by the GPU run changed since {run_frozen['version']}: "
            f"{sorted(set(differing) - allowed)}")
    return {"run_frozen_version": run_frozen["version"], "same_version": False,
            "run_manifest_sha256": run_frozen["manifest_sha256"], "files_changed_since_run": differing}
