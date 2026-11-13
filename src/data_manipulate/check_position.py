"""STEP 5-2b 케이스 일관성 검사 (CPU). 모델 출력은 보지 않는다.

재생성 일치, 개수, 같은 계열·장수의 5-2 입력과 이미지 집합·P 순서·N 위치·프롬프트 일치, 끝 위치가 5-2와 같은 입력인지,
새 이미지 없음, case_id·입력 중복 없음, 이미지 파일 존재, 5-2 고정 기록이 그대로인지 확인한다.
"""
from collections import Counter

from src.common.config import read_yaml
from src.common.data import read_json, read_jsonl
from src.common.paths import project_path, project_relative_path
from src.core.accumulation_freeze import verify_accumulation_freeze
from src.core.probe_freeze import sha256_file
from src.data_manipulate.build_position import build_position_cases, paths, summarize


def check_cases(config, data_config):
    target = paths(config)
    saved = list(read_jsonl(target["cases"]))
    failures = []

    def fail(case_id, message):
        failures.append({"case_id": case_id, "message": message})

    if saved != build_position_cases(config):
        fail(None, "Saved cases differ from cases rebuilt with the same config")
    manifest = read_json(project_path(config["accumulation_frozen_manifest"]))
    changed = verify_accumulation_freeze(manifest)
    if changed:
        fail(None, f"STEP 5-2 frozen files changed: {changed}")
    source_path = config["accumulation_cases"]
    if manifest["files_sha256"].get(source_path) != sha256_file(project_path(source_path)):
        fail(None, "5-2 cases differ from the 5-2 freeze")
    template = read_yaml(project_path(config["accumulation_config"]))["prompt"]

    source = {case["case_id"]: case for case in read_jsonl(project_path(source_path))}
    source_inputs = {(case["prompt"], tuple(case["image_ids"])): case["case_id"] for case in source.values()}
    source_images = {i for case in source.values() for i in case["image_ids"]}
    image_dir = project_path(data_config["image_dir"])
    ids, cells = Counter(), Counter()
    for case in saved:
        cid, k, a = case["case_id"], case["series_step"], case["n_before"]
        ids[cid] += 1
        cells[(case["n_images"], a)] += 1
        first = source.get(case["source_cases"].get("N-first"))
        last = source.get(case["source_cases"].get("N-last"))
        reference = first or last
        if reference is None:
            fail(cid, "Source 5-2 case not found")
            continue
        negative = reference["anchor_negative_id"]
        positives = reference["series_image_ids"][:k]
        expected_ids = [*positives[:a], negative, *positives[a:]]
        if case["image_ids"] != expected_ids:
            fail(cid, "Image order is not P1..Pk with N inserted after a P images")
        if case["image_order"] != [f"{i:012d}.jpg" for i in case["image_ids"]]:
            fail(cid, "Image file names do not follow image IDs")
        if not all((image_dir / name).exists() for name in case["image_order"]):
            fail(cid, "Image file missing")
        if not (0 <= a <= k and case["n_after"] == k - a and case["n_images"] == k + 1 == len(case["image_ids"])):
            fail(cid, "Position fields disagree")
        if case["image_labels"] != ["P"] * a + ["N"] + ["P"] * (k - a) or case["negative_positions"] != [a]:
            fail(cid, "Labels or N position disagree")
        if case["composition"] != "+".join(case["image_labels"]) or case["gt"] != "No":
            fail(cid, "Composition or ground truth disagree")
        if case["prompt"] != template.format(object=case["class"], count=k + 1) or case["prompt"] != reference["prompt"]:
            fail(cid, "Prompt differs from the 5-2 template or the 5-2 input of the same count")
        for name in ("pair_id", "class", "group_id", "strength", "anchor_negative_id", "series_image_ids", "image_dir"):
            if case[name] != reference[name]:
                fail(cid, f"Field {name} differs from the 5-2 series")
        expected_role = "first" if a == 0 else "last" if a == k else "middle"
        if case["role"] != expected_role:
            fail(cid, "Role does not match the N position")
        key = (case["prompt"], tuple(case["image_ids"]))
        if expected_role == "middle":
            if key in source_inputs:
                fail(cid, "A middle-position input already exists in 5-2")
            if set(case["source_cases"]) != {"N-first", "N-last"}:
                fail(cid, "Middle case must link both 5-2 end cases")
        elif source_inputs.get(key) != reference["case_id"]:
            fail(cid, "End-position input is not the linked 5-2 input")
        if not set(case["image_ids"]) <= source_images:
            fail(cid, "Uses an image outside STEP 5-2")
    duplicates = [cid for cid, count in ids.items() if count > 1]
    if duplicates:
        fail(None, f"Duplicate case_id: {duplicates[:5]}")
    if set(ids) & set(source):
        fail(None, "case_id collides with a 5-2 case_id")
    series = {case["pair_id"] for case in saved}
    for (n, a), count in sorted(cells.items()):
        if count != len(series):
            fail(None, f"{n} images, a={a}: {count} cases for {len(series)} series")
    expected_cells = {(k + 1, a) for k in config["p_counts"] for a in range(k + 1)}
    if set(cells) != expected_cells:
        fail(None, "Position grid cells differ from the design")
    return {
        "check": "step5_2b_position_cases", "status": "passed" if not failures else "failed",
        "note": "Consistency with STEP 5-2 inputs and the position-grid rule only. No model output is read.",
        "cases": project_relative_path(target["cases"]), "cases_sha256": sha256_file(target["cases"]),
        "source_cases": source_path, "source_cases_sha256": sha256_file(project_path(source_path)),
        "accumulation_frozen_manifest_sha256": sha256_file(project_path(config["accumulation_frozen_manifest"])),
        "counts": summarize(saved), "failure_count": len(failures), "failures": failures[:200],
    }
