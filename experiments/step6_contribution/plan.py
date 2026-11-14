"""STEP 6-2 계획 파일 (CPU, STEP 6-2 GPU 출력 전). 덮어쓰지 않는다.

python -m experiments.step6_contribution.plan
→ data/step6_contribution/pool_inputs.jsonl, items.json, plan_check.json

1. STEP 3 탐색용 풀 입력: STEP 1~3 A·B 케이스(STEP 2 추출·STEP 3 Probe에 쓴 입력)의 고유 입력.
   (이미지 순서, 클래스, 질문)이 같은 레코드는 하나로 본다. P 기저·평균 차이 방향은 이 입력에서만 학습한다.
2. 연습·본 실행 항목: STEP 5-3 계획(b_items)에서 고른다.
   - 연습(선택 절반, 무결성·시간 전용): 강화 모집단(False-Yes) one-N에서 (장수 3·4·5 × N 위치 첫·가운데·끝)마다
     item_id 순 첫 항목 9개, All-P 항목에서 장수마다 첫 항목 3개
   - 본 실행(확증 절반): 강화 모집단 one-N 항목 전부(주 판정), All-P 항목 전부(부작용)
3. 검사 (하나라도 실패하면 status failed, freeze 불가):
   - STEP 3 풀 이미지와 STEP 5-2·5-2b·5-3 입력 이미지의 겹침 0 (COCO id, 파일 SHA-256 둘 다)
   - 평균 차이 방향을 만들 N·P 기여가 충분한지 (아래 SUFFICIENCY, 사전에 고정)
"""
import hashlib
import json
from collections import Counter, defaultdict

from src.common.config import read_yaml
from src.common.data import read_jsonl, write_json
from src.common.paths import project_path, project_relative_path
from src.core.causal_plan import load_plan
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file


OUTPUT_DIR = "data/step6_contribution"
SPLIT_CONFIG = "configs/data/step5_causal_split.yaml"
POOL_CASES = ("data/balanced3/inference_cases/inference_groups_top20_seed42_A_cases.jsonl",
              "data/balanced3/inference_cases/inference_groups_top20_seed42_B_cases.jsonl")
OTHER_CASES = {"5-2": "data/step5_accumulation/step5_seed52_cases.jsonl",
               "5-2b": "data/step5_position/step5b_position_cases.jsonl"}
IMAGE_DIR = "data/raw/coco/val2017"
# 평균 차이 방향·P 기저를 만들 수 있다고 볼 최소 조건 (STEP 6-2 GPU 출력 전에 고정)
SUFFICIENCY = {"min_n_images_per_class_mixed": 5, "min_p_images_per_class": 5,
               "min_n_slots_mixed": 1000, "min_p_slots_mixed": 1000, "min_p_slots_basis": 1000}
PILOT_ONE_NEGATIVE = [(n, position) for n in (3, 4, 5) for position in ("first", "middle", "last")]


def pool_inputs():
    unique = {}
    for path in POOL_CASES:
        for case in read_jsonl(project_path(path)):
            key = (tuple(case["image_ids"]), case["class"], case["prompt"])
            unique.setdefault(key, []).append(case)
    rows = []
    for cases in unique.values():
        cases = sorted(cases, key=lambda c: c["case_id"])
        case = cases[0]
        labels = case["image_labels"]
        require(all(c["image_labels"] == labels for c in cases), "Same input with different labels")
        rows.append({"input_id": f"pool::{case['case_id']}", "case": case, "case_ids": [c["case_id"] for c in cases],
                     "class": case["class"], "composition": case["composition"], "labels": labels,
                     "mixed": "P" in labels and "N" in labels})
    return sorted(rows, key=lambda r: r["input_id"])


def image_hash(image_id):
    return hashlib.sha256(project_path(f"{IMAGE_DIR}/{image_id:012d}.jpg").read_bytes()).hexdigest()


def overlap_check(pool, plan53):
    pool_ids = {i for row in pool for i in row["case"]["image_ids"]}
    others = {name: {i for case in read_jsonl(project_path(path)) for i in case["image_ids"]}
              for name, path in OTHER_CASES.items()}
    others["5-3"] = {i for entry in plan53["inputs"].values() for i in entry["case"]["image_ids"]}
    pool_hashes = {image_hash(i) for i in pool_ids}
    every = set().union(*others.values())
    other_hashes = {image_hash(i) for i in every}
    result = {"step3_pool_images": len(pool_ids),
              "by_id": {name: {"images": len(ids), "overlap": len(ids & pool_ids)} for name, ids in others.items()},
              "by_file_sha256": {"step3_pool_unique_files": len(pool_hashes),
                                 "evaluation_unique_files": len(other_hashes),
                                 "overlap": len(pool_hashes & other_hashes)}}
    result["passed"] = (all(v["overlap"] == 0 for v in result["by_id"].values())
                        and result["by_file_sha256"]["overlap"] == 0)
    return result


def sufficiency(pool):
    images = defaultdict(lambda: defaultdict(set))
    slots = Counter()
    for row in pool:
        for image_id, label in zip(row["case"]["image_ids"], row["labels"]):
            if label == "P":
                slots["p_slots_basis"] += 1
                images[row["class"]]["P"].add(image_id)
            if row["mixed"]:
                slots[f"{label.lower()}_slots_mixed"] += 1
                if label == "N":
                    images[row["class"]]["N_mixed"].add(image_id)
    per_class_n = {cls: len(v["N_mixed"]) for cls, v in sorted(images.items())}
    per_class_p = {cls: len(v["P"]) for cls, v in sorted(images.items())}
    result = {"criteria": SUFFICIENCY, "slots": dict(slots), "classes": len(images),
              "n_images_per_class_mixed": per_class_n, "p_images_per_class": per_class_p,
              "unique_n_images_mixed": len({i for v in images.values() for i in v["N_mixed"]}),
              "unique_p_images": len({i for v in images.values() for i in v["P"]})}
    result["passed"] = (len(images) == 20
                        and min(per_class_n.values()) >= SUFFICIENCY["min_n_images_per_class_mixed"]
                        and min(per_class_p.values()) >= SUFFICIENCY["min_p_images_per_class"]
                        and slots["n_slots_mixed"] >= SUFFICIENCY["min_n_slots_mixed"]
                        and slots["p_slots_mixed"] >= SUFFICIENCY["min_p_slots_mixed"]
                        and slots["p_slots_basis"] >= SUFFICIENCY["min_p_slots_basis"])
    return result


def stage_items(plan53):
    items = sorted(plan53["b_items"], key=lambda i: i["item_id"])
    pilot_one, pilot_allp = {}, {}
    for item in items:
        if item["half"] != "selection":
            continue
        if item["population"] == "boost":
            pilot_one.setdefault((item["n_images"], item["position"]), item["item_id"])
        elif item["population"] == "all_positive":
            pilot_allp.setdefault(item["n_images"], item["item_id"])
    require(set(pilot_one) == set(PILOT_ONE_NEGATIVE) and set(pilot_allp) == {3, 4, 5}, "Pilot items are incomplete")
    return {
        "pilot": {"one_negative": [pilot_one[key] for key in PILOT_ONE_NEGATIVE],
                  "all_positive": [pilot_allp[n] for n in (3, 4, 5)]},
        "main": {"one_negative": [i["item_id"] for i in items if i["half"] == "confirmation"
                                  and i["population"] == "boost"],
                 "all_positive": [i["item_id"] for i in items if i["half"] == "confirmation"
                                  and i["population"] == "all_positive"]},
        "rules": {
            "pilot_one_negative": "selection half, boost population (False-Yes), first item_id per image count x N position",
            "pilot_all_positive": "selection half, All-P items, first item_id per image count; every slot as pseudo-N",
            "main_one_negative": "confirmation half, boost population (False-Yes), every item (primary population)",
            "main_all_positive": "confirmation half, All-P items, every item and every slot (side effects)"}}


def build():
    output_dir = project_path(OUTPUT_DIR)
    require(not (output_dir / "plan_check.json").exists(), f"{OUTPUT_DIR} already has a plan; it is never overwritten")
    plan53 = load_plan(read_yaml(project_path(SPLIT_CONFIG)))
    pool = pool_inputs()
    items = stage_items(plan53)
    overlap = overlap_check(pool, plan53)
    enough = sufficiency(pool)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pool_inputs.jsonl").open("w", encoding="utf-8") as handle:
        for row in pool:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(items, output_dir / "items.json", allow_nan=False)
    composition = Counter(row["composition"] for row in pool)
    counts = {"pool_inputs": len(pool), "pool_mixed_inputs": sum(r["mixed"] for r in pool),
              "pool_composition": dict(sorted(composition.items())),
              **{f"{stage}_{kind}": len(ids) for stage in ("pilot", "main") for kind, ids in items[stage].items()
                 if kind != "rules"}}
    items53 = {i["item_id"]: i for i in plan53["b_items"]}
    counts["main_all_positive_batches"] = sum(items53[i]["n_images"] for i in items["main"]["all_positive"])
    counts["pilot_all_positive_batches"] = sum(items53[i]["n_images"] for i in items["pilot"]["all_positive"])
    files = [project_relative_path(output_dir / name) for name in ("pool_inputs.jsonl", "items.json")]
    split_config = read_yaml(project_path(SPLIT_CONFIG))
    plan53_check = f"{split_config['output_dir']}/{split_config['check_file']}"
    sources = [*POOL_CASES, *OTHER_CASES.values(), SPLIT_CONFIG, plan53_check, *plan53["check"]["files_sha256"]]
    check = {"status": "passed" if overlap["passed"] and enough["passed"] else "failed",
             "overlap": overlap, "sufficiency": enough, "counts": counts,
             "files_sha256": {path: sha256_file(project_path(path)) for path in files},
             "sources_sha256": {path: sha256_file(project_path(path)) for path in sources}}
    write_json(check, output_dir / "plan_check.json", allow_nan=False)
    return check


def load_step6_plan(plan_dir=OUTPUT_DIR):
    """계획 파일과 검사 결과를 읽는다 (검사 통과·해시 일치 확인)."""
    output_dir = project_path(plan_dir)
    check = json.loads((output_dir / "plan_check.json").read_text(encoding="utf-8"))
    require(check["status"] == "passed", "The STEP 6 plan check has not passed")
    for path, digest in check["files_sha256"].items():
        require(sha256_file(project_path(path)) == digest, f"STEP 6 plan file changed: {path}")
    return {"check": check, "pool": list(read_jsonl(output_dir / "pool_inputs.jsonl")),
            "items": json.loads((output_dir / "items.json").read_text(encoding="utf-8"))}


def main():
    check = build()
    print(json.dumps({k: check[k] for k in ("status", "counts")}, ensure_ascii=False, indent=1))
    print(json.dumps(check["overlap"], ensure_ascii=False))
    print(json.dumps({k: check["sufficiency"][k] for k in ("slots", "unique_n_images_mixed", "unique_p_images",
                                                          "passed")}, ensure_ascii=False))
    require(check["status"] == "passed", "Plan check failed: see plan_check.json")


if __name__ == "__main__":
    main()
