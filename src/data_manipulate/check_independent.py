"""STEP 4 케이스의 Annotation Consistency Check.

생성에 쓴 것과 같은 COCO-REM 주석으로 이미지 ID·클래스·P/N 조건·이미지 순서·구성 규칙이 생성 규칙과
일치하는지 검사한다. 같은 주석을 쓰므로 독립적인 정답 검증이 아니다.
"""
from collections import Counter, defaultdict
import hashlib

from src.common.data import normalize_category, read_json, read_jsonl
from src.common.paths import project_path, project_relative_path
from src.data_manipulate.build_independent import excluded_image_ids, pair_variants


NOTE = ("Annotation consistency check: uses the same COCO-REM annotations as case generation, so it verifies "
        "consistency with the generation rules, not independent ground truth.")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annotation_index(data, min_ratio, exclude_crowd):
    sizes = {image["id"]: image["width"] * image["height"] for image in data["images"]}
    present, positive = defaultdict(set), defaultdict(set)
    for annotation in data["annotations"]:
        image_id, category_id = annotation["image_id"], annotation["category_id"]
        present[image_id].add(category_id)
        _, _, width, height = annotation["bbox"]
        if width * height >= min_ratio * sizes[image_id] and not (exclude_crowd and annotation.get("iscrowd", 0)):
            positive[image_id].add(category_id)
    categories = {normalize_category(category["name"]): category["id"] for category in data["categories"]}
    return set(sizes), present, positive, categories


def check_cases(config, data_config, cases_path, groups_path):
    data = read_json(data_config["annotation_path"])
    images, present, positive, categories = annotation_index(
        data, data_config["min_bbox_area_ratio"], data_config["exclude_crowd_positives"])
    cases = list(read_jsonl(cases_path))
    groups = read_json(groups_path)["groups"]
    excluded = excluded_image_ids(config["exclude_cases_jsonl"])
    image_dir = data_config["image_dir"]
    failures = []

    def fail(case, message):
        failures.append({"case_id": case.get("case_id"), "message": message})

    owner = {}
    for case in cases:
        name = case["class"]
        category = categories.get(normalize_category(name))
        if category is None:
            fail(case, f"unknown class {name}")
            continue
        if case["prompt"] != data_config["prompt"].format(object=name):
            fail(case, "prompt differs from the STEP 1 template")
        ids, order, labels = case["image_ids"], case["image_order"], case["image_labels"]
        if not (len(ids) == len(order) == len(labels) == 3) or len(set(ids)) != 3:
            fail(case, "expected three distinct images")
            continue
        for image_id, filename, label in zip(ids, order, labels):
            if filename != f"{image_id:012d}.jpg":
                fail(case, f"image order mismatch: {filename} vs id {image_id}")
            if image_id not in images:
                fail(case, f"image {image_id} not in annotations")
            if not (image_dir / filename).exists():
                fail(case, f"image file missing: {filename}")
            if label == "P" and category not in positive[image_id]:
                fail(case, f"image {image_id} labeled P but has no qualifying {name} box")
            if label == "N" and category in present[image_id]:
                fail(case, f"image {image_id} labeled N but has a {name} annotation")
            if image_id in excluded:
                fail(case, f"image {image_id} was used in STEP 1-3")
            if owner.setdefault(image_id, name) != name:
                fail(case, f"image {image_id} reused across classes ({owner[image_id]}, {name})")
        if case["composition"] != "+".join(labels):
            fail(case, "composition differs from labels")
        if case["gt"] != ("Yes" if labels == ["P", "P", "P"] else "No"):
            fail(case, "ground truth differs from labels")
        if case["negative_positions"] != [i for i, label in enumerate(labels) if label == "N"]:
            fail(case, "negative positions differ from labels")

    expected = Counter()
    for group_id, group in groups.items():
        positives, negatives = group["pos_image"], group["neg_image"]
        for index in range(len(positives) // 3):
            p, n = tuple(positives[3 * index:3 * index + 3]), tuple(negatives[3 * index:3 * index + 3])
            for mode, variants in pair_variants(p, n).items():
                for composition, images in variants:
                    expected[(group_id, mode, composition, images)] += 1
    actual = Counter((case["group_id"], case["mode"], "".join(case["image_labels"]), tuple(case["image_order"]))
                     for case in cases)
    for key in (expected - actual):
        failures.append({"case_id": None, "message": f"missing case from generation rule: {key}"})
    for key in (actual - expected):
        failures.append({"case_id": None, "message": f"case not produced by generation rule: {key}"})

    inputs = defaultdict(list)
    for case in cases:
        inputs[(case["prompt"], tuple(case["image_ids"]))].append(case)
    for members in inputs.values():
        if len(members) > 1:
            kinds = sorted(case["mode"] + "".join(case["image_labels"]) for case in members)
            if kinds != ["APPN", "BPPN"] or len({case["pair_id"] for case in members}) != 1:
                failures.append({"case_id": members[0]["case_id"], "message": f"unexpected duplicate input {kinds}"})
    unique = {key: members[0] for key, members in inputs.items()}
    by_composition = Counter("".join(case["image_labels"]) for case in unique.values())
    if len({by_composition[c] for c in ("PPN", "PNP", "NPP")}) != 1:
        failures.append({"case_id": None, "message": f"one-N compositions are not balanced: {dict(by_composition)}"})

    return {
        "check": "annotation_consistency_check", "note": NOTE,
        "status": "passed" if not failures else "failed",
        "cases": project_relative_path(cases_path), "cases_sha256": sha256_file(cases_path),
        "groups": project_relative_path(groups_path), "groups_sha256": sha256_file(groups_path),
        "annotation": project_relative_path(data_config["annotation_path"]),
        "annotation_sha256": sha256_file(data_config["annotation_path"]),
        "counts": {"records": len(cases), "unique_inputs": len(unique), "images": len(owner),
                   "classes": len({case["class"] for case in cases}), "excluded_step1_3_images": len(excluded),
                   "unique_by_composition": dict(sorted(by_composition.items()))},
        "failures": failures[:200], "failure_count": len(failures),
    }


def default_paths(config):
    output_dir = project_path(config["output_dir"])
    stem = f"step4_seed{config['seed']}"
    return output_dir / f"{stem}_cases.jsonl", output_dir / f"{stem}_groups.json", output_dir / f"{stem}_consistency_check.json"

