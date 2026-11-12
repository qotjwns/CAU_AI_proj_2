"""STEP 4 독립 검증용 케이스 생성.

- STEP 1~3에서 쓴 이미지를 전역에서 제외하고, 클래스마다 새 P·N 이미지를 뽑는다.
- STEP 4 클래스끼리도 이미지를 공유하지 않는다.
- 클래스 안에서 P·N을 겹치지 않는 3장 묶음으로 나눠 pair를 만들므로, 같은 입력이 반복되지 않는다.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from src.common.config import read_yaml
from src.common.data import normalize_category, read_json, read_jsonl, write_json
from src.common.paths import project_path, project_relative_path
from src.core.feature_validation import require
from src.data_manipulate.build_cases import make_case
from src.data_manipulate.build_groups import select_category_dirs


def pair_variants(p, n):
    """한 pair(P 3장, N 3장)의 케이스 구성.

    A·B는 STEP 1과 같은 구성이다. C는 N 1장 구성을 늘린 것으로, n2·n3을 세 위치에 모두 두고
    P 짝을 (p2, p3), (p3, p1)로 돌려 모든 P가 위치 1에 한 번 이상 오게 한다.
    """
    (p1, p2, p3), (n1, n2, n3) = p, n
    return {
        "A": [("PPP", (p1, p2, p3)), ("PPN", (p1, p2, n1)), ("PNN", (p1, n2, n1)), ("NNN", (n3, n2, n1))],
        "B": [("NPP", (n1, p1, p2)), ("PNP", (p1, n1, p2)), ("PPN", (p1, p2, n1))],
        "C": [("NPP", (n2, p2, p3)), ("PNP", (p2, n2, p3)), ("PPN", (p2, p3, n2)),
              ("NPP", (n3, p3, p1)), ("PNP", (p3, n3, p1)), ("PPN", (p3, p1, n3))],
    }


def excluded_image_ids(paths):
    return {image_id for path in paths for case in read_jsonl(project_path(path)) for image_id in case["image_ids"]}


def build_independent_cases(config, data_config):
    seed, per_label = config["seed"], config["images_per_label"]
    require(per_label % 3 == 0, "images_per_label must be a multiple of 3")
    excluded = excluded_image_ids(config["exclude_cases_jsonl"])
    classes = sorted({case["class"] for path in config["exclude_cases_jsonl"]
                      for case in read_jsonl(project_path(path))})
    pool_dir = project_path(config["pool_dir"])
    taken = set(excluded)
    groups, cases = {}, []
    for name in classes:
        category_dir = select_category_dirs(pool_dir, [name], None)[0]
        chosen = {}
        for label, filename in (("P", "positive.json"), ("N", "negative_candidates.json")):
            pool = read_json(category_dir / filename)
            candidates = sorted((image for image in pool["images"] if image["image_id"] not in taken),
                                key=lambda image: image["image_id"])
            require(len(candidates) >= per_label, f"{name}: only {len(candidates)} unused {label} images")
            rng = random.Random(f"{seed}:{pool['category_id']}:{label}")
            chosen[label] = rng.sample(candidates, k=per_label)
            taken.update(image["image_id"] for image in chosen[label])
        names = {label: [Path(image["image_path"]).name for image in images] for label, images in chosen.items()}
        group_id = f"step4_{normalize_category(name).replace(' ', '_')}"
        group = {
            "class": name,
            "pos_image": names["P"], "neg_image": names["N"],
            "image_id_by_name": {Path(image["image_path"]).name: image["image_id"]
                                 for images in chosen.values() for image in images},
        }
        groups[group_id] = group
        for index in range(per_label // 3):
            p = tuple(names["P"][3 * index:3 * index + 3])
            n = tuple(names["N"][3 * index:3 * index + 3])
            ids = [group["image_id_by_name"][image] for image in p + n]
            key = json.dumps([group_id, data_config["prompt"], ids], ensure_ascii=False)
            pair_id = f"{group_id}__{hashlib.sha256(key.encode()).hexdigest()[:12]}"
            for mode, variants in pair_variants(p, n).items():
                for composition, images in variants:
                    cases.append(make_case(group_id, group, pair_id, images, composition, data_config, mode))
    return classes, excluded, groups, cases


def summarize(cases, excluded, groups):
    inputs = Counter((case["prompt"], tuple(case["image_ids"])) for case in cases)
    images = [image_id for group in groups.values() for image_id in group["image_id_by_name"].values()]
    require(not set(images) & excluded, "STEP 4 images overlap STEP 1-3 images")
    require(len(images) == len(set(images)), "An image is reused across STEP 4 classes")
    duplicates = Counter(tuple(sorted(case["mode"] + "".join(case["image_labels"]) for case in cases
                                      if (case["prompt"], tuple(case["image_ids"])) == key))
                         for key, count in inputs.items() if count > 1)
    return {
        "records": len(cases), "unique_inputs": len(inputs),
        "duplicate_groups": {"+".join(key): count for key, count in duplicates.items()},
        "by_mode": dict(Counter(case["mode"] for case in cases)),
        "by_composition_unique": dict(Counter("".join(case["image_labels"]) for case in
                                              {(c["prompt"], tuple(c["image_ids"])): c for c in cases}.values())),
        "classes": len(groups), "images": len(images), "excluded_step1_3_images": len(excluded),
    }


def save_independent_cases(config_path, data_config):
    config = read_yaml(project_path(config_path))
    classes, excluded, groups, cases = build_independent_cases(config, data_config)
    summary = summarize(cases, excluded, groups)
    output_dir = project_path(config["output_dir"])
    stem = f"step4_seed{config['seed']}"
    cases_path = output_dir / f"{stem}_cases.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with cases_path.open("x", encoding="utf-8") as destination:
        for case in cases:
            destination.write(json.dumps(case, ensure_ascii=False) + "\n")
    write_json({"config": config, "config_path": project_relative_path(config_path), "summary": summary,
                "classes": classes, "groups": groups}, output_dir / f"{stem}_groups.json")
    return cases_path, summary

