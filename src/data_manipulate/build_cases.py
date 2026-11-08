# python src/data_manipulate/build_cases.py --groups-json data/balanced3/inference_pools/inference_groups_top20_seed42.json --mode A
# python src/data_manipulate/build_cases.py --groups-json data/balanced3/inference_pools/inference_groups_top20_seed42.json --mode B

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import read_json
from src.common.paths import PROJECT_ROOT, project_path


def make_case(group_id, group, pair_id, images, composition, config, mode):
    labels = list(composition)
    positive_count = labels.count("P")
    answer = "Yes" if positive_count == 3 else "No"
    record = {
        "experiment_type": "paired3",
        "group_id": group_id,
        "pair_id": pair_id,
        "class": group["class"],
        "composition": "+".join(labels),
        "image_path_base": "project_root",
        "image_dir": Path(os.path.relpath(project_path(config["image_dir"]), PROJECT_ROOT)).as_posix(),
        "image_order": list(images),
        "image_ids": [group["image_id_by_name"][name] for name in images],
        "image_labels": labels,
        "prompt": config["prompt"].format(object=group["class"]),
        "gt": answer,
        "positive_count": positive_count,
        "negative_count": 3 - positive_count,
        "negative_images": [name for name, label in zip(images, labels) if label == "N"],
        "negative_positions": [index for index, label in enumerate(labels) if label == "N"],
    }
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return {"case_id": f"paired3__{group_id}__{digest}", "mode": mode, **record}


def build_cases(groups, config, mode):
    cases = []
    for group_id, group in groups.items():
        rng = random.Random(f"{config['seed']}:{group_id}:pairs")
        for p1, p2, p3 in combinations(group["pos_image"], 3):
            n1, n2, n3 = rng.sample(group["neg_image"], 3)
            image_ids = [group["image_id_by_name"][name] for name in (p1, p2, p3, n1, n2, n3)]
            pair_key = json.dumps([group_id, config["prompt"], image_ids], ensure_ascii=False)
            pair_id = f"{group_id}__{hashlib.sha256(pair_key.encode()).hexdigest()[:12]}"
            if mode == "A":
                variants = [
                    ("PPP", (p1, p2, p3)),
                    ("PPN", (p1, p2, n1)),
                    ("PNN", (p1, n2, n1)),
                    ("NNN", (n3, n2, n1)),
                ]
            else:
                variants = [
                    ("NPP", (n1, p1, p2)),
                    ("PNP", (p1, n1, p2)),
                    ("PPN", (p1, p2, n1)),
                ]
            for composition, images in variants:
                cases.append(make_case(group_id, group, pair_id, images, composition, config, mode))
    return cases


def save_cases(groups_json, mode, output=None):
    source = groups_json.expanduser().resolve()
    data = read_json(source)
    config = data["config"]
    cases = build_cases(data["groups"], config, mode)
    output_path = (
        output.expanduser().resolve() if output
        else project_path(config["output_dir"]) / "inference_cases" / f"{source.stem}_{mode}_cases.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as destination:
        for case in cases:
            destination.write(json.dumps(case, ensure_ascii=False) + "\n")
    counts = Counter(case["gt"] for case in cases)
    print(f"Mode {mode}: 전체 {len(cases)}개 | Yes {counts['Yes']} / No {counts['No']}")
    print(f"저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="A: 한 장 교체 비교, B: 같은 이미지의 위치 비교")
    parser.add_argument("--groups-json", type=Path, required=True)
    parser.add_argument("--mode", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    save_cases(args.groups_json, args.mode, args.output)


if __name__ == "__main__":
    main()
