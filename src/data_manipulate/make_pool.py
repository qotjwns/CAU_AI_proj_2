import argparse
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import read_json, write_json
from src.common.config import load_config
from src.common.paths import DEFAULT_CONFIG, PROJECT_ROOT
from src.data_manipulate.make_inference_pool import generate_groups
from src.data_manipulate.build_cases import save_cases


def build_pool_ids(data, config):
    excluded_ids = set(config["exclude_image_ids"])
    images = {image["id"]: image for image in data["images"] if image["id"] not in excluded_ids}
    categories = {category["id"]: category for category in data["categories"]}
    all_image_ids = set(images)

    present = {category_id: set() for category_id in categories}
    positive = {category_id: set() for category_id in categories}

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id in excluded_ids:
            continue
        category_id = annotation["category_id"]
        image = images[image_id]
        present[category_id].add(image_id)

        _, _, bbox_width, bbox_height = annotation["bbox"]
        image_area = image["width"] * image["height"]
        if (
            bbox_width * bbox_height >= config["min_bbox_area_ratio"] * image_area
            and not (config["exclude_crowd_positives"] and annotation.get("iscrowd", 0))
        ):
            positive[category_id].add(image_id)

    pools = {}
    for category_id in categories:
        positive_ids = positive[category_id]
        negative_candidate_ids = all_image_ids - present[category_id]
        excluded_positive_ids = present[category_id] - positive_ids

        pools[category_id] = {
            "positive": sorted(positive_ids),
            "negative_candidates": sorted(negative_candidate_ids),
            "excluded_positive": sorted(excluded_positive_ids),
        }
    return pools


def summarize_pools(data, pools, config):
    excluded_ids = set(config["exclude_image_ids"])
    rows = []
    for category in data["categories"]:
        category_id = category["id"]
        pool = pools[category_id]

        row = {
            "category_id": category_id,
            "category_name": category["name"],
            "positive_count": len(pool["positive"]),
            "negative_candidate_count": len(pool["negative_candidates"]),
            "excluded_positive_count": len(pool["excluded_positive"]),
        }
        rows.append(row)

    rows.sort(key=lambda row: (-row["positive_count"], row["category_id"]))
    return {
        "image_count": sum(image["id"] not in excluded_ids for image in data["images"]),
        "category_count": len(data["categories"]),
        "min_bbox_area_ratio": config["min_bbox_area_ratio"],
        "exclude_crowd_positives": config["exclude_crowd_positives"],
        "exclude_image_ids": config["exclude_image_ids"],
        "definitions": {
            "positive": "대상 카테고리의 bbox 중 하나 이상이 설정한 면적·crowd 기준을 충족",
            "negative_candidate": "크기와 관계없이 대상 카테고리의 주석이 하나도 없음",
            "excluded_positive": "대상 카테고리의 주석이 있지만 양성 기준을 충족하지 않음",
        },
        "categories": rows,
    }


def save_pools(data, pools, config):
    images = {image["id"]: image for image in data["images"]}
    image_records = {}
    for image_id, image in images.items():
        path = config["image_dir"] / image["file_name"]
        image_records[image_id] = {
            "image_id": image_id,
            "image_path": Path(os.path.relpath(path, PROJECT_ROOT)).as_posix(),
        }

    for category in sorted(data["categories"], key=lambda item: item["id"]):
        category_id = category["id"]
        folder_name = f"{category_id:02d}_{category['name'].replace(' ', '_')}"
        category_dir = config["output_dir"] / "pools" / folder_name

        for pool_type in ("positive", "negative_candidates"):
            image_ids = pools[category_id][pool_type]
            manifest = {
                "category_id": category_id,
                "category_name": category["name"],
                "pool_type": pool_type,
                "annotation_source": Path(os.path.relpath(config["annotation_path"], PROJECT_ROOT)).as_posix(),
                "image_path_base": "project_root",
                "image_count": len(image_ids),
            }
            if pool_type == "positive":
                manifest["selection_rule"] = "at_least_one_bbox_meets_area_ratio"
                manifest["min_bbox_area_ratio"] = config["min_bbox_area_ratio"]
                manifest["exclude_crowd_positives"] = config["exclude_crowd_positives"]
            else:
                manifest["selection_rule"] = "no_annotation_for_category_at_any_size"
                manifest["manual_review_completed"] = False
            manifest["images"] = [image_records[image_id] for image_id in image_ids]

            output_path = category_dir / f"{pool_type}.json"
            write_json(manifest, output_path)


def main():
    parser = argparse.ArgumentParser(description="설정에 따라 클래스별 양성·음성 후보 목록 생성")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("A", "B"), help="지정하면 그룹과 해당 모드의 케이스까지 생성")
    args = parser.parse_args()
    config = load_config(args.config)
    data = read_json(config["annotation_path"])
    pools = build_pool_ids(data, config)
    summary = summarize_pools(data, pools, config)
    save_pools(data, pools, config)

    print(f"전체 이미지: {summary['image_count']:,}장")
    print(f"카테고리: {summary['category_count']:,}개")
    print(f"양성 기준: bbox 면적 / 이미지 면적 >= {config['min_bbox_area_ratio']:.0%}")

    output_path = config["output_dir"] / "pool_counts.json"
    write_json(summary, output_path)
    print(f"\n집계 저장: {output_path}")
    print(f"양성·음성 후보 이미지 목록 저장: {config['output_dir'] / 'pools'}")
    if args.mode:
        groups_json = generate_groups(config)
        save_cases(groups_json, args.mode)


if __name__ == "__main__":
    main()
