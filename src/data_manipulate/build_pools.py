from src.common.data import write_json
from src.common.paths import project_relative_path


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

        rows.append({
            "category_id": category_id,
            "category_name": category["name"],
            "positive_count": len(pool["positive"]),
            "negative_candidate_count": len(pool["negative_candidates"]),
            "excluded_positive_count": len(pool["excluded_positive"]),
        })

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
    image_records = {
        image["id"]: {
            "image_id": image["id"],
            "image_path": project_relative_path(config["image_dir"] / image["file_name"]),
        }
        for image in data["images"]
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
                "annotation_source": project_relative_path(config["annotation_path"]),
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
