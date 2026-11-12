"""STEP 5-2 입력 수준 실험 케이스 생성.

- STEP 4 이미지만 쓴다 (고정 Probe가 본 적 없는 이미지, 새 이미지는 최종 시험용으로 남긴다).
- N 이미지마다, 증거 강도(High / Low)마다 같은 클래스의 P 층에서 P 5장을 seed로 뽑아 순서를 정한다 (P 계열).
- one-N: P1..Pk에 N을 앞(N-first) 또는 뒤(N-last)에 붙인다 (k = 1~4, 전체 2~5장).
- All-P: 같은 계열에서 N 자리를 P(k+1)로 채운다 (같은 장수, 같은 배치).
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import random

from src.common.config import read_yaml
from src.common.data import read_json, read_jsonl, write_json
from src.common.paths import project_path, project_relative_path
from src.core.feature_validation import require


def read_scores(path, layer, position):
    """STEP 4 고정 z_i Probe의 위치 1 확률: {(class, image_id): (label, probability)}."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source) if int(row["position"]) == position]
    scores = {}
    for row in rows:
        key = (row["class"], int(row["image_id"]))
        require(key not in scores, f"Duplicate position-{position} vector for {key}")
        scores[key] = (row["label"], float(row[f"p_layer{layer}"]))
    return scores


def make_strata(group, scores, settings):
    """클래스 안의 P를 확률로 정렬해 하위·상위 1/3 (Probe가 P로 읽는 P만)."""
    name, threshold = group["class"], settings["valid_P_min_probability"]
    ranked, excluded = [], []
    for image in group["pos_image"]:
        image_id = group["image_id_by_name"][image]
        label, probability = scores[(name, image_id)]
        require(label == "P", f"{name} {image_id}: STEP 4 label {label}, expected P")
        (ranked if probability >= threshold else excluded).append((probability, image_id, image))
    ranked.sort()
    m = len(ranked) // 3
    return {"Low": ranked[:m], "High": ranked[-m:], "excluded_read_as_N": sorted(excluded),
            "valid": len(ranked), "stratum_size": m}


def make_case(group_id, group, series_id, images, labels, config, data_config, meta):
    count = len(images)
    record = {
        "experiment_type": "accumulation",
        "group_id": group_id,
        "pair_id": series_id,
        "class": group["class"],
        "composition": "+".join(labels),
        "image_path_base": "project_root",
        "image_dir": project_relative_path(data_config["image_dir"]),
        "image_order": list(images),
        "image_ids": [group["image_id_by_name"][name] for name in images],
        "image_labels": list(labels),
        "prompt": config["prompt"].format(object=group["class"], count=count),
        "gt": "Yes" if all(label == "P" for label in labels) else "No",
        "positive_count": labels.count("P"),
        "negative_count": labels.count("N"),
        "negative_images": [name for name, label in zip(images, labels) if label == "N"],
        "negative_positions": [index for index, label in enumerate(labels) if label == "N"],
        **meta,
    }
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return {"case_id": f"accum__{group_id}__{digest}", "mode": "5-2", **record}


def series_cases(group_id, group, negative, series, strength, config, data_config):
    """한 (N, 강도) 계열의 16개 케이스: 배치 2 × P 장수 4 × (one-N, All-P)."""
    negative_id = group["image_id_by_name"][negative]
    series_id = f"{group_id}__N{negative_id}__{strength}"
    cases = []
    for k in config["p_counts"]:
        for placement in config["placements"]:
            prefix = series[:k]
            if placement == "N-first":
                variants = (("one_negative", [negative, *prefix], ["N", *"P" * k]),
                            ("all_positive", [series[k], *prefix], ["P"] * (k + 1)))
            else:
                variants = (("one_negative", [*prefix, negative], [*"P" * k, "N"]),
                            ("all_positive", [*prefix, series[k]], ["P"] * (k + 1)))
            for condition, images, labels in variants:
                meta = {"condition": condition, "placement": placement, "strength": strength, "series_step": k,
                        "n_images": k + 1, "anchor_negative_id": negative_id,
                        "series_image_ids": [group["image_id_by_name"][name] for name in series]}
                cases.append(make_case(group_id, group, series_id, images, labels, config, data_config, meta))
    return cases


def build_accumulation_cases(config, data_config):
    seed, length = config["seed"], config["series_length"]
    require(length == max(config["p_counts"]) + 1, "series_length must be max(p_counts) + 1")
    for name in ("Bus", "dining table"):
        require(config["prompt"].format(object=name, count=3) == data_config["prompt"].format(object=name),
                "The prompt template with count=3 must equal the STEP 1-4 prompt")
    groups = read_json(project_path(config["step4_groups"]))["groups"]
    settings = config["selection"]
    scores = read_scores(project_path(config["step4_scores"]), settings["score_layer"], settings["score_position"])
    cases, design = [], {}
    for group_id, group in sorted(groups.items()):
        strata = make_strata(group, scores, settings)
        require(strata["stratum_size"] >= length, f"{group['class']}: stratum of {strata['stratum_size']} < {length}")
        negatives = sorted(group["neg_image"], key=lambda name: group["image_id_by_name"][name])
        if config["negatives_per_class"] is not None:
            negatives = negatives[:config["negatives_per_class"]]
        series_by_negative = {}
        for negative in negatives:
            negative_id = group["image_id_by_name"][negative]
            label, _ = scores[(group["class"], negative_id)]
            require(label == "N", f"{group['class']} {negative_id}: STEP 4 label {label}, expected N")
            series_by_negative[negative_id] = {}
            for strength in config["strengths"]:
                pool = [image for _, _, image in strata[strength]]
                rng = random.Random(f"{seed}:{group_id}:{negative_id}:{strength}")
                series = rng.sample(pool, k=length)
                series_by_negative[negative_id][strength] = [group["image_id_by_name"][name] for name in series]
                cases += series_cases(group_id, group, negative, series, strength, config, data_config)
        design[group_id] = {
            "class": group["class"], "valid_P": strata["valid"], "stratum_size": strata["stratum_size"],
            **{strength: [{"image_id": image_id, "probability": probability}
                          for probability, image_id, _ in strata[strength]] for strength in config["strengths"]},
            "excluded_read_as_N": [{"image_id": image_id, "probability": probability}
                                   for probability, image_id, _ in strata["excluded_read_as_N"]],
            "series": series_by_negative,
        }
    return cases, design


def reuse_summary(cases):
    """P 이미지 재사용 구조: P 한 장이 몇 개의 N 계열에, 계열의 몇 번째로 쓰이는지."""
    series = {(case["pair_id"]): case["series_image_ids"] for case in cases}
    per_image = defaultdict(Counter)
    for image_ids in series.values():
        for index, image_id in enumerate(image_ids, start=1):
            per_image[image_id][index] += 1
    totals = [sum(counter.values()) for counter in per_image.values()]
    return {"series": len(series), "distinct_P_images": len(per_image),
            "series_per_P_image": {"min": min(totals), "max": max(totals), "mean": sum(totals) / len(totals)},
            "per_image": {str(image_id): dict(sorted(counter.items())) for image_id, counter in sorted(per_image.items())}}


def summarize(cases, design):
    inputs = Counter((case["prompt"], tuple(case["image_ids"])) for case in cases)
    reuse = reuse_summary(cases)
    return {
        "records": len(cases), "unique_inputs": len(inputs),
        "duplicate_records": sum(count - 1 for count in inputs.values() if count > 1),
        "by_condition": dict(Counter(case["condition"] for case in cases)),
        "by_n_images": dict(sorted(Counter(case["n_images"] for case in cases).items())),
        "by_cell": dict(Counter(f"{c['condition']}|{c['placement']}|{c['strength']}" for c in cases)),
        "classes": len(design), "negatives": len({(c["class"], c["anchor_negative_id"]) for c in cases}),
        "P_images_used": reuse["distinct_P_images"], "series": reuse["series"],
        "series_per_P_image": reuse["series_per_P_image"],
        "excluded_read_as_N": sum(len(group["excluded_read_as_N"]) for group in design.values()),
        "stratum_sizes": dict(Counter(group["stratum_size"] for group in design.values())),
    }


def used_images(config, cases):
    sources = {path: sorted({i for case in read_jsonl(project_path(path)) for i in case["image_ids"]})
               for path in config["used_image_sources"]}
    sources["STEP 5-2 cases"] = sorted({i for case in cases for i in case["image_ids"]})
    union = sorted({i for ids in sources.values() for i in ids})
    return {"note": "Images used in STEP 1-5. Exclude all of them when drawing the final-test images.",
            "count": len(union), "by_source": {name: len(ids) for name, ids in sources.items()}, "image_ids": union}


def paths(config):
    output_dir = project_path(config["output_dir"])
    stem = f"step5_seed{config['seed']}"
    return {"cases": output_dir / f"{stem}_cases.jsonl", "design": output_dir / f"{stem}_design.json",
            "used": output_dir / "used_images_step1_5.json",
            "check": output_dir / f"{stem}_consistency_check.json"}


def save_accumulation_cases(config_path, data_config):
    config = read_yaml(project_path(config_path))
    cases, design = build_accumulation_cases(config, data_config)
    summary = summarize(cases, design)
    step4_images = {i for case in read_jsonl(project_path(config["step4_cases"])) for i in case["image_ids"]}
    require({i for case in cases for i in case["image_ids"]} <= step4_images, "A case uses an image outside STEP 4")
    target = paths(config)
    target["cases"].parent.mkdir(parents=True, exist_ok=True)
    with target["cases"].open("x", encoding="utf-8") as destination:
        for case in cases:
            destination.write(json.dumps(case, ensure_ascii=False) + "\n")
    write_json({"config": config, "config_path": project_relative_path(config_path), "summary": summary,
                "design": design, "reuse": reuse_summary(cases)}, target["design"])
    write_json(used_images(config, cases), target["used"])
    return target, summary


def save_vision_subset(config):
    """비전 float32 민감도 분석용 부분집합: 클래스마다 N을 seed로 고르고, 그 N의 지정 조건 케이스를 모두 담는다."""
    settings = config["vision_fp32_subset"]
    cases = list(read_jsonl(paths(config)["cases"]))
    anchors = defaultdict(set)
    for case in cases:
        anchors[case["group_id"]].add(case["anchor_negative_id"])
    chosen = set()
    for group_id, ids in sorted(anchors.items()):
        rng = random.Random(f"{settings['seed']}:vision_fp32:{group_id}")
        chosen.update((group_id, i) for i in rng.sample(sorted(ids), k=settings["negatives_per_class"]))
    subset = [case for case in cases if (case["group_id"], case["anchor_negative_id"]) in chosen
              and case["condition"] in settings["conditions"]]
    output = project_path(settings["output"])
    with output.open("x", encoding="utf-8") as destination:
        for case in subset:
            destination.write(json.dumps(case, ensure_ascii=False) + "\n")
    return output, {"records": len(subset), "negatives": len(chosen),
                    "series": len({case["pair_id"] for case in subset})}
