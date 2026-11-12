"""STEP 5-2 케이스 일관성 검사 (CPU).

같은 설정·seed로 다시 생성했을 때 저장된 케이스와 똑같은지, 그리고 설계 규칙을 지키는지 확인한다.
라벨은 STEP 4 케이스의 라벨(COCO-REM 기준으로 이미 검사됨)과 대조한다. 모델 출력은 보지 않는다.
"""
from collections import defaultdict

from src.common.data import read_json, read_jsonl
from src.common.paths import project_path
from src.data_manipulate.build_accumulation import build_accumulation_cases, paths


def check_cases(config, data_config):
    target = paths(config)
    saved = list(read_jsonl(target["cases"]))
    rebuilt, design = build_accumulation_cases(config, data_config)
    failures = []

    def fail(case_id, message):
        failures.append({"case_id": case_id, "message": message})

    if saved != rebuilt:
        fail(None, "Saved cases differ from cases rebuilt with the same config and seed")
    saved_design = read_json(target["design"])["design"]
    if {key: {k: v for k, v in value.items() if k != "series"} for key, value in saved_design.items()} != \
            {key: {k: v for k, v in value.items() if k != "series"} for key, value in design.items()}:
        fail(None, "Saved strata differ from rebuilt strata")

    step4_labels = {}
    for case in read_jsonl(project_path(config["step4_cases"])):
        for image_id, label in zip(case["image_ids"], case["image_labels"]):
            if step4_labels.setdefault((case["class"], image_id), label) != label:
                fail(case["case_id"], f"STEP 4 label conflict for image {image_id}")
    strata = {(group["class"], strength): {row["image_id"] for row in group[strength]}
              for group in design.values() for strength in config["strengths"]}
    image_dir = project_path(data_config["image_dir"])
    cells = defaultdict(set)
    keys = set()
    for case in saved:
        cid, ids, labels = case["case_id"], case["image_ids"], case["image_labels"]
        n = len(ids)
        if (case["mode"], cid) in keys:
            fail(cid, "Duplicate (mode, case_id)")
        keys.add((case["mode"], cid))
        if not (n == len(case["image_order"]) == len(labels) == case["n_images"] == case["series_step"] + 1):
            fail(cid, "Image count fields disagree")
        if len(set(ids)) != n:
            fail(cid, "Repeated image within one input")
        for name, image_id, label in zip(case["image_order"], ids, labels):
            if name != f"{image_id:012d}.jpg" or not (image_dir / name).exists():
                fail(cid, f"Image file missing or misnamed: {name}")
            if step4_labels.get((case["class"], image_id)) != label:
                fail(cid, f"Label of image {image_id} differs from STEP 4")
        if case["prompt"] != config["prompt"].format(object=case["class"], count=n):
            fail(cid, "Prompt does not match the template and image count")
        if case["gt"] != ("Yes" if "N" not in labels else "No"):
            fail(cid, "Ground truth differs from image labels")
        series, k = case["series_image_ids"], case["series_step"]
        positives = [i for i, label in zip(ids, labels) if label == "P"]
        if not set(series) <= strata[(case["class"], case["strength"])]:
            fail(cid, "Series image outside its strength stratum")
        first = case["placement"] == "N-first"
        if case["condition"] == "one_negative":
            expected = ([case["anchor_negative_id"], *series[:k]] if first else [*series[:k], case["anchor_negative_id"]])
            if labels.count("N") != 1 or ids != expected or positives != series[:k]:
                fail(cid, "one-N input does not follow the cumulative series")
        else:
            expected = [series[k], *series[:k]] if first else [*series[:k], series[k]]
            if "N" in labels or ids != expected:
                fail(cid, "All-P input does not fill the N slot with the next series image")
        cells[(case["pair_id"], case["placement"], case["condition"])].add(k)
    for key, steps in cells.items():
        if sorted(steps) != list(config["p_counts"]):
            fail(None, f"Incomplete series cell {key}: steps {sorted(steps)}")
    for (name, strength), images in strata.items():
        other = strata[(name, "Low" if strength == "High" else "High")]
        if images & other:
            fail(None, f"{name}: High and Low strata overlap")
    return {
        "check": "step5_accumulation_consistency", "status": "passed" if not failures else "failed",
        "note": "Rule check against the generator and STEP 4 labels; no model output is read.",
        "counts": {"records": len(saved), "series_cells": len(cells)},
        "failure_count": len(failures), "failures": failures[:200],
    }
