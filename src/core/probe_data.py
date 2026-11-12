from collections import Counter, defaultdict
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.core.feature_validation import inspect_run, load_validated_feature, require


def input_key(case):
    """동일 모델 입력 판별 키. prompt에 클래스 이름이 들어 있어 클래스도 구분된다."""
    return case["prompt"], tuple(case["image_ids"])


def load_slots(run_dir, atol=1e-5, rtol=1e-5, with_query=False, progress=False):
    """검증된 STEP 2 특징을 읽어 이미지 슬롯(케이스 × 위치) 단위로 펼친다.

    with_query=True이면 레코드별 h_q(query_features, [레이어, 차원])도 함께 반환한다.
    """
    metadata, records, checks, _ = inspect_run(run_dir)
    failed = [name for name, passed in checks.items() if not passed]
    require(not failed, f"Invalid or incomplete extraction folder: {run_dir} ({', '.join(failed)}); run validate first")
    features, slots, queries = [], [], []
    for index, record in enumerate(tqdm(records, desc="Load features", unit="record", disable=not progress)):
        payload, _ = load_validated_feature(run_dir, record, metadata, atol, rtol)
        features.append(payload["image_features"].clone())
        if with_query:
            queries.append(payload["query_features"].clone())
        del payload
        case = record["case"]
        for position, (image_id, label) in enumerate(zip(case["image_ids"], case["image_labels"]), start=1):
            slots.append({
                "record": index, "mode": case["mode"], "case_id": case["case_id"], "class": case["class"],
                "pair_id": case["pair_id"], "composition": "".join(case["image_labels"]),
                "position": position, "image_id": image_id, "label": label,
                "prefix": tuple(case["image_ids"][:position]),
            })
    if with_query:
        return metadata, records, torch.stack(features), slots, torch.stack(queries)
    return metadata, records, torch.stack(features), slots


def build_vectors(slots, features, tolerance):
    """(class, 순서 있는 prefix image_ids)가 같은 슬롯을 하나의 문맥 벡터로 묶는다.

    Decoder는 causal이므로 prefix가 같으면 이론상 같은 벡터다. 실제 값은 전체 입력 길이에 따른
    수치 오차만큼 다를 수 있어, 대표 벡터와의 최대 cosine distance를 기록하고 허용치를 넘으면 중단한다.
    """
    groups = defaultdict(list)
    for index, slot in enumerate(slots):
        groups[(slot["class"], slot["prefix"])].append(index)
    vectors, matrix = [], []
    for (name, prefix), members in groups.items():
        first = slots[members[0]]
        require(len({slots[member]["label"] for member in members}) == 1,
                f"Label differs within one context vector: {name} {prefix}")
        base = features[first["record"], :, first["position"] - 1]
        max_distance = 0.0
        if len(members) > 1:
            others = torch.stack([
                features[slots[member]["record"], :, slots[member]["position"] - 1] for member in members[1:]
            ]).double()
            distance = 1 - F.cosine_similarity(others, base.double().unsqueeze(0), dim=-1)
            max_distance = max(float(distance.max()), 0.0)
        require(max_distance <= tolerance,
                f"Duplicate context vectors differ beyond tolerance: {name} {prefix} ({max_distance:.3e})")
        for member in members:
            slots[member]["vector"] = len(vectors)
        vectors.append({
            "vector": len(vectors), "class": name, "image_id": first["image_id"], "label": first["label"],
            "position": first["position"], "prefix": prefix, "n_slots": len(members),
            "representative_record": first["record"], "max_duplicate_cosine_distance": max_distance,
        })
        matrix.append(base)
    return vectors, torch.stack(matrix)


def assign_folds(vectors, folds, seed):
    """전역 image_id 단위 Fold 배정.

    여러 클래스에 쓰이는 이미지를 먼저 배정해, 같은 이미지가 모든 클래스에서 같은 Fold에 있도록 한다.
    각 이미지는 자기 (class, label) 칸들의 현재 배정 수가 가장 적은 Fold로 간다.
    """
    require(folds >= 2, "At least two folds are required")
    labels = defaultdict(dict)
    for vector in vectors:
        labels[vector["image_id"]][vector["class"]] = vector["label"]
    rng = random.Random(seed)
    images = sorted(labels)
    rng.shuffle(images)
    images.sort(key=lambda image_id: -len(labels[image_id]))
    counts = Counter()
    image_fold = {}
    for image_id in images:
        cells = list(labels[image_id].items())
        fold = min(range(folds), key=lambda k: (sum(counts[(name, label, k)] for name, label in cells), rng.random()))
        image_fold[image_id] = fold
        for name, label in cells:
            counts[(name, label, fold)] += 1

    for vector in vectors:
        vector["fold"] = image_fold[vector["image_id"]]
        vector["context"] = "pure" if all(image_fold[i] == vector["fold"] for i in vector["prefix"]) else "mixed"

    classes = sorted({vector["class"] for vector in vectors})
    cells = []
    for name in classes:
        totals = {label: sum(counts[(name, label, k)] for k in range(folds)) for label in ("P", "N")}
        for k in range(folds):
            cells.append({
                "class": name, "fold": k,
                "val_images_P": counts[(name, "P", k)], "val_images_N": counts[(name, "N", k)],
                "train_images_P": totals["P"] - counts[(name, "P", k)],
                "train_images_N": totals["N"] - counts[(name, "N", k)],
            })
    expected = {(cell["val_images_P"], cell["val_images_N"]) for cell in cells}
    return image_fold, {
        "folds": folds, "seed": seed, "unit": "image_id (global across classes)",
        "shared_images": sum(len(value) > 1 for value in labels.values()),
        "balanced": len(expected) == 1,
        "val_image_counts": sorted(expected),
        "cells": cells,
        "image_fold": {str(image_id): image_fold[image_id] for image_id in sorted(image_fold)},
    }
