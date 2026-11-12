"""STEP 2 이미지 표현 z_i의 문맥 효과와 bf16 수치 잡음 비교 (CPU, 읽기 전용).

- 잡음: 같은 이미지·같은 선행 이미지(prefix가 동일) 슬롯끼리의 거리. 토큰이 같으므로 차이는 함께 처리한 뒤쪽
  이미지에 따른 bf16 수치 오차뿐이다 (STEP 5-2 N-first와 같은 상황).
- 문맥: 같은 이미지가 앞에 다른 이미지를 두고 위치 2·3에 올 때, 위치 1(선행 이미지 없음) 벡터와의 거리,
  그리고 같은 위치에서 선행 이미지가 다른 슬롯끼리의 거리.
"""
from collections import defaultdict
from itertools import combinations
import statistics

import torch

from src.core.probe_data import load_slots


def cosine_distance(first, second):
    """[레이어, 차원] 두 벡터의 레이어별 코사인 거리."""
    return (1 - torch.nn.functional.cosine_similarity(first.double(), second.double(), dim=-1)).tolist()


def describe(values, threshold):
    ordered = sorted(values)
    return {"n": len(ordered), "median": statistics.median(ordered),
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], "max": ordered[-1],
            "share_above_threshold": sum(value > threshold for value in ordered) / len(ordered)}


def run_context_noise(run_dir, config):
    metadata, _, features, slots = load_slots(run_dir, config["atol"], config["rtol"])
    layers = metadata["config"]["layers"]
    vector = [features[slot["record"], :, slot["position"] - 1] for slot in slots]
    by_prefix = defaultdict(list)
    by_image = defaultdict(lambda: defaultdict(list))
    for index, slot in enumerate(slots):
        by_prefix[(slot["class"], tuple(slot["prefix"]))].append(index)
        by_image[(slot["class"], slot["image_id"])][slot["position"]].append(index)

    values = defaultdict(list)                    # (비교, 위치, 레이어) -> 거리
    for (_, prefix), members in by_prefix.items():
        for a, b in combinations(members, 2):
            for layer, distance in zip(layers, cosine_distance(vector[a], vector[b])):
                values[("noise_same_prefix", len(prefix), layer)].append(distance)
    for positions in by_image.values():
        references = positions.get(1, [])
        for position in (2, 3):
            for index in positions.get(position, []):
                for reference in references[:1]:
                    for layer, distance in zip(layers, cosine_distance(vector[index], vector[reference])):
                        values[("context_vs_position1", position, layer)].append(distance)
            groups = defaultdict(list)
            for index in positions.get(position, []):
                groups[tuple(slots[index]["prefix"])].append(index)
            representatives = [members[0] for members in groups.values()]
            for a, b in combinations(representatives, 2):
                for layer, distance in zip(layers, cosine_distance(vector[a], vector[b])):
                    values[("context_different_prefix", position, layer)].append(distance)

    threshold = config["threshold"]
    noise = {layer: statistics.median(values[("noise_same_prefix", 1, layer)]) for layer in layers
             if values[("noise_same_prefix", 1, layer)]}
    rows = []
    for (comparison, position, layer), distances in sorted(values.items()):
        row = {"comparison": comparison, "position": position, "layer": layer, **describe(distances, threshold)}
        row["median_over_noise_median"] = row["median"] / noise[layer] if noise.get(layer) else None
        row["median_over_step5_pilot_max"] = row["median"] / config["step5_pilot_max_cosine_distance"]
        rows.append(row)
    return {"layers": layers, "threshold": threshold, "noise_median_position1": noise,
            "step5_pilot_max_cosine_distance": config["step5_pilot_max_cosine_distance"],
            "counts": {"slots": len(slots), "records": int(features.shape[0])}}, rows
