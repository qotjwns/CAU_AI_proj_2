"""STEP 3 케이스 단위 평가: Strict Discrepancy, 핵심 표, 이미지 ID 부트스트랩, 기준선, 대표 사례."""
import random
from collections import defaultdict

import torch

from src.core.feature_validation import require
from src.core.probe import auroc, cross_fit


COMPOSITIONS = ("PPP", "PPN", "PNP", "NPP", "PNN", "NNN")
# 비율 지표: (분자 마스크, 분모 마스크)
RATES = {
    "fy_negative_detection_rate": ("fy_any_detected", "fy"),
    "strict_discrepancy_rate": ("fy_strict", "fy"),
    "fy_correction_rate": ("corrected", "fy"),
    "ppp_retention_rate": ("ppp_retained", "ppp_vlm_correct"),
}
NULL_METRICS = ("image_accuracy", "image_auroc", *RATES, "probe_and_accuracy")


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def case_arrays(cases):
    return {
        "vectors": torch.tensor([case["vectors"] for case in cases]),
        "labels": torch.tensor([[label == "P" for label in case["labels"]] for case in cases]),
        "image_ids": [case["image_ids"] for case in cases],
        "vlm_yes": torch.tensor([case["prediction"] == "Yes" for case in cases]),
        "vlm_no": torch.tensor([case["prediction"] == "No" for case in cases]),
        "classes": [case["class"] for case in cases],
        "compositions": [case["composition"] for case in cases],
    }


def case_outcomes(arrays, oof, column, threshold):
    probability = oof[arrays["vectors"], column]
    labels = arrays["labels"]
    predicted = probability >= threshold
    correct = predicted == labels
    negative = ~labels
    gt_yes = labels.all(dim=1)
    and_yes = predicted.all(dim=1)
    fy = ~gt_yes & arrays["vlm_yes"]
    any_detected = (negative & ~predicted).any(dim=1)
    strict = correct.all(dim=1)
    ppp_vlm_correct = gt_yes & arrays["vlm_yes"]
    return {
        "probability": probability, "predicted": predicted, "correct_images": correct,
        "gt_yes": gt_yes, "and_yes": and_yes, "strict": strict,
        "vlm_correct": torch.where(gt_yes, arrays["vlm_yes"], arrays["vlm_no"]),
        "and_correct": and_yes == gt_yes,
        "fy": fy, "fy_any_detected": fy & any_detected,
        "fy_all_detected": fy & ~(negative & predicted).any(dim=1),
        "fy_strict": fy & strict,
        "corrected": fy & ~and_yes,
        "corrected_via_negative": fy & ~and_yes & any_detected,
        "corrected_via_positive_error": fy & ~and_yes & ~any_detected,
        "new_fy": ~gt_yes & arrays["vlm_no"] & and_yes,
        "and_fy": ~gt_yes & and_yes,
        "ppp": gt_yes, "ppp_vlm_correct": ppp_vlm_correct,
        "ppp_retained": ppp_vlm_correct & and_yes,
        "ppp_rescued": gt_yes & ~arrays["vlm_yes"] & and_yes,
        "vlm_invalid": ~arrays["vlm_yes"] & ~arrays["vlm_no"],
    }


def image_units(vectors):
    """이미지 단위 평가 단위: (class, image_id)의 위치 1 문맥 벡터."""
    units = [vector for vector in vectors if vector["position"] == 1]
    return {
        "index": torch.tensor([vector["vector"] for vector in units]),
        "labels": torch.tensor([vector["label"] == "P" for vector in units]),
        "image_ids": [vector["image_id"] for vector in units],
        "classes": [vector["class"] for vector in units],
    }


class ImageBootstrap:
    """전역 image_id를 복원추출하는 클러스터 부트스트랩.

    이미지 단위 지표는 각 단위에 해당 이미지의 추출 횟수를 가중치로 준다.
    케이스 단위 지표는 케이스에 들어 있는 세 이미지 추출 횟수의 곱을 가중치로 준다
    (이미지를 공유하는 케이스 사이의 의존성을 반영하며, 한 이미지라도 빠지면 그 케이스는 제외된다).
    """

    def __init__(self, image_ids, reps, seed):
        require(reps >= 1, "Bootstrap repetitions must be positive")
        self.index = {image_id: i for i, image_id in enumerate(sorted(set(image_ids)))}
        generator = torch.Generator().manual_seed(seed)
        n = len(self.index)
        draws = torch.randint(n, (reps, n), generator=generator)
        self.counts = torch.zeros(reps, n, dtype=torch.float64).scatter_add_(
            1, draws, torch.ones(reps, n, dtype=torch.float64))
        self.reps = reps

    def unit_weights(self, image_ids):
        return self.counts[:, torch.tensor([self.index[i] for i in image_ids])]

    def case_weights(self, image_id_rows):
        columns = torch.tensor([[self.index[i] for i in row] for row in image_id_rows])
        return self.counts[:, columns].prod(dim=2)


def interval(weights, numerator, denominator):
    if weights is None or not bool(denominator.any()):
        return None, None, 0
    totals = weights @ denominator.double()
    hits = weights @ (numerator & denominator).double()
    valid = totals > 0
    if not bool(valid.any()):
        return None, None, 0
    values = hits[valid] / totals[valid]
    low, high = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(low), float(high), int(valid.sum())


def key_row(outcomes, select, unit_correct, unit_select, weights=None, unit_weights=None):
    def count(name):
        return int((outcomes[name] & select).sum())

    n_cases = int(select.sum())
    n_units = int(unit_select.sum())
    row = {
        "n_cases": n_cases, "n_image_units": n_units,
        "image_accuracy": rate(int((unit_correct & unit_select).sum()), n_units),
        "vlm_accuracy": rate(count("vlm_correct"), n_cases),
        "probe_and_accuracy": rate(count("and_correct"), n_cases),
        "always_no_accuracy": rate(n_cases - count("gt_yes"), n_cases),
        "vlm_false_yes": count("fy"), "fy_any_negative_detected": count("fy_any_detected"),
        "fy_all_negatives_detected": count("fy_all_detected"), "fy_strict": count("fy_strict"),
        "fy_corrected": count("corrected"), "corrected_via_negative": count("corrected_via_negative"),
        "corrected_via_positive_error": count("corrected_via_positive_error"),
        "and_new_false_yes": count("new_fy"), "and_false_yes_total": count("and_fy"),
        "ppp_cases": count("ppp"), "ppp_vlm_correct": count("ppp_vlm_correct"),
        "ppp_retained": count("ppp_retained"), "ppp_rescued": count("ppp_rescued"),
        "vlm_invalid": count("vlm_invalid"),
    }
    for name, (numerator, denominator) in RATES.items():
        row[name] = rate(count(numerator), count(denominator))
        low, high, valid = interval(weights, outcomes[numerator] & select, outcomes[denominator] & select)
        row[f"{name}_ci_low"], row[f"{name}_ci_high"], row[f"{name}_ci_valid"] = low, high, valid
    low, high, valid = interval(unit_weights, unit_correct, unit_select)
    row["image_accuracy_ci_low"], row["image_accuracy_ci_high"], row["image_accuracy_ci_valid"] = low, high, valid
    return row


def image_scores(units, oof, column, threshold):
    probability = oof[units["index"], column]
    return probability, (probability >= threshold) == units["labels"]


def key_tables(levels, vectors, oof, layers, threshold, bootstrap):
    """레이어별(`key_table`)과 클래스·레이어별(`key_table_by_class`) 핵심 지표."""
    units = image_units(vectors)
    unit_weights = bootstrap.unit_weights(units["image_ids"]) if bootstrap else None
    classes = sorted(set(units["classes"]))
    class_images = {name: len({v["image_id"] for v in vectors if v["class"] == name}) for name in classes}
    overall, by_class, strict_rows = [], [], []
    for level, cases in levels.items():
        arrays = case_arrays(cases)
        weights = bootstrap.case_weights(arrays["image_ids"]) if bootstrap else None
        everything = torch.ones(len(cases), dtype=torch.bool)
        for column, layer in enumerate(layers):
            outcomes = case_outcomes(arrays, oof, column, threshold)
            probability, unit_correct = image_scores(units, oof, column, threshold)
            all_units = torch.ones(len(unit_correct), dtype=torch.bool)
            row = {"level": level, "layer": layer,
                   **key_row(outcomes, everything, unit_correct, all_units, weights, unit_weights)}
            row["image_auroc"] = auroc(probability, units["labels"])
            for composition in COMPOSITIONS:
                select = torch.tensor([value == composition for value in arrays["compositions"]])
                n = int(select.sum())
                row[f"vlm_accuracy_{composition}"] = rate(int((outcomes["vlm_correct"] & select).sum()), n)
                row[f"probe_and_accuracy_{composition}"] = rate(int((outcomes["and_correct"] & select).sum()), n)
            overall.append(row)
            strict_rows.append(strict_row(level, layer, "all", "all", outcomes, everything, arrays, weights))
            for name in classes:
                select = torch.tensor([value == name for value in arrays["classes"]])
                unit_select = torch.tensor([value == name for value in units["classes"]])
                class_row = key_row(outcomes, select, unit_correct, unit_select, weights, unit_weights)
                class_row["image_auroc"] = auroc(probability[unit_select], units["labels"][unit_select])
                class_row["n_class_images"] = class_images[name]
                class_row.update(image_counts(outcomes, select, arrays))
                by_class.append({"level": level, "layer": layer, "class": name, **class_row})
                strict_rows.append(strict_row(level, layer, "class", name, outcomes, select, arrays, weights))
            for composition in COMPOSITIONS:
                if composition == "PPP":
                    continue
                select = torch.tensor([value == composition for value in arrays["compositions"]])
                strict_rows.append(strict_row(level, layer, "composition", composition, outcomes, select, arrays, weights))
    return overall, by_class, strict_rows


def image_counts(outcomes, select, arrays):
    fy = (outcomes["fy"] & select).nonzero().flatten().tolist()
    images = {i for index in fy for i in arrays["image_ids"][index]}
    negatives = {i for index in fy
                 for i, positive in zip(arrays["image_ids"][index], arrays["labels"][index].tolist()) if not positive}
    return {"n_fy_images": len(images), "n_fy_negative_images": len(negatives)}


def strict_row(level, layer, group_type, group, outcomes, select, arrays, weights):
    def count(name):
        return int((outcomes[name] & select).sum())

    n_fy = count("fy")
    low, high, valid = interval(weights, outcomes["fy_strict"] & select, outcomes["fy"] & select)
    return {
        "level": level, "layer": layer, "group_type": group_type, "group": group,
        "n_false_yes": n_fy, **image_counts(outcomes, select, arrays),
        "n_any_negative_detected": count("fy_any_detected"),
        "n_all_negatives_detected": count("fy_all_detected"),
        "n_strict": count("fy_strict"),
        "strict_discrepancy_rate": rate(count("fy_strict"), n_fy),
        "strict_ci_low": low, "strict_ci_high": high, "strict_ci_valid": valid,
        "any_negative_detected_rate": rate(count("fy_any_detected"), n_fy),
        "n_corrected": count("corrected"), "corrected_via_negative": count("corrected_via_negative"),
        "corrected_via_positive_error": count("corrected_via_positive_error"),
    }


def case_record(case, outcomes, index, layer, category=None):
    probability = outcomes["probability"][index].tolist()
    predicted = outcomes["predicted"][index].tolist()
    correct_side = [p if label == "P" else 1 - p for p, label in zip(probability, case["labels"])]
    row = {
        "layer": layer, "class": case["class"], "composition": case["composition"], "mode": case["mode"],
        "case_id": case["case_id"], "pair_id": case["pair_id"], "n_records": case.get("n_records", 1),
        "image_ids": case["image_ids"], "image_files": case["image_order"], "labels": "".join(case["labels"]),
        "p_image1": probability[0], "p_image2": probability[1], "p_image3": probability[2],
        "probe_labels": "".join("P" if value else "N" for value in predicted),
        "n_images_correct": int(outcomes["correct_images"][index].sum()),
        "min_correct_probability": min(correct_side),
        "vlm_prediction": case["prediction"], "vlm_logit_margin": case["logit_margin"],
        "probe_and": "Yes" if bool(outcomes["and_yes"][index]) else "No",
    }
    if category:
        row = {"category": category, **row}
    return row


def strict_cases(cases, oof, layers, threshold):
    arrays = case_arrays(cases)
    rows = []
    for column, layer in enumerate(layers):
        outcomes = case_outcomes(arrays, oof, column, threshold)
        for index in outcomes["fy_strict"].nonzero().flatten().tolist():
            rows.append(case_record(cases[index], outcomes, index, layer))
    return rows


def examples(cases, oof, layers, threshold, per_category):
    """레이어·범주별 대표 사례. 클래스가 겹치지 않게 먼저 고르고, 남으면 확신도 순으로 채운다."""
    arrays = case_arrays(cases)
    rows = []
    for column, layer in enumerate(layers):
        outcomes = case_outcomes(arrays, oof, column, threshold)
        probability, labels = outcomes["probability"], arrays["labels"]
        correct_side = torch.where(labels, probability, 1 - probability)
        negative_max = torch.where(~labels, probability, torch.zeros_like(probability)).max(dim=1).values
        categories = {
            "fy_strict_discrepancy": (outcomes["fy_strict"], -correct_side.min(dim=1).values),
            "fy_corrected_with_image_error": (outcomes["corrected"] & ~outcomes["strict"], negative_max),
            "fy_missed_negative": (outcomes["fy"] & outcomes["and_yes"], -negative_max),
            "ppp_broken_by_probe": (outcomes["ppp_retained"].logical_not() & outcomes["ppp_vlm_correct"],
                                    probability.min(dim=1).values),
        }
        for category, (mask, order) in categories.items():
            candidates = sorted(mask.nonzero().flatten().tolist(), key=lambda i: float(order[i]))
            chosen, seen = [], set()
            for index in candidates:
                if len(chosen) < per_category and cases[index]["class"] not in seen:
                    chosen.append(index)
                    seen.add(cases[index]["class"])
            for index in candidates:
                if len(chosen) < per_category and index not in chosen:
                    chosen.append(index)
            rows.extend(case_record(cases[index], outcomes, index, layer, category) for index in chosen)
    return rows


def evaluate_layers(cases, vectors, oof, layers, threshold):
    """순열 기준선 비교용 지표 (실제 라벨과 실제 VLM 응답 기준)."""
    arrays = case_arrays(cases)
    units = image_units(vectors)
    everything = torch.ones(len(cases), dtype=torch.bool)
    result = {}
    for column, layer in enumerate(layers):
        outcomes = case_outcomes(arrays, oof, column, threshold)
        probability, unit_correct = image_scores(units, oof, column, threshold)
        row = key_row(outcomes, everything, unit_correct, torch.ones(len(unit_correct), dtype=torch.bool))
        row["image_auroc"] = auroc(probability, units["labels"])
        result[layer] = {name: row[name] for name in NULL_METRICS}
    return result


def permute_labels(vectors, seed):
    """클래스마다 (class, image_id) 단위로 P/N 라벨을 섞는다. 클래스별 P/N 수는 유지된다."""
    rng = random.Random(seed)
    mapping = {}
    for name in sorted({vector["class"] for vector in vectors}):
        labels = {}
        for vector in vectors:
            if vector["class"] == name:
                labels[vector["image_id"]] = vector["label"]
        images = sorted(labels)
        shuffled = [labels[image_id] for image_id in images]
        rng.shuffle(shuffled)
        mapping.update({(name, image_id): label for image_id, label in zip(images, shuffled)})
    return [{**vector, "label": mapping[(vector["class"], vector["image_id"])]} for vector in vectors]


def constant_baselines(observed, cases, vectors, layers):
    """학습 없는 상수 기준선: 이미지 다수 라벨, 케이스 Always-No."""
    units = image_units(vectors)
    n_p = int(units["labels"].sum())
    n_units = len(units["labels"])
    arrays = case_arrays(cases)
    gt_yes = arrays["labels"].all(dim=1)
    fy = ~gt_yes & arrays["vlm_yes"]
    rows = [
        {"baseline": "majority_image_label", "layer": None, "metric": "image_accuracy",
         "value": max(n_p, n_units - n_p) / n_units, "n_runs": 1},
        {"baseline": "majority_image_label", "layer": None, "metric": "image_auroc", "value": 0.5, "n_runs": 1},
        {"baseline": "always_no", "layer": None, "metric": "probe_and_accuracy",
         "value": float((~gt_yes).double().mean()), "n_runs": 1},
        {"baseline": "always_no", "layer": None, "metric": "fy_correction_rate",
         "value": 1.0 if bool(fy.any()) else None, "n_runs": 1},
        {"baseline": "always_no", "layer": None, "metric": "strict_discrepancy_rate",
         "value": 0.0 if bool(fy.any()) else None, "n_runs": 1},
        {"baseline": "always_no", "layer": None, "metric": "ppp_retention_rate", "value": 0.0, "n_runs": 1},
    ]
    for row in rows:
        row["observed_by_layer"] = {layer: observed[layer].get(row["metric"]) for layer in layers}
    return rows


def baselines(observed, cases, vectors, matrix, layers, image_fold, folds, settings, threshold, runs, seed):
    """상수 기준선과 라벨 순열 기준선.

    순열: 이미지 단위로 라벨을 섞어 Fold·클래스·레이어 Probe 전체를 다시 학습하고, OOF 예측을
    실제 라벨·실제 VLM 응답 기준으로 평가한다. p = (1 + #{null >= observed}) / (1 + runs).
    """
    rows = constant_baselines(observed, cases, vectors, layers)
    null_runs = []
    for run in range(runs):
        shuffled = permute_labels(vectors, seed + run)
        null_oof, _, _ = cross_fit(shuffled, matrix, layers, image_fold, folds, settings, threshold)
        for layer, metrics in evaluate_layers(cases, vectors, null_oof, layers, threshold).items():
            for metric, value in metrics.items():
                null_runs.append({"run": run, "layer": layer, "metric": metric, "value": value})
    for layer in layers:
        for metric in NULL_METRICS:
            values = [row["value"] for row in null_runs
                      if row["layer"] == layer and row["metric"] == metric and row["value"] is not None]
            target = observed[layer][metric]
            entry = {"baseline": "label_permutation", "layer": layer, "metric": metric, "n_runs": len(values),
                     "observed": target}
            if values:
                tensor = torch.tensor(values, dtype=torch.float64)
                low, high = torch.quantile(tensor, torch.tensor([0.025, 0.975], dtype=torch.float64))
                entry.update(value=float(tensor.mean()), sd=float(tensor.std()) if len(values) > 1 else None,
                             null_q025=float(low), null_q975=float(high))
                if target is not None:
                    entry["p_value"] = (1 + sum(value >= target for value in values)) / (1 + len(values))
            rows.append(entry)
    return rows, null_runs


ONE_NEGATIVE = ("PPN", "PNP", "NPP")
GROUPS = {
    "False-Yes": lambda gt_yes, yes, no, one: ~gt_yes & yes,
    "Correct-No": lambda gt_yes, yes, no, one: ~gt_yes & no,
    "Correct-No (one N)": lambda gt_yes, yes, no, one: ~gt_yes & no & one,
    "Correct-Yes": lambda gt_yes, yes, no, one: gt_yes & yes,
    "False-No": lambda gt_yes, yes, no, one: gt_yes & no,
}
GROUP_METRICS = {"strict": "strict", "negative_detection": "any_detected"}


def weighted_ratio(weights, numerator, denominator):
    totals = weights @ denominator.double()
    return (weights @ (numerator & denominator).double()) / totals, totals > 0


def quantiles(values):
    low, high = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(low), float(high)


def matched_difference(weights, metric, first, second, cells, n_cells):
    """first(False-Yes) vs second(Correct-No)를 (class, composition) 칸별로 비교한다.

    second는 first의 칸 분포로 표준화하고, 두 그룹이 모두 있는 칸만 사용한다. weights: [reps, cases].
    """
    def cell_sums(mask):
        index = cells.expand(weights.shape[0], -1)
        base = torch.zeros(weights.shape[0], n_cells, dtype=torch.float64)
        total = base.clone().scatter_add_(1, index, weights * mask.double())
        hits = base.clone().scatter_add_(1, index, weights * (mask & metric).double())
        return total, hits

    first_total, first_hits = cell_sums(first)
    second_total, second_hits = cell_sums(second)
    covered = (first_total > 0) & (second_total > 0)
    first_weight = torch.where(covered, first_total, torch.zeros_like(first_total))
    norm = first_weight.sum(dim=1)
    first_rate = torch.where(covered, first_hits, torch.zeros_like(first_hits)).sum(dim=1) / norm
    second_cell = torch.where(covered, second_hits / second_total.clamp_min(1e-300), torch.zeros_like(second_hits))
    second_rate = (first_weight * second_cell).sum(dim=1) / norm
    return first_rate, second_rate, norm > 0


def group_comparison(levels, oof, layers, threshold, bootstrap):
    """VLM 응답 그룹별 Probe 판별 가능성 비교 (그룹 비율, 클래스·구성 맞춤 차이, 같은 N 이미지 짝 비교)."""
    rows = []
    for level, cases in levels.items():
        arrays = case_arrays(cases)
        weights = bootstrap.case_weights(arrays["image_ids"])
        ones = torch.ones(1, len(cases), dtype=torch.float64)
        gt_yes = arrays["labels"].all(dim=1)
        one = torch.tensor([value in ONE_NEGATIVE for value in arrays["compositions"]])
        masks = {name: rule(gt_yes, arrays["vlm_yes"], arrays["vlm_no"], one) for name, rule in GROUPS.items()}
        keys = sorted({(name, composition) for name, composition in zip(arrays["classes"], arrays["compositions"])})
        cell_index = {key: i for i, key in enumerate(keys)}
        cells = torch.tensor([cell_index[key] for key in zip(arrays["classes"], arrays["compositions"])]).unsqueeze(0)
        negative_column = [labels.index(False) if labels.count(False) == 1 else None
                           for labels in arrays["labels"].tolist()]
        for column, layer in enumerate(layers):
            outcomes = case_outcomes(arrays, oof, column, threshold)
            outcomes["any_detected"] = (~arrays["labels"] & ~outcomes["predicted"]).any(dim=1)
            base = {"level": level, "layer": layer}
            for group, mask in masks.items():
                for metric, key in GROUP_METRICS.items():
                    if metric == "negative_detection" and group in ("Correct-Yes", "False-No"):
                        continue
                    n = int(mask.sum())
                    value = rate(int((outcomes[key] & mask).sum()), n)
                    low, high, valid = interval(weights, outcomes[key] & mask, mask)
                    rows.append({**base, "analysis": "group", "metric": metric, "group": group, "n_cases": n,
                                 "value": value, "ci_low": low, "ci_high": high, "ci_target": "value",
                                 "ci_valid": valid})
            first, second = masks["False-Yes"], masks["Correct-No (one N)"]
            for metric, key in GROUP_METRICS.items():
                point_first, point_second, ok = matched_difference(ones, outcomes[key], first, second, cells, len(keys))
                boot_first, boot_second, boot_ok = matched_difference(weights, outcomes[key], first, second, cells, len(keys))
                difference = (boot_first - boot_second)[boot_ok]
                covered_cells = [i for i in range(len(keys))
                                 if bool((first & (cells[0] == i)).any()) and bool((second & (cells[0] == i)).any())]
                covered = torch.tensor([int(c) in covered_cells for c in cells[0]])
                low, high = quantiles(difference) if len(difference) else (None, None)
                rows.append({
                    **base, "analysis": "matched_class_composition", "metric": metric, "group": "False-Yes",
                    "comparison_group": "Correct-No (one N), standardized to False-Yes cells",
                    "n_cases": int((first & covered).sum()), "n_comparison_cases": int((second & covered).sum()),
                    "n_units": len(covered_cells),
                    "value": float(point_first[0]) if bool(ok[0]) else None,
                    "comparison_value": float(point_second[0]) if bool(ok[0]) else None,
                    "difference": float(point_first[0] - point_second[0]) if bool(ok[0]) else None,
                    "ci_low": low, "ci_high": high, "ci_target": "difference", "ci_valid": int(boot_ok.sum()),
                })
            units = defaultdict(lambda: {"False-Yes": [], "Correct-No (one N)": []})
            for index, position in enumerate(negative_column):
                if position is None:
                    continue
                for group in ("False-Yes", "Correct-No (one N)"):
                    if bool(masks[group][index]):
                        key = (arrays["classes"][index], arrays["image_ids"][index][position])
                        units[key][group].append(index)
            paired = [(key, value) for key, value in units.items() if value["False-Yes"] and value["Correct-No (one N)"]]
            for metric, key in GROUP_METRICS.items():
                if not paired:
                    continue
                first_mean = torch.tensor([float(outcomes[key][value["False-Yes"]].double().mean()) for _, value in paired])
                second_mean = torch.tensor([float(outcomes[key][value["Correct-No (one N)"]].double().mean())
                                            for _, value in paired])
                unit_weights = bootstrap.unit_weights([unit[1] for unit, _ in paired])
                totals = unit_weights.sum(dim=1)
                valid = totals > 0
                difference = ((unit_weights @ (first_mean - second_mean).double()) / totals)[valid]
                low, high = quantiles(difference)
                delta = first_mean - second_mean
                rows.append({
                    **base, "analysis": "paired_negative_image", "metric": metric, "group": "False-Yes",
                    "comparison_group": "Correct-No (one N), same (class, N image)",
                    "n_cases": sum(len(value["False-Yes"]) for _, value in paired),
                    "n_comparison_cases": sum(len(value["Correct-No (one N)"]) for _, value in paired),
                    "n_units": len(paired), "value": float(first_mean.mean()),
                    "comparison_value": float(second_mean.mean()), "difference": float(delta.mean()),
                    "ci_low": low, "ci_high": high, "ci_target": "difference", "ci_valid": int(valid.sum()),
                    "greater": int((delta > 0).sum()), "equal": int((delta == 0).sum()), "less": int((delta < 0).sum()),
                })
    return rows


def negative_positions(arrays):
    """N 1장 구성이면 N의 위치(0~2), 아니면 -1."""
    labels = arrays["labels"]
    one = (~labels).sum(dim=1) == 1
    return torch.where(one, (~labels).double().argmax(dim=1), torch.full((len(labels),), -1)), one


def localization_hits(arrays, predictions, column):
    position, one = negative_positions(arrays)
    probability = predictions[arrays["vectors"], column]
    return (probability.argmin(dim=1) == position) & one, one


def localization(cases, predictions, layers, bootstrap):
    """N 1장 구성(PPN·PNP·NPP)에서 확률이 가장 낮은 위치가 N 위치와 같은 비율 (우연 1/3).

    위치를 모르고 'N이 있다'만 알아서는 맞힐 수 없으므로, 구성 패턴에 기대지 않는 위치 판별 지표다.
    """
    arrays = case_arrays(cases)
    weights = bootstrap.case_weights(arrays["image_ids"]) if bootstrap else None
    gt_yes = arrays["labels"].all(dim=1)
    scopes = [("all", "all", torch.ones(len(cases), dtype=torch.bool))]
    scopes += [("composition", value, torch.tensor([c == value for c in arrays["compositions"]])) for value in ONE_NEGATIVE]
    scopes += [("vlm_group", "False-Yes", ~gt_yes & arrays["vlm_yes"]), ("vlm_group", "Correct-No", ~gt_yes & arrays["vlm_no"])]
    rows = []
    for column, layer in enumerate(layers):
        hits, one = localization_hits(arrays, predictions, column)
        for scope, value, mask in scopes:
            mask = mask & one
            n = int(mask.sum())
            low, high, valid = interval(weights, hits & mask, mask)
            rows.append({"layer": layer, "scope": scope, "value": value, "n": n,
                         "accuracy": rate(int((hits & mask).sum()), n), "chance": 1 / 3,
                         "ci_low": low, "ci_high": high, "ci_valid": valid})
    return rows


def mcnemar_exact(first_only, second_only):
    """양측 정확 McNemar p (불일치 쌍의 이항검정). 케이스 독립을 가정하므로 참고용."""
    from math import comb
    n = first_only + second_only
    if not n:
        return None
    k = min(first_only, second_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_feature_comparison(cases, first, second, threshold, bootstrap):
    """같은 입력에서 두 특징(예: z_i L36 vs h_q L27)의 성공 여부를 짝지어 비교한다.

    first/second: (이름, 예측 행렬, 레이어 열 index, 레이어). 차이 CI는 이미지 클러스터 부트스트랩.
    """
    arrays = case_arrays(cases)
    weights = bootstrap.case_weights(arrays["image_ids"]) if bootstrap else None
    gt_yes = arrays["labels"].all(dim=1)
    success = {}
    for name, predictions, column, _ in (first, second):
        outcomes = case_outcomes(arrays, predictions, column, threshold)
        hits, _ = localization_hits(arrays, predictions, column)
        success[name] = {"strict": outcomes["strict"], "localization": hits}
    fy = ~gt_yes & arrays["vlm_yes"]
    cn = ~gt_yes & arrays["vlm_no"]
    _, one = negative_positions(arrays)
    subsets = (("strict", "False-Yes", fy), ("strict", "Correct-No (one N)", cn & one),
               ("localization", "one-N all", one), ("localization", "False-Yes", fy & one),
               ("localization", "Correct-No", cn & one))
    rows = []
    for metric, subset, mask in subsets:
        a, b = success[first[0]][metric], success[second[0]][metric]
        n = int(mask.sum())
        row = {"first": f"{first[0]} L{first[3]}", "second": f"{second[0]} L{second[3]}", "metric": metric,
               "subset": subset, "n": n,
               "both": int((a & b & mask).sum()), "first_only": int((a & ~b & mask).sum()),
               "second_only": int((~a & b & mask).sum()), "neither": int((~a & ~b & mask).sum())}
        row["first_rate"] = rate(int((a & mask).sum()), n)
        row["second_rate"] = rate(int((b & mask).sum()), n)
        row["difference"] = None if not n else row["first_rate"] - row["second_rate"]
        row["mcnemar_p_independent"] = mcnemar_exact(row["first_only"], row["second_only"])
        row["ci_low"] = row["ci_high"] = None
        row["ci_valid"] = 0
        if weights is not None and n:
            totals = weights @ mask.double()
            valid = totals > 0
            difference = ((weights @ (a & mask).double()) - (weights @ (b & mask).double()))[valid] / totals[valid]
            if len(difference):
                row["ci_low"], row["ci_high"] = quantiles(difference)
                row["ci_valid"] = int(valid.sum())
        rows.append(row)
    return rows
