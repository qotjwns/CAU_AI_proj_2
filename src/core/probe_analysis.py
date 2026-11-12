from collections import defaultdict
from statistics import fmean

import torch

from src.core.feature_analysis import extraction_signature
from src.core.feature_validation import require
from src.core.probe import cross_fit, score
from src.core.probe_cases import (
    ImageBootstrap, baselines, evaluate_layers, examples, group_comparison, key_tables, strict_cases,
)
from src.core.probe_data import assign_folds, build_vectors, input_key, load_slots


INDEPENDENCE_SCOPE = (
    "Each evaluated target image_id was never used by that probe's training data, either as a target "
    "or inside a prefix. Validation prefixes may contain training images, so this is not an evaluation "
    "with fully independent model inputs."
)
DEFINITIONS = {
    "context_vector": "Unique (class, ordered prefix image_ids); duplicate slots share one representative vector.",
    "image_level": "Position-1 context vector of each (class, image_id); no preceding image.",
    "image_context_mean": "Mean OOF probability over all context vectors of one (class, image_id).",
    "context_pure": "All prefix images belong to the target image's validation fold.",
    "context_mixed": "The prefix contains at least one image outside the target image's validation fold.",
    "paired_context": "Targets observed in both pure and mixed contexts at the same position; per-target means.",
    "false_yes": "Ground truth No and VLM prediction Yes.",
    "correct_no": "Ground truth No and VLM prediction No.",
    "negative_detection": "OOF probability of a negative (N) image below the threshold.",
    "probe_and": "Predict Yes only if all three image probabilities are at or above the threshold.",
    "unique_input": "Records deduplicated by (prompt, ordered image_ids); OOF probabilities are identical within a group.",
    "vlm_invalid": "Unparsed VLM responses count as incorrect.",
    "strict_discrepancy": "VLM False-Yes and the probe classifies all three images correctly; rate = count / VLM False-Yes.",
    "fy_negative_detection_rate": "VLM False-Yes cases where at least one N image has p < threshold, divided by VLM False-Yes.",
    "fy_correction_rate": "VLM False-Yes cases where Probe AND answers No, divided by VLM False-Yes. Split into "
                          "corrected_via_negative (an N image detected) and corrected_via_positive_error (only P images flipped).",
    "and_new_false_yes": "Ground truth No, VLM No, Probe AND Yes.",
    "ppp_retention_rate": "PPP cases answered Yes by the VLM that Probe AND also answers Yes, divided by VLM-correct PPP.",
    "bootstrap_ci": "95% percentile interval from resampling global image_ids with replacement. Image-level units are "
                    "weighted by their image's draw count; cases by the product of their three images' draw counts.",
    "label_permutation": "Labels shuffled per class at the (class, image_id) level keeping P/N counts; all 320 probes "
                         "are retrained and OOF predictions are scored against true labels and true VLM responses.",
}


def correct_probability(probability, label):
    return probability if label == "P" else 1 - probability


def scored_row(oof, column, members, threshold, **labels):
    result = score([oof[v["vector"], column] for v in members], [v["label"] == "P" for v in members], threshold)
    return {**labels, **result, "n_targets": len({(v["class"], v["image_id"]) for v in members}),
            "n_slots": sum(v["n_slots"] for v in members)}


def image_level(vectors, oof, layers, threshold):
    classes = sorted({vector["class"] for vector in vectors})
    first = [vector for vector in vectors if vector["position"] == 1]
    contexts = defaultdict(list)
    for vector in vectors:
        contexts[(vector["class"], vector["image_id"])].append(vector)
    rows = []
    for column, layer in enumerate(layers):
        for scope in ["all", *classes]:
            members = [v for v in first if scope == "all" or v["class"] == scope]
            rows.append(scored_row(oof, column, members, threshold, layer=layer, cls=scope, measure="position1"))
            keys = [key for key in contexts if scope == "all" or key[0] == scope]
            means = [fmean(float(oof[v["vector"], column]) for v in contexts[key]) for key in keys]
            result = score(means, [contexts[key][0]["label"] == "P" for key in keys], threshold)
            rows.append({"layer": layer, "cls": scope, "measure": "context_mean", **result,
                         "n_targets": len(keys), "n_slots": sum(v["n_slots"] for key in keys for v in contexts[key])})
    return rows


def context_level(vectors, oof, layers, threshold):
    classes = sorted({vector["class"] for vector in vectors})
    rows = []
    for column, layer in enumerate(layers):
        rows.append(scored_row(oof, column, vectors, threshold, layer=layer, cls="all", position="all", context="all"))
        for position in (1, 2, 3):
            at_position = [v for v in vectors if v["position"] == position]
            rows.append(scored_row(oof, column, at_position, threshold,
                                   layer=layer, cls="all", position=position, context="all"))
            if position > 1:
                for context in ("pure", "mixed"):
                    members = [v for v in at_position if v["context"] == context]
                    rows.append(scored_row(oof, column, members, threshold,
                                           layer=layer, cls="all", position=position, context=context))
        for name in classes:
            members = [v for v in vectors if v["class"] == name]
            rows.append(scored_row(oof, column, members, threshold, layer=layer, cls=name, position="all", context="all"))
    return rows


def paired_context(vectors, oof, layers, threshold):
    targets = defaultdict(lambda: {"pure": [], "mixed": []})
    for vector in vectors:
        if vector["position"] > 1:
            targets[(vector["class"], vector["image_id"], vector["position"])][vector["context"]].append(vector)
    rows = []
    for column, layer in enumerate(layers):
        for position in (2, 3):
            pairs = [value for key, value in targets.items() if key[2] == position and value["pure"] and value["mixed"]]
            summary = {"layer": layer, "position": position, "n_targets": len(pairs),
                       "n_pure_vectors": sum(len(p["pure"]) for p in pairs),
                       "n_mixed_vectors": sum(len(p["mixed"]) for p in pairs)}
            if pairs:
                stats = []
                for pair in pairs:
                    entry = {}
                    for context in ("pure", "mixed"):
                        members = pair[context]
                        probabilities = [float(oof[v["vector"], column]) for v in members]
                        entry[f"{context}_accuracy"] = fmean(
                            (p >= threshold) == (v["label"] == "P") for p, v in zip(probabilities, members)
                        )
                        entry[f"{context}_probability"] = fmean(
                            correct_probability(p, v["label"]) for p, v in zip(probabilities, members)
                        )
                    stats.append(entry)
                difference = [entry["mixed_accuracy"] - entry["pure_accuracy"] for entry in stats]
                summary.update({
                    "mean_pure_accuracy": fmean(e["pure_accuracy"] for e in stats),
                    "mean_mixed_accuracy": fmean(e["mixed_accuracy"] for e in stats),
                    "mean_accuracy_difference": fmean(difference),
                    "mean_pure_correct_probability": fmean(e["pure_probability"] for e in stats),
                    "mean_mixed_correct_probability": fmean(e["mixed_probability"] for e in stats),
                    "mixed_better": sum(d > 0 for d in difference), "equal": sum(d == 0 for d in difference),
                    "pure_better": sum(d < 0 for d in difference),
                })
            rows.append(summary)
    return rows


def case_table(records, slots):
    by_record = defaultdict(list)
    for slot in slots:
        by_record[slot["record"]].append(slot)
    cases = []
    for index, record in enumerate(records):
        members = sorted(by_record[index], key=lambda slot: slot["position"])
        case = record["case"]
        cases.append({
            "record": index, "mode": case["mode"], "case_id": case["case_id"], "class": case["class"],
            "pair_id": case["pair_id"], "composition": "".join(case["image_labels"]), "gt": case["gt"],
            "prediction": record["prediction"], "logit_margin": record["logit_margin"],
            "input_key": input_key(case), "image_ids": list(case["image_ids"]),
            "image_order": list(case["image_order"]), "vectors": [slot["vector"] for slot in members],
            "labels": [slot["label"] for slot in members],
        })
    return cases


def unique_cases(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[case["input_key"]].append(case)
    inconsistent = sum(len({case["prediction"] for case in members}) > 1 for members in groups.values())
    for members in groups.values():
        require(len({tuple(case["vectors"]) for case in members}) == 1, "Duplicate inputs map to different vectors")
    return [{**members[0], "n_records": len(members)} for members in groups.values()], inconsistent


def denominator_trace(cases, unique, oof):
    """2,800 레코드에서 고유 입력까지 줄어드는 단계와 사유."""
    within_pair = {(case["pair_id"], case["input_key"]) for case in cases}
    groups = defaultdict(set)
    for pair_id, key in within_pair:
        groups[key].add(pair_id)
    composition = {case["input_key"]: case["composition"] for case in cases}
    cross = defaultdict(int)
    for key, pairs in groups.items():
        if len(pairs) > 1:
            cross[composition[key]] += len(pairs) - 1
    missing = sum(bool(torch.isnan(oof[case["vectors"]]).any()) for case in cases)
    return {
        "records": len(cases),
        "excluded": {"missing_oof_prediction": missing},
        "invalid_vlm_predictions_kept_as_incorrect": sum(case["prediction"] is None for case in cases),
        "within_pair_duplicates": len(cases) - len(within_pair),
        "after_within_pair_dedup": len(within_pair),
        "cross_pair_duplicates": sum(cross.values()),
        "cross_pair_duplicates_by_composition": dict(sorted(cross.items())),
        "unique_inputs": len(unique),
        "explanation": (
            "Records are deduplicated by (prompt, ordered image_ids). Within one pair, mode A PPN and mode B PPN "
            "are the same input. Across pairs of the same class, independently built pairs can repeat the same "
            "ordered image triple because each class has only 8 P and 8 N images. No case is excluded."
        ),
    }


def false_yes(levels, oof, vectors, layers, threshold):
    rows, probabilities = [], {}
    for level, cases in levels.items():
        probabilities[level] = {}
        for column, layer in enumerate(layers):
            probabilities[level][layer] = {}
            for group, prediction in (("False-Yes", "Yes"), ("Correct-No", "No")):
                selected = [case for case in cases if case["gt"] == "No" and case["prediction"] == prediction]
                values, all_detected, contexts = [], [], defaultdict(list)
                for case in selected:
                    negatives = [float(oof[v, column]) for v, label in zip(case["vectors"], case["labels"]) if label == "N"]
                    values.extend(negatives)
                    all_detected.append(all(value < threshold for value in negatives))
                    for v, label in zip(case["vectors"], case["labels"]):
                        if label == "N":
                            contexts[vectors[v]["context"]].append(float(oof[v, column]) < threshold)
                rows.append({
                    "level": level, "layer": layer, "vlm_group": group, "n_cases": len(selected),
                    "n_negative_slots": len(values),
                    "n_negative_images": len({(case["class"], vectors[v]["image_id"]) for case in selected
                                              for v, label in zip(case["vectors"], case["labels"]) if label == "N"}),
                    "negative_detection_rate": fmean(value < threshold for value in values) if values else None,
                    "all_negatives_detected_rate": fmean(all_detected) if all_detected else None,
                    "mean_negative_probability": fmean(values) if values else None,
                    "pure_negative_slots": len(contexts["pure"]),
                    "pure_negative_detection_rate": fmean(contexts["pure"]) if contexts["pure"] else None,
                    "mixed_negative_slots": len(contexts["mixed"]),
                    "mixed_negative_detection_rate": fmean(contexts["mixed"]) if contexts["mixed"] else None,
                })
                probabilities[level][layer][group] = values
    return rows, probabilities


def probe_and(levels, oof, layers, threshold):
    rows = []
    for level, cases in levels.items():
        compositions = sorted({case["composition"] for case in cases})
        scopes = [("all", "all", lambda case: True)]
        scopes += [("gt", value, lambda case, value=value: case["gt"] == value) for value in ("Yes", "No")]
        scopes += [("mode", value, lambda case, value=value: case["mode"] == value) for value in ("A", "B")]
        scopes += [("composition", value, lambda case, value=value: case["composition"] == value) for value in compositions]
        for column, layer in enumerate(layers):
            decisions = {
                case["record"]: "Yes" if all(float(oof[v, column]) >= threshold for v in case["vectors"]) else "No"
                for case in cases
            }
            for scope, value, keep in scopes:
                selected = [case for case in cases if keep(case)]
                if not selected:
                    continue
                vlm = [case["prediction"] == case["gt"] for case in selected]
                probe = [decisions[case["record"]] == case["gt"] for case in selected]
                rows.append({
                    "level": level, "layer": layer, "scope": scope, "value": value, "n": len(selected),
                    "vlm_accuracy": fmean(vlm), "probe_and_accuracy": fmean(probe),
                    "both_correct": sum(v and p for v, p in zip(vlm, probe)),
                    "vlm_only_correct": sum(v and not p for v, p in zip(vlm, probe)),
                    "probe_only_correct": sum(p and not v for v, p in zip(vlm, probe)),
                    "both_wrong": sum(not v and not p for v, p in zip(vlm, probe)),
                    "vlm_invalid": sum(case["prediction"] is None for case in selected),
                })
    return rows


def compare_reference(reference, image_rows, context_rows, signature, layers):
    if not reference:
        return None
    require(reference["comparison_status"] == "reference_only", "Historical probe values support reference comparison only")
    category = reference["category"]
    rows = []
    for layer in layers:
        image = next((row for row in image_rows if row["layer"] == layer and row["cls"] == category
                      and row["measure"] == "position1"), None)
        context = next((row for row in context_rows if row["layer"] == layer and row["cls"] == category), None)
        rows.append({
            "layer": layer, "category": category,
            "reference_accuracy_low": reference["validation_accuracy_range"][0],
            "reference_accuracy_high": reference["validation_accuracy_range"][1],
            "current_image_accuracy": image["accuracy"] if image else None,
            "current_image_n": image["n"] if image else 0,
            "current_context_accuracy": context["accuracy"] if context else None,
            "current_context_n": context["n"] if context else 0,
        })
    return {
        "status": reference["comparison_status"], "source": reference["source"], "category": category,
        "reference_split": reference["split"], "notes": reference.get("notes", []),
        "limitations": reference["limitations"],
        "model_and_layers_compatible": (signature["model_name"] == reference["model_name"]
                                        and signature["layers"] == reference["layers"]),
        "rows": rows,
    }


def run_probe_study(run_dir, config, reference=None, permutation_runs=None):
    atol, rtol = config["verification"]["atol"], config["verification"]["rtol"]
    threshold = config["threshold"]
    metadata, records, features, slots = load_slots(run_dir, atol, rtol)
    signature = extraction_signature(metadata)
    layers = signature["layers"]
    vectors, matrix = build_vectors(slots, features, config["duplicate_cosine_tolerance"])
    del features
    image_fold, split = assign_folds(vectors, config["folds"], config["seed"])
    for slot in slots:
        vector = vectors[slot["vector"]]
        slot.update(fold=vector["fold"], context=vector["context"], prediction=records[slot["record"]]["prediction"])
    oof, probes, checkpoints = cross_fit(vectors, matrix, layers, image_fold, config["folds"], config["probe"], threshold)

    evaluation = evaluate_predictions(records, slots, vectors, oof, layers, threshold, config)
    unique = evaluation["unique"]
    observed = evaluate_layers(unique, vectors, oof, layers, threshold)
    runs = config["permutation_runs"] if permutation_runs is None else permutation_runs
    baseline_rows, null_runs = baselines(
        observed, unique, vectors, matrix, layers, image_fold, config["folds"], config["probe"], threshold,
        runs, config["seed"] + 1000,
    )

    report = {
        "status": config["status"], "independence_scope": INDEPENDENCE_SCOPE, "definitions": DEFINITIONS,
        "run_dir": str(run_dir), "extraction_settings": signature,
        "source": {"versions": metadata["versions"], "device": metadata["device"],
                   "attention_implementation": metadata["attention_implementation"]},
        "settings": {
            **{key: config[key] for key in ("folds", "seed", "threshold", "probe", "duplicate_cosine_tolerance",
                                            "bootstrap_reps", "examples_per_category")},
            "permutation_runs": runs,
        },
        "counts": {
            "records": len(records), "slots": len(slots), "context_vectors": len(vectors),
            "images": len({vector["image_id"] for vector in vectors}),
            "class_images": len({(vector["class"], vector["image_id"]) for vector in vectors}),
            "classes": len({vector["class"] for vector in vectors}),
            "unique_inputs": len(unique),
            "unique_inputs_with_inconsistent_prediction": evaluation["inconsistent"],
            "probes_trained": len(probes), "probes_not_converged": sum(not row["converged"] for row in probes),
            "max_duplicate_cosine_distance": max(vector["max_duplicate_cosine_distance"] for vector in vectors),
        },
        "split": {key: value for key, value in split.items() if key not in ("cells", "image_fold")},
        **evaluation["sections"],
        "baselines": baseline_rows,
        "reference_comparison": compare_reference(
            reference, evaluation["sections"]["image_level"], evaluation["sections"]["context_level"], signature, layers),
    }
    return {
        "report": report, "split": split, "vectors": vectors, "slots": slots, "oof": oof,
        "probes": probes, "checkpoints": checkpoints, "layers": layers,
        "strict_cases": evaluation["strict_cases"], "examples": evaluation["examples"], "null_runs": null_runs,
    }


def evaluate_predictions(records, slots, vectors, predictions, layers, threshold, config):
    """확률 행렬(문맥 벡터 × 레이어)에서 STEP 3·4 공통 평가 표를 만든다.

    predictions는 STEP 3에서는 OOF 예측, STEP 4에서는 고정 Probe 예측이다.
    """
    cases = case_table(records, slots)
    unique, inconsistent = unique_cases(cases)
    levels = {"record": cases, "unique_input": unique}
    image_rows = image_level(vectors, predictions, layers, threshold)
    context_rows = context_level(vectors, predictions, layers, threshold)
    false_yes_rows, negative_probabilities = false_yes(levels, predictions, vectors, layers, threshold)
    bootstrap = ImageBootstrap([vector["image_id"] for vector in vectors], config["bootstrap_reps"], config["seed"])
    key_rows, class_rows, strict_rows = key_tables(levels, vectors, predictions, layers, threshold, bootstrap)
    sections = {
        "denominators": denominator_trace(cases, unique, predictions),
        "headline": [row for row in key_rows if row["level"] == "unique_input"],
        "key_table": key_rows, "key_table_by_class": class_rows, "strict_discrepancy": strict_rows,
        "group_comparison": group_comparison(levels, predictions, layers, threshold, bootstrap),
        "image_level": image_rows, "context_level": context_rows,
        "paired_context": paired_context(vectors, predictions, layers, threshold),
        "false_yes": false_yes_rows, "probe_and": probe_and(levels, predictions, layers, threshold),
        "negative_probabilities": negative_probabilities,
    }
    return {
        "sections": sections, "unique": unique, "inconsistent": inconsistent,
        "strict_cases": strict_cases(unique, predictions, layers, threshold),
        "examples": examples(unique, predictions, layers, threshold, config["examples_per_category"]),
    }
