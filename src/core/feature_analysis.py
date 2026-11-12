from collections import defaultdict
from itertools import combinations
import math
from statistics import fmean, median, stdev

import torch

from src.common.data import normalize_category
from src.core.feature_validation import inspect_run, load_validated_feature, require
from src.core.metrics import COMPOSITIONS


TARGET_COMPOSITIONS = {"A": ("PPN", "PNN", "NNN"), "B": ("NPP", "PNP", "PPN")}


def extraction_signature(metadata):
    config = metadata["config"]
    return {
        "model_name": config["model_name"], "layers": config["layers"],
        "hidden_size": config["expected_hidden_size"], "pooling": metadata["pooling"],
        "layer_convention": metadata["layer_convention"],
        "final_layer_after_norm": metadata["final_layer_after_norm"],
        "dtype": metadata["dtype"], "image_processor": metadata["image_processor"],
        "chat_template": metadata["chat_template"],
        "model_config": {key: value for key, value in metadata["model_config"].items()
                         if key not in ("_name_or_path", "transformers_version")},
    }


def find_target(mode, records):
    cases = [records[name]["case"] for name in TARGET_COMPOSITIONS[mode]]
    require(len({case["prompt"] for case in cases}) == 1, "Prompts differ within one comparison group")
    labels = {}
    for case in cases:
        for image_id, label in zip(case["image_ids"], case["image_labels"]):
            require(image_id not in labels or labels[image_id] == label, "Image label changes within one group")
            labels[image_id] = label
    negative_ids = [
        {image_id for image_id, label in zip(case["image_ids"], case["image_labels"]) if label == "N"}
        for case in cases
    ]
    common = set.intersection(*negative_ids)
    require(len(common) == 1, "Expected exactly one shared negative image N1")
    target = next(iter(common))
    if mode == "A":
        ppn, pnn, nnn = (case["image_ids"] for case in cases)
        require(all(ids[2] == target for ids in (ppn, pnn, nnn)), "Mode A: N1 is not fixed at image 3")
        require(ppn[0] == pnn[0] and pnn[1] == nnn[1], "Mode A context replacement pattern differs")
    else:
        require(all(set(case["image_ids"]) == set(cases[0]["image_ids"]) for case in cases),
                "Mode B cases must contain the same three images")
        require([case["image_ids"].index(target) for case in cases] == [0, 1, 2],
                "Mode B: expected N1 at positions 1, 2 and 3")
    return target


def cosine_distance(left, right):
    left, right = left.double(), right.double()
    left_norm, right_norm = torch.linalg.vector_norm(left), torch.linalg.vector_norm(right)
    require(bool(left_norm > 0 and right_norm > 0), "Cosine distance is undefined for zero vectors")
    similarity = torch.dot(left / left_norm, right / right_norm).clamp(-1, 1)
    return float(1 - similarity)


def summarize_distances(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["mode"], row["category"], row["comparison"], row["layer"])
        groups[(*key, "All")].append(row["cosine_distance"])
        groups[(*key, row["response_group"])].append(row["cosine_distance"])
    summary = []
    for (mode, category, comparison, layer, response_group), values in sorted(groups.items()):
        summary.append({
            "mode": mode, "category": category, "comparison": comparison, "layer": layer,
            "response_group": response_group, "n": len(values), "mean": fmean(values),
            "median": median(values), "std": stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values),
        })
    return summary


def compare_reference(summary, records, signature, reference):
    require(reference["comparison_status"] == "reference_only", "Historical summary supports reference comparison only")
    require(reference["mode_a_grouping"] == "PPN_response_estimated", "Expected estimated PPN grouping for historical Mode A")
    category = normalize_category(reference["category"])
    selected = [record for record in records if normalize_category(record["case"]["class"]) == category]
    cases = [record["case"] for record in selected]
    counts = {
        "cases": len(cases),
        "unique_inputs": len({(case["class"], case["prompt"], tuple(case["image_ids"])) for case in cases}),
        "unique_images": len({image_id for case in cases for image_id in case["image_ids"]}),
    }
    compatible = (signature["model_name"] == reference["model_name"]
                  and signature["layers"] == reference["layers"]
                  and signature["hidden_size"] == reference["hidden_size"])
    lookup = {
        (row["mode"], row["comparison"], row["layer"], row["response_group"]): row
        for row in summary if normalize_category(row["category"]) == category
    }
    rows = []
    for expected in reference["values"]:
        key = tuple(expected[name] for name in ("mode", "comparison", "layer", "response_group"))
        actual = lookup.get(key)
        observed = actual["mean"] if actual else None
        within = None
        if actual and compatible:
            within = math.isclose(observed, expected["mean"], abs_tol=reference["atol"], rel_tol=reference["rtol"])
        rows.append({
            "mode": expected["mode"], "category": reference["category"],
            "comparison": expected["comparison"], "layer": expected["layer"],
            "response_group": expected["response_group"], "expected_mean": expected["mean"],
            "observed_mean": observed, "n": actual["n"] if actual else 0,
            "abs_diff": abs(observed - expected["mean"]) if actual else None,
            "within_tolerance": within,
        })
    return {
        "status": reference["comparison_status"], "source": reference["source"],
        "mode_a_grouping": reference["mode_a_grouping"], "limitations": reference["limitations"],
        "category": reference["category"], "model_and_layers_compatible": compatible,
        "sample_counts": counts, "expected_sample_counts": reference["sample_counts"],
        "sample_count_matches": {name: counts[name] == count for name, count in reference["sample_counts"].items()},
        "atol": reference["atol"], "rtol": reference["rtol"],
        "all_numeric_values_close": bool(rows) and all(row["within_tolerance"] is True for row in rows),
        "rows": rows,
    }


def analyze_runs(run_dirs, category=None, reference=None, atol=1e-5, rtol=1e-5):
    groups, sources, selected = {}, [], []
    seen = set()
    signature = None
    for run_dir in run_dirs:
        metadata, records, checks, _ = inspect_run(run_dir)
        require(all(checks.values()), f"Invalid or incomplete extraction folder: {run_dir}; run validate first")
        current = extraction_signature(metadata)
        if signature is None:
            signature = current
        require(current == signature, "Extraction settings differ; analyze these runs separately")
        sources.append({
            "run_dir": str(run_dir), "cases": len(records), "versions": metadata["versions"],
            "device": metadata["device"], "attention_implementation": metadata["attention_implementation"],
            "model_config": metadata["model_config"],
        })
        for record in records:
            case = record["case"]
            if category and normalize_category(case["class"]) != normalize_category(category):
                continue
            key = (case["mode"], case["case_id"])
            require(key not in seen, f"Duplicate case across extraction folders: {key}")
            seen.add(key)
            require(case["mode"] in TARGET_COMPOSITIONS, f"Unknown experiment mode: {case['mode']}")
            selected.append(record)
            group_key = (case["mode"], case["class"], case["pair_id"])
            groups.setdefault(group_key, []).append((run_dir, metadata, record))
    require(bool(selected), "No extracted cases matched the selection")

    distances, skipped = [], []
    complete_groups = 0
    for (mode, name, pair_id), entries in groups.items():
        features, records, locations = {}, {}, {}
        for run_dir, metadata, record in entries:
            composition = "".join(record["case"]["image_labels"])
            require(composition in COMPOSITIONS[mode], f"Unexpected composition: {mode}/{composition}")
            require(composition not in records, f"Duplicate composition within {mode}/{pair_id}: {composition}")
            payload, _ = load_validated_feature(run_dir, record, metadata, atol, rtol)
            features[composition] = payload["image_features"].clone()
            del payload
            records[composition] = record
            locations[composition] = str(run_dir)
        required = TARGET_COMPOSITIONS[mode]
        missing = [composition for composition in required if composition not in records]
        if missing:
            skipped.append({
                "mode": mode, "category": name, "pair_id": pair_id, "reason": "missing_compositions",
                "present": list(records), "missing": missing,
            })
            continue
        target_id = find_target(mode, records)
        complete_groups += 1
        anchor = records["PPN"]["prediction"] if mode == "A" else None
        for left_name, right_name in combinations(required, 2):
            left, right = records[left_name], records[right_name]
            left_index = left["case"]["image_ids"].index(target_id)
            right_index = right["case"]["image_ids"].index(target_id)
            if mode == "A":
                response_group = {"Yes": "False-Yes", "No": "Correct-No", None: "Invalid"}[anchor]
            else:
                response_group = f"{left['prediction'] or 'Invalid'}->{right['prediction'] or 'Invalid'}"
            for index, layer in enumerate(signature["layers"]):
                distances.append({
                    "mode": mode, "category": name, "pair_id": pair_id,
                    "comparison": f"{left_name}-{right_name}", "layer": layer, "image_id": target_id,
                    "left_case_id": left["case"]["case_id"], "right_case_id": right["case"]["case_id"],
                    "left_run": locations[left_name], "right_run": locations[right_name],
                    "left_position": left_index + 1, "right_position": right_index + 1,
                    "left_prediction": left["prediction"], "right_prediction": right["prediction"],
                    "ppn_prediction": anchor, "response_group": response_group,
                    "cosine_distance": cosine_distance(features[left_name][index, left_index],
                                                       features[right_name][index, right_index]),
                })
    summary = summarize_distances(distances)
    return {
        "status": "completed" if distances else "no_complete_groups",
        "sources": sources, "extraction_settings": signature, "category": category,
        "selected_cases": len(selected), "groups_total": len(groups),
        "groups_analyzed": complete_groups, "groups_skipped": len(skipped),
        "validation_tolerance": {"atol": atol, "rtol": rtol},
        "mode_a_response_group": "PPN prediction: Yes = False-Yes, No = Correct-No; historical definition estimated.",
        "mode_b_response_group": "Prediction of the left composition -> prediction of the right composition.",
        "distance_definition": "1 - cosine similarity of the same N1 image mean vectors, computed in float64 on CPU.",
        "summary_definition": "One observation per complete comparison group and layer; sample SD is null when n=1.",
        "distances": distances, "summary": summary, "skipped_groups": skipped,
        "reference_comparison": compare_reference(summary, selected, signature, reference) if reference else None,
    }
