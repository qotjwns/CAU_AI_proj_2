"""Compute A/B metrics from normalized prediction records."""

from collections import Counter, defaultdict
from itertools import combinations, pairwise
from statistics import fmean


LABELS = ("Yes", "No")
COMPOSITIONS = {"A": ("PPP", "PPN", "PNN", "NNN"), "B": ("NPP", "PNP", "PPN")}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(records):
    total = len(records)
    correct = sum(row["correct"] for row in records)
    counts = Counter((row["gt"], row["prediction"] or "Invalid") for row in records)
    confusion = {
        gt: {prediction: counts[gt, prediction] for prediction in (*LABELS, "Invalid")}
        for gt in LABELS
    }
    gt_yes = sum(confusion["Yes"].values())
    gt_no = sum(confusion["No"].values())
    invalid = sum(confusion[gt]["Invalid"] for gt in LABELS)
    yes_recall = ratio(confusion["Yes"]["Yes"], gt_yes)
    no_recall = ratio(confusion["No"]["No"], gt_no)
    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": ratio(correct, total),
        "error_rate": ratio(total - correct, total),
        "gt_yes_count": gt_yes,
        "gt_no_count": gt_no,
        "invalid_count": invalid,
        "invalid_rate": ratio(invalid, total),
        "valid_response_rate": ratio(total - invalid, total),
        "yes_recall": yes_recall,
        "no_recall": no_recall,
        "balanced_accuracy": (yes_recall + no_recall) / 2 if gt_yes and gt_no else None,
        "false_positive_rate": ratio(confusion["No"]["Yes"], gt_no),
        "mean_logit_margin": fmean(row["logit_margin"] for row in records),
        "mean_input_token_count": fmean(row["input_token_count"] for row in records),
        "mean_image_token_count": fmean(row["image_token_count"] for row in records),
        "confusion_matrix": confusion,
    }


def group_records(records, keys):
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in keys)].append(record)
    return sorted(groups.items())


def summarize_by(records, keys):
    return [
        {**dict(zip(keys, values)), **summarize(rows)}
        for values, rows in group_records(records, keys)
    ]


def compute_pairs(records):
    summaries = []
    details = []
    for (mode,), mode_records in group_records(records, ("mode",)):
        order = COMPOSITIONS[mode]
        steps = list(pairwise(order) if mode == "A" else combinations(order, 2))
        groups = group_records(mode_records, ("group_id", "pair_id", "object"))
        complete = valid_complete = all_correct = consistent = 0
        for (group_id, pair_id, category), rows in groups:
            variants = {row["composition"]: row for row in rows}
            if len(variants) != len(rows):
                raise ValueError(f"Duplicate composition in pair: {mode}, {pair_id}")
            # Compare the same complete cohort across all transitions.
            if set(variants) != set(order):
                continue
            complete += 1
            all_correct += all(row["correct"] for row in rows)
            if all(row["prediction"] in LABELS for row in rows):
                valid_complete += 1
                consistent += len({row["prediction"] for row in rows}) == 1

            for source, target in steps:
                before, after = variants[source], variants[target]
                valid = before["prediction"] in LABELS and after["prediction"] in LABELS
                details.append({
                    "mode": mode,
                    "object": category,
                    "group_id": group_id,
                    "pair_id": pair_id,
                    "from_composition": source,
                    "to_composition": target,
                    "from_prediction": before["prediction"],
                    "to_prediction": after["prediction"],
                    "valid_predictions": valid,
                    "prediction_changed": before["prediction"] != after["prediction"] if valid else None,
                    "both_correct": before["correct"] and after["correct"],
                    "from_logit_margin": before["logit_margin"],
                    "to_logit_margin": after["logit_margin"],
                    "margin_delta": after["logit_margin"] - before["logit_margin"],
                })

        summaries.append({
            "mode": mode,
            "observed_pairs": len(groups),
            "complete_pairs": complete,
            "incomplete_pairs": len(groups) - complete,
            "valid_complete_pairs": valid_complete,
            "all_correct_pairs": all_correct,
            "all_correct_rate": ratio(all_correct, complete),
            # Only B keeps the same ground truth under every permutation.
            "order_agreement_rate": ratio(consistent, valid_complete) if mode == "B" else None,
        })

    transitions = []
    keys = ("mode", "from_composition", "to_composition")
    for values, rows in group_records(details, keys):
        valid_count = sum(row["valid_predictions"] for row in rows)
        changed_count = sum(row["prediction_changed"] is True for row in rows)
        transitions.append({
            **dict(zip(keys, values)),
            "pair_count": len(rows),
            "valid_prediction_pairs": valid_count,
            "invalid_prediction_pairs": len(rows) - valid_count,
            "prediction_changed_count": changed_count,
            "prediction_change_rate": ratio(changed_count, valid_count),
            "both_correct_rate": ratio(sum(row["both_correct"] for row in rows), len(rows)),
            "mean_margin_delta": fmean(row["margin_delta"] for row in rows),
        })
    return summaries, transitions, details


def compute_metrics(records):
    positions = [
        {**row, "negative_position": row["negative_indices"][0] + 1}
        for row in records if row["mode"] == "B"
    ]
    pair_summary, pair_transitions, pair_details = compute_pairs(records)
    return {
        "inference_method": records[0]["inference_method"],
        "model_name": records[0]["model_name"],
        "overall": summarize(records),
        "by_mode": summarize_by(records, ("mode",)),
        "by_class": summarize_by(records, ("mode", "object")),
        "by_composition": summarize_by(records, ("mode", "composition")),
        "by_negative_position": summarize_by(positions, ("mode", "negative_position")),
        "pair_summary": pair_summary,
        "pair_transitions": pair_transitions,
        "pair_details": pair_details,
        "logit_margins": [
            {
                "mode": mode,
                "correct": [row["logit_margin"] for row in rows if row["correct"]],
                "incorrect": [row["logit_margin"] for row in rows if not row["correct"]],
            }
            for (mode,), rows in group_records(records, ("mode",))
        ],
    }
