import csv
from pathlib import Path

from src.common.data import read_jsonl, write_json
from src.common.paths import project_relative_path
from src.core.prompts import LABELS


def build_result(case, inference, config):
    image_dir = Path(case["image_dir"])
    return {
        "case_id": case["case_id"],
        "group_id": case["group_id"],
        "pair_id": case["pair_id"],
        "mode": case["mode"],
        "inference_method": "baseline",
        "model_name": config["model_name"],
        "object": case["class"],
        "composition": "".join(case["image_labels"]),
        "images": [project_relative_path(image_dir / name) for name in case["image_order"]],
        "per_image_gt": [int(label == "P") for label in case["image_labels"]],
        "negative_indices": case["negative_positions"],
        "prompt": case["prompt"],
        "gt": case["gt"],
        **inference,
        "correct": inference["prediction"] == case["gt"],
    }


def load_results(paths):
    records = []
    seen = set()
    for path in paths:
        for record in read_jsonl(path):
            key = (record["mode"], record["case_id"])
            if key in seen:
                raise ValueError(f"Duplicate result: {key}")
            seen.add(key)
            if record["prediction"] not in LABELS:
                record["prediction"] = None
            record["correct"] = record["prediction"] in LABELS and record["prediction"] == record["gt"]
            record["composition"] = record["composition"].replace("+", "")
            records.append(record)

    runs = {(row["inference_method"], row["model_name"]) for row in records}
    if len(runs) != 1:
        raise ValueError("Provide non-empty results from one inference method and one model.")
    return records


def save_metrics(report, output_dir):
    write_json(report, output_dir / "metrics.json", allow_nan=False)

    tables = {"overall": [report["overall"]]}
    for name in ("by_mode", "by_class", "by_composition", "by_negative_position",
                 "pair_summary", "pair_transitions", "pair_details"):
        tables[name] = report[name]
    for name, rows in tables.items():
        output_path = output_dir / f"{name}.csv"
        if not rows:
            output_path.unlink(missing_ok=True)
            continue
        columns = [key for key in rows[0] if key != "confusion_matrix"]
        with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
