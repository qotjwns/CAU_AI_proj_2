import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import torch

from src.common.data import read_json, read_jsonl
from src.core.features import compare_tensors, map_image_tokens
from src.core.results import load_results


def require(condition, message):
    if not condition:
        raise ValueError(message)


def feature_path(run_dir, record):
    path = (run_dir / record["feature_file"]).resolve()
    require(path.is_relative_to(run_dir.resolve()), "Feature file is outside the extraction folder")
    return path


def inspect_run(run_dir):
    metadata = read_json(run_dir / "run.json")
    records = list(read_jsonl(run_dir / "manifest.jsonl"))
    paths = [feature_path(run_dir, record) for record in records]
    actual = set((run_dir / "raw").rglob("*.pt"))
    actual = {path.resolve() for path in actual}
    expected = set(paths)
    keys = [(record["case"]["mode"], record["case"]["case_id"]) for record in records]
    checks = {
        "completed": metadata["status"] == "completed",
        "extraction_verification": (
            metadata.get("verification_passed") is True
            if metadata["verify_repeat"] or metadata["verify_reference"]
            else metadata.get("verification_passed") is None
        ),
        "non_empty": bool(records),
        "case_counts": metadata["selected_cases"] == metadata["written_cases"] == len(records),
        "unique_cases": len(set(keys)) == len(keys),
        "unique_feature_files": len(expected) == len(paths),
        "record_indices": [record["index"] for record in records] == list(range(len(records))),
        "all_feature_files_present": expected <= actual,
        "no_unreferenced_features": actual <= expected,
    }
    inventory = {
        "missing": [str(path.relative_to(run_dir.resolve())) for path in sorted(expected - actual)],
        "unreferenced": [str(path.relative_to(run_dir.resolve())) for path in sorted(actual - expected)],
    }
    return metadata, records, checks, inventory


def load_validated_feature(run_dir, record, metadata, atol=1e-5, rtol=1e-5):
    payload = torch.load(feature_path(run_dir, record), map_location="cpu", weights_only=True)
    case = record["case"]
    config = metadata["config"]
    layers = config["layers"]
    width = config["expected_hidden_size"]
    decoder_config = metadata["model_config"].get("text_config") or metadata["model_config"]
    require(bool(layers) and width > 0, "Empty layer selection or invalid hidden size")
    require(width == decoder_config["hidden_size"], "Hidden size differs from saved model configuration")
    require(metadata["final_layer_after_norm"] == decoder_config["num_hidden_layers"],
            "Final layer differs from saved model configuration")
    require(config["pooling_dtype"] in ("float32", "model"), "Unsupported pooling dtype")
    require(payload["case"] == case, "Case metadata differs between manifest and .pt")
    require(len(case["image_ids"]) == len(case["image_order"]) == len(case["image_labels"]) == 3,
            "Expected three images, filenames and labels")
    require(len(set(case["image_ids"])) == 3, "Repeated image IDs within one case")
    require(all(label in ("P", "N") for label in case["image_labels"]), "Invalid image label")
    require(case["composition"].replace("+", "") == "".join(case["image_labels"]), "Composition mismatch")
    require(case["gt"] == ("Yes" if all(label == "P" for label in case["image_labels"]) else "No"),
            "Ground truth differs from image labels")
    require(payload["layers"] == layers and len(set(layers)) == len(layers), "Layer selection mismatch")
    require(payload["pooling"] == metadata["pooling"] == f"mean_{config['pooling_dtype']}",
            "Pooling convention mismatch")
    require(payload["layer_convention"] == metadata["layer_convention"] == "huggingface_hidden_states",
            "Layer convention mismatch")
    require(payload["final_layer_after_norm"] == metadata["final_layer_after_norm"],
            "Final normalization layer mismatch")
    require(all(1 <= layer <= metadata["final_layer_after_norm"] for layer in layers), "Invalid layer number")

    for name, shape in (("image_features", (len(layers), 3, width)),
                        ("query_features", (len(layers), width))):
        tensor = payload[name]
        require(tuple(tensor.shape) == shape, f"{name}: expected shape {shape}, got {tuple(tensor.shape)}")
        require(tensor.dtype == torch.float32, f"{name}: expected float32")
        require(bool(torch.isfinite(tensor).all()), f"{name}: NaN or Inf")
        require(bool((torch.linalg.vector_norm(tensor.double(), dim=-1) > 0).all()), f"{name}: zero vector")

    ids, mask, grids = (payload[key] for key in ("input_ids", "attention_mask", "image_grid_thw"))
    require(ids.ndim == 2 and ids.shape[0] == 1 and tuple(mask.shape) == tuple(ids.shape),
            "Invalid input IDs or attention mask shape")
    require(ids.dtype == torch.int64 and grids.dtype == torch.int64, "Input IDs and image grid must be int64")
    require(bool(((mask == 0) | (mask == 1)).all()), "Attention mask must contain only 0 and 1")
    require(tuple(grids.shape) == (3, 3), "Expected image_grid_thw[3, 3]")
    spans, query_index = map_image_tokens(
        payload, SimpleNamespace(**metadata["model_config"]), metadata["image_processor"]["merge_size"],
        case["image_ids"],
    )
    require(spans == payload["image_spans"] == record["image_spans"], "Recomputed image spans differ")
    require(query_index == payload["query_index"] == record["query_index"], "Final input token differs")
    require(sum(span["token_count"] for span in spans) == record["image_token_count"], "Image token count mismatch")
    require(int(mask.sum()) == record["input_token_count"], "Input token count mismatch")
    require(all(key in payload["inference"] for key in (
        "prediction", "raw_response", "yes_logit", "no_logit", "logit_margin", "input_token_count", "image_token_count",
    )), "Missing inference fields in .pt")
    for key, value in payload["inference"].items():
        require(record[key] == value, f"Inference metadata differs: {key}")
    require(payload["inference"]["prediction"] in ("Yes", "No", None), "Invalid prediction label")
    require(record["correct"] == (record["prediction"] == case["gt"]), "Correctness flag mismatch")
    for key in ("yes_logit", "no_logit", "logit_margin"):
        require(math.isfinite(record[key]), f"{key}: NaN or Inf")
    require(math.isclose(record["yes_logit"] - record["no_logit"], record["logit_margin"],
                         abs_tol=atol, rel_tol=rtol), "Logit margin differs from yes_logit - no_logit")
    verification = record["verification"]
    expected_checks = []
    if metadata["verify_repeat"]:
        expected_checks.extend(("repeat_features", "repeat_inference"))
    if metadata["verify_reference"]:
        expected_checks.append("reference")
    require(all(verification[name]["passed"] for name in expected_checks), "Extraction-time verification failed")

    has_tokens = "visual_token_states" in payload
    require(has_tokens == metadata["save_token_states"], "Raw token storage flag mismatch")
    pooling_diff = None
    if has_tokens:
        states = payload["visual_token_states"]
        require(set(states) == set(layers), "Raw token layers mismatch")
        pooled = []
        dtype = {"torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16,
                 "torch.float32": torch.float32}[metadata["dtype"]]
        for layer in layers:
            require(len(states[layer]) == 3, f"Layer {layer}: expected three raw token tensors")
            images = []
            for span, tensor in zip(spans, states[layer]):
                require(tuple(tensor.shape) == (span["token_count"], width), "Raw token tensor shape mismatch")
                require(tensor.dtype == dtype and bool(torch.isfinite(tensor).all()), "Invalid raw token values or dtype")
                values = tensor.float() if config["pooling_dtype"] == "float32" else tensor
                images.append(values.mean(dim=0).float())
            pooled.append(torch.stack(images))
        comparison = compare_tensors(torch.stack(pooled), payload["image_features"], atol, rtol)
        require(comparison["allclose"], f"Recomputed mean pooling differs: {comparison['max_abs_diff']}")
        pooling_diff = comparison["max_abs_diff"]
    details = {
        "image_shape": list(payload["image_features"].shape),
        "query_shape": list(payload["query_features"].shape),
        "raw_tokens_checked": has_tokens, "pooling_max_abs_diff": pooling_diff,
    }
    return payload, details


def compare_step1(record, metadata, baseline, atol=1e-5, rtol=1e-5):
    case = record["case"]
    row = {"mode": case["mode"], "case_id": case["case_id"], "status": "missing"}
    if baseline is None:
        return row
    if baseline["model_name"] != metadata["config"]["model_name"] or baseline["inference_method"] != "baseline":
        return {**row, "status": "incompatible_model"}
    same_input = (
        baseline["prompt"] == case["prompt"] and baseline["object"] == case["class"]
        and baseline["gt"] == case["gt"] and baseline["pair_id"] == case["pair_id"]
        and [Path(path).name for path in baseline["images"]] == [Path(path).name for path in case["image_order"]]
        and baseline["per_image_gt"] == [int(label == "P") for label in case["image_labels"]]
    )
    if not same_input:
        return {**row, "status": "incompatible_input"}
    row.update(
        prediction_step1=baseline["prediction"], prediction_step2=record["prediction"],
        prediction_match=baseline["prediction"] == record["prediction"],
        raw_response_match=(baseline["raw_response"] == record["raw_response"] if "raw_response" in baseline else None),
        input_token_count_match=baseline["input_token_count"] == record["input_token_count"],
        image_token_count_match=baseline["image_token_count"] == record["image_token_count"],
    )
    for name in ("yes_logit", "no_logit", "logit_margin"):
        left, right = baseline[name], record[name]
        finite = math.isfinite(left) and math.isfinite(right)
        row[f"{name}_step1"], row[f"{name}_step2"] = (left if math.isfinite(left) else None), right
        row[f"{name}_abs_diff"] = abs(left - right) if finite else None
        row[f"{name}_match"] = finite and math.isclose(left, right, abs_tol=atol, rel_tol=rtol)
    matches = [value for key, value in row.items() if key.endswith("_match") and value is not None]
    row["status"] = "match" if all(matches) else "different"
    return row


def validate_run(run_dir, step1_paths=None, atol=1e-5, rtol=1e-5):
    metadata, records, checks, inventory = inspect_run(run_dir)
    baseline = None
    if step1_paths:
        baseline = {(row["mode"], row["case_id"]): row for row in load_results(step1_paths)}
    cases, comparisons = [], []
    for record in records:
        case = record["case"]
        row = {"mode": case["mode"], "case_id": case["case_id"], "feature_file": record["feature_file"]}
        try:
            payload, details = load_validated_feature(run_dir, record, metadata, atol, rtol)
            del payload
            row.update(passed=True, error=None, **details)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError,
                AttributeError, EOFError, pickle.UnpicklingError) as error:
            row.update(passed=False, error=f"{type(error).__name__}: {error}")
        cases.append(row)
        if baseline is not None:
            if row["passed"]:
                comparisons.append(compare_step1(record, metadata, baseline.get((case["mode"], case["case_id"])), atol, rtol))
            else:
                comparisons.append({"mode": case["mode"], "case_id": case["case_id"], "status": "invalid_feature"})
    feature_passed = all(checks.values()) and all(row["passed"] for row in cases)
    baseline_passed = None
    if baseline is not None:
        baseline_passed = bool(comparisons) and all(row["status"] == "match" for row in comparisons)
    return {
        "run_dir": str(run_dir), "atol": atol, "rtol": rtol, "run_checks": checks, "file_inventory": inventory,
        "cases_checked": len(cases), "cases_failed": sum(not row["passed"] for row in cases),
        "feature_validation_passed": feature_passed, "step1_comparison_passed": baseline_passed,
        "step1_sources": [str(path) for path in step1_paths or []],
        "step1_comparison_scope": "Same model name and case inputs; STEP 1 runtime and weight provenance are not recorded.",
        "passed": feature_passed and baseline_passed is not False,
        "cases": cases, "step1_comparisons": comparisons,
    }
