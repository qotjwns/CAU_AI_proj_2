"""STEP 3·4 공통 결과 파일 저장."""
from src.common.data import write_csv, write_json


CASE_COLUMNS = [
    "layer", "class", "composition", "mode", "case_id", "pair_id", "n_records", "image_ids", "image_files",
    "labels", "p_image1", "p_image2", "p_image3", "probe_labels", "n_images_correct", "min_correct_probability",
    "vlm_prediction", "vlm_logit_margin", "probe_and",
]
BASELINE_COLUMNS = [
    "baseline", "layer", "metric", "value", "sd", "null_q025", "null_q975", "n_runs", "observed", "p_value",
    "observed_by_layer",
]


def probability_columns(layers):
    return [f"p_layer{layer}" for layer in layers]


def with_probabilities(row, predictions, index, layers):
    return {**row, **{column: float(predictions[index, i]) for i, column in enumerate(probability_columns(layers))}}


def write_evaluation(report, vectors, slots, predictions, layers, strict_cases, examples, output_dir):
    write_json(report, output_dir / "analysis.json", allow_nan=False)
    write_csv(
        [with_probabilities({**vector, "prefix": list(vector["prefix"])}, predictions, vector["vector"], layers)
         for vector in vectors],
        output_dir / "vectors.csv",
        ["vector", "class", "image_id", "label", "position", "prefix", "fold", "context", "n_slots",
         "representative_record", "max_duplicate_cosine_distance", *probability_columns(layers)],
    )
    write_csv(
        [with_probabilities({**slot, "prefix": list(slot["prefix"])}, predictions, slot["vector"], layers)
         for slot in slots],
        output_dir / "slots.csv",
        ["record", "mode", "case_id", "class", "pair_id", "composition", "position", "image_id", "label",
         "prefix", "vector", "fold", "context", "prediction", *probability_columns(layers)],
    )
    write_csv(report["image_level"], output_dir / "image_level.csv", [
        "layer", "cls", "measure", "n", "n_P", "n_N", "n_targets", "n_slots", "accuracy", "auroc",
    ])
    write_csv(report["context_level"], output_dir / "context_level.csv", [
        "layer", "cls", "position", "context", "n", "n_P", "n_N", "n_targets", "n_slots", "accuracy", "auroc",
    ])
    write_csv(report["paired_context"], output_dir / "paired_context.csv", [
        "layer", "position", "n_targets", "n_pure_vectors", "n_mixed_vectors", "mean_pure_accuracy",
        "mean_mixed_accuracy", "mean_accuracy_difference", "mean_pure_correct_probability",
        "mean_mixed_correct_probability", "mixed_better", "equal", "pure_better",
    ])
    write_csv(report["false_yes"], output_dir / "false_yes.csv", [
        "level", "layer", "vlm_group", "n_cases", "n_negative_slots", "n_negative_images",
        "negative_detection_rate", "all_negatives_detected_rate", "mean_negative_probability",
        "pure_negative_slots", "pure_negative_detection_rate", "mixed_negative_slots", "mixed_negative_detection_rate",
    ])
    write_csv(report["probe_and"], output_dir / "probe_and.csv", [
        "level", "layer", "scope", "value", "n", "vlm_accuracy", "probe_and_accuracy", "both_correct",
        "vlm_only_correct", "probe_only_correct", "both_wrong", "vlm_invalid",
    ])
    write_csv(report["key_table"], output_dir / "key_table.csv", list(report["key_table"][0]))
    write_csv(report["key_table_by_class"], output_dir / "key_table_by_class.csv", list(report["key_table_by_class"][0]))
    write_csv(report["strict_discrepancy"], output_dir / "strict_discrepancy.csv", list(report["strict_discrepancy"][0]))
    write_csv(report["group_comparison"], output_dir / "group_comparison.csv", [
        "level", "layer", "analysis", "metric", "group", "comparison_group", "n_cases", "n_comparison_cases",
        "n_units", "value", "comparison_value", "difference", "ci_low", "ci_high", "ci_target", "ci_valid",
        "greater", "equal", "less",
    ])
    write_csv(strict_cases, output_dir / "strict_discrepancy_cases.csv", CASE_COLUMNS)
    write_csv(examples, output_dir / "examples.csv", ["category", *CASE_COLUMNS])
    write_csv(report["baselines"], output_dir / "baselines.csv", BASELINE_COLUMNS)
