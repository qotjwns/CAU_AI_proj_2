import argparse
from pathlib import Path

from src.common.data import write_csv, write_json
from src.common.paths import PROJECT_ROOT, project_path
from src.core.feature_validation import validate_run


def main():
    parser = argparse.ArgumentParser(description="저장된 STEP 2 특징 검증 및 STEP 1 추론 결과 비교 (CPU)")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step1-results", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if not 0 <= args.atol < float("inf") or not 0 <= args.rtol < float("inf"):
        parser.error("허용 오차는 유한한 0 이상의 값이어야 합니다.")
    run_dir = project_path(args.run_dir).resolve()
    root = PROJECT_ROOT / "results"
    destination = run_dir if run_dir.is_relative_to(root) else root / "step2" / run_dir.name
    output_dir = project_path(args.output_dir) if args.output_dir else destination / "validation"
    step1_paths = [project_path(path) for path in args.step1_results] if args.step1_results else None
    report = validate_run(run_dir, step1_paths, args.atol, args.rtol)
    write_json(report, output_dir / "validation.json", allow_nan=False)
    write_csv(report["cases"], output_dir / "cases.csv", [
        "mode", "case_id", "feature_file", "passed", "error", "image_shape", "query_shape",
        "raw_tokens_checked", "pooling_max_abs_diff",
    ])
    columns = ["mode", "case_id", "status", "prediction_step1", "prediction_step2", "prediction_match",
               "raw_response_match", "input_token_count_match", "image_token_count_match"]
    columns.extend(f"{name}_{suffix}" for name in ("yes_logit", "no_logit", "logit_margin")
                   for suffix in ("step1", "step2", "abs_diff", "match"))
    write_csv(report["step1_comparisons"], output_dir / "step1_comparison.csv", columns)
    print(f"저장: {output_dir} | features={report['feature_validation_passed']} "
          f"| STEP 1={report['step1_comparison_passed']}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
