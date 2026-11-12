import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import write_csv, write_json
from src.common.paths import PROJECT_ROOT, project_path
from src.core.feature_analysis import analyze_runs


DEFAULT_REFERENCE = PROJECT_ROOT / "configs" / "experiments" / "step2_reference.yaml"


def main():
    parser = argparse.ArgumentParser(description="저장된 Hidden State의 동일 N1 거리 분석 및 과거 수치 참고 비교 (CPU)")
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--category")
    parser.add_argument("--output-dir", type=Path, help="여러 추출 폴더를 합쳐 분석할 때 필수")
    parser.add_argument("--reference-config", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--atol", type=float, default=1e-5, help="저장 특징 검증의 절대 허용 오차")
    parser.add_argument("--rtol", type=float, default=1e-5, help="저장 특징 검증의 상대 허용 오차")
    args = parser.parse_args()
    if len(args.run_dir) > 1 and args.output_dir is None:
        parser.error("여러 폴더의 분석 결과를 저장할 --output-dir을 지정하세요.")
    if not 0 <= args.atol < float("inf") or not 0 <= args.rtol < float("inf"):
        parser.error("허용 오차는 유한한 0 이상의 값이어야 합니다.")
    run_dirs = [project_path(path).resolve() for path in args.run_dir]
    root = PROJECT_ROOT / "results"
    destination = run_dirs[0] if run_dirs[0].is_relative_to(root) else root / "step2" / run_dirs[0].name
    output_dir = project_path(args.output_dir) if args.output_dir else destination / "analysis"
    reference = read_yaml(project_path(args.reference_config))
    report = analyze_runs(run_dirs, args.category, reference, args.atol, args.rtol)
    report["reference_config"] = str(project_path(args.reference_config))
    write_json(report, output_dir / "analysis.json", allow_nan=False)
    write_csv(report["distances"], output_dir / "distances.csv", [
        "mode", "category", "pair_id", "comparison", "layer", "image_id", "left_case_id", "right_case_id",
        "left_run", "right_run", "left_position", "right_position", "left_prediction", "right_prediction",
        "ppn_prediction", "response_group", "cosine_distance",
    ])
    write_csv(report["summary"], output_dir / "summary.csv", [
        "mode", "category", "comparison", "layer", "response_group", "n", "mean", "median", "std", "min", "max",
    ])
    write_csv(report["skipped_groups"], output_dir / "skipped_groups.csv", [
        "mode", "category", "pair_id", "reason", "present", "missing",
    ])
    write_csv(report["reference_comparison"]["rows"], output_dir / "reference_comparison.csv", [
        "mode", "category", "comparison", "layer", "response_group", "expected_mean", "observed_mean",
        "n", "abs_diff", "within_tolerance",
    ])
    print(f"저장: {output_dir} | 분석 {report['groups_analyzed']}묶음, 제외 {report['groups_skipped']}묶음")
    print("과거 수치는 참고 비교이며, 수치가 가까워도 기존 실험 재현을 입증하지 않습니다.")
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
