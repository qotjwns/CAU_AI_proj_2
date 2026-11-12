import argparse
from pathlib import Path

import torch

from src.common.config import read_yaml
from src.common.data import write_csv, write_json
from src.common.paths import PROJECT_ROOT, project_path, project_relative_path
from src.common.threads import configure_cpu_threads
from src.core.probe_analysis import run_probe_study
from src.core.probe_outputs import write_evaluation


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step3.yaml"
DEFAULT_REFERENCE = PROJECT_ROOT / "configs" / "experiments" / "step3_reference.yaml"


def main():
    configure_cpu_threads()
    parser = argparse.ArgumentParser(description="STEP 2 Hidden State로 20개 클래스 Linear Probe 학습·OOF 평가 (CPU)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, help="STEP 2 추출 폴더 (기본: 설정 파일의 run_dir)")
    parser.add_argument("--output-dir", type=Path, help="기본: 설정 파일의 output_dir")
    parser.add_argument("--reference-config", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--permutation-runs", type=int, help="라벨 순열 기준선 반복 수 (기본: 설정 파일, 0이면 생략)")
    args = parser.parse_args()
    if args.permutation_runs is not None and args.permutation_runs < 0:
        parser.error("--permutation-runs는 0 이상이어야 합니다.")

    config = read_yaml(project_path(args.config))
    run_dir = project_path(args.run_dir or config["run_dir"]).resolve()
    output_dir = project_path(args.output_dir or config["output_dir"])
    reference = read_yaml(project_path(args.reference_config)) if args.reference_config else None

    result = run_probe_study(run_dir, config, reference, args.permutation_runs)
    report, layers, oof = result["report"], result["layers"], result["oof"]
    report["config"] = project_relative_path(args.config)
    report["reference_config"] = project_relative_path(args.reference_config) if args.reference_config else None
    report["run_dir"] = project_relative_path(run_dir)
    report["output_dir"] = project_relative_path(output_dir)

    write_evaluation(report, result["vectors"], result["slots"], oof, layers, result["strict_cases"],
                     result["examples"], output_dir)
    write_json(result["split"], output_dir / "split.json", allow_nan=False)
    write_csv(result["probes"], output_dir / "probes.csv", [
        "fold", "class", "layer", "n_train_vectors", "n_train_P", "n_train_N", "n_train_images",
        "n_excluded_prefix", "n_val_vectors", "n_val_images", "train_accuracy", "val_context_accuracy",
        "val_context_auroc", "val_image_accuracy", "val_image_auroc", "iterations", "grad_max", "converged", "loss",
    ])
    write_csv(result["null_runs"], output_dir / "baselines_permutation_runs.csv", ["run", "layer", "metric", "value"])
    checkpoint = output_dir / "checkpoints" / "probes.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"status": report["status"], "layers": layers, "settings": report["settings"],
                "image_fold": result["split"]["image_fold"], "probes": result["checkpoints"]}, checkpoint)

    counts, trace = report["counts"], report["denominators"]
    print(f"저장: {output_dir} | 상태 {report['status']} | Probe {counts['probes_trained']}개 "
          f"(미수렴 {counts['probes_not_converged']}) | 문맥 벡터 {counts['context_vectors']}개")
    print(f"분모: 레코드 {trace['records']} → pair 내 중복 제거 {trace['after_within_pair_dedup']} "
          f"→ pair 간 중복 제거 {trace['unique_inputs']} (제외 {sum(trace['excluded'].values())})")
    nulls = {(row["layer"], row["metric"]): row for row in report["baselines"] if row["baseline"] == "label_permutation"}
    for row in report["headline"]:
        null = nulls.get((row["layer"], "strict_discrepancy_rate"), {})
        null_text = f" (순열 평균 {null['value']:.3f}, p={null['p_value']:.3f})" if null.get("value") is not None else ""
        print(f"  L{row['layer']}: 이미지 정확도 {row['image_accuracy']:.3f} | "
              f"VLM False-Yes {row['vlm_false_yes']} 중 Strict {row['fy_strict']} "
              f"= {row['strict_discrepancy_rate']:.3f}{null_text} | "
              f"교정 {row['fy_corrected']} (N 탐지 {row['corrected_via_negative']}, P 오판 {row['corrected_via_positive_error']}) | "
              f"AND 신규 False-Yes {row['and_new_false_yes']} | PPP 유지 {row['ppp_retention_rate']:.3f}")
    print("평가 범위: 대상 image_id만 Probe 학습과 독립이며, 입력 전체의 독립 평가는 아닙니다 (exploratory).")


if __name__ == "__main__":
    main()
