import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import write_csv, write_json
from src.common.paths import PROJECT_ROOT, project_path, project_relative_path
from src.core.hq_probe import run_hq_probe
from src.visualization.hq_probe import plot_hq_readability, plot_localization


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step5_hq_probe.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 5 사전 진단: h_q Probe와 z_i Probe의 이미지별 P/N 판별 비교 (CPU)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    run_dir = project_path(config["run_dir"]).resolve()
    step3_dir = project_path(config["step3_dir"]).resolve()
    output_dir = project_path(args.output_dir or config["output_dir"])

    report, cases, predictions, probe_rows = run_hq_probe(run_dir, step3_dir, config)
    report.update(config=project_relative_path(args.config), run_dir=project_relative_path(run_dir),
                  step3_dir=project_relative_path(step3_dir), output_dir=project_relative_path(output_dir))
    layers = report["extraction_settings"]["layers"]
    write_json(report, output_dir / "analysis.json", allow_nan=False)
    write_csv(report["readability"], output_dir / "readability.csv", list(report["readability"][0]))
    write_csv(report["group_comparison"], output_dir / "group_comparison.csv", [
        "feature", "level", "layer", "analysis", "metric", "group", "comparison_group", "n_cases",
        "n_comparison_cases", "n_units", "value", "comparison_value", "difference", "ci_low", "ci_high",
        "ci_target", "ci_valid", "greater", "equal", "less",
    ])
    write_csv(report["localization"], output_dir / "localization.csv", list(report["localization"][0]))
    write_csv(probe_rows, output_dir / "probes.csv", [
        "scope", "group", "fold", "position", "layer", "n_train", "n_train_P", "n_val", "status", "converged", "iterations",
    ])
    rows = []
    for case in cases:
        for position in range(3):
            slot = case["vectors"][position]
            row = {key: case[key] for key in ("mode", "case_id", "class", "composition", "gt", "prediction", "n_records")}
            row.update(position=position + 1, image_id=case["image_ids"][position], label=case["labels"][position])
            for feature, values in predictions.items():
                row.update({f"p_{feature}_layer{layer}": float(values[slot, i]) for i, layer in enumerate(layers)})
            rows.append(row)
    write_csv(rows, output_dir / "predictions.csv", list(rows[0]))

    report_dir = (PROJECT_ROOT / "reports" / output_dir.relative_to(PROJECT_ROOT / "results")
                  if output_dir.is_relative_to(PROJECT_ROOT / "results") else output_dir)
    for figure in (plot_hq_readability(report, report_dir), plot_localization(report, report_dir, inputs="all"),
                   plot_localization(report, report_dir, inputs="pure")):
        print(f"그림: {figure}")
    counts = report["counts"]
    print(f"저장: {output_dir} | 고유 입력 {counts['unique_inputs']} | Probe {counts['probes_fitted']}개 "
          f"(단일 라벨 {counts['probes_single_label']}, 미수렴 {counts['probes_not_converged']})")
    for row in report["readability"]:
        if row["context"] == "all":
            print(f"  L{row['layer']} pos{row['position']}: AUROC z_i {row['z_i_auroc']:.3f} | "
                  + " | ".join(f"{key[:-6]} {row[key]:.3f}" for key in row if key.startswith("h_q") and key.endswith("_auroc")))
    for row in report["localization"]:
        if row["inputs"] == "all" and row["scope"] == "all" and row["feature"] == "h_q_pooled":
            print(f"  N 위치 정확도 h_q_pooled L{row['layer']}: {row['accuracy']:.3f} (우연 0.333, n={row['n']})")
    print("진단용: h_q 판별력의 높고 낮음은 정보 부재·통합 실패·실제 사용 여부를 확정하지 않습니다.")
    print("PNN만 N 2장 구성이므로 위치 판별은 N 1장 구성의 N 위치 정확도로 확인합니다.")


if __name__ == "__main__":
    main()
