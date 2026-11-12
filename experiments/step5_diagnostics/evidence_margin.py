import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import write_csv, write_json
from src.common.paths import PROJECT_ROOT, project_path, project_relative_path
from src.common.threads import configure_cpu_threads
from src.core.evidence_margin import run_evidence_margin
from src.visualization.evidence_margin import plot_evidence_margin


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step5_evidence_margin.yaml"


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def main():
    configure_cpu_threads()
    parser = argparse.ArgumentParser(
        description="STEP 5-1 탐색: N 1장 입력에서 z_i Positive 증거가 Negative 증거 외에 VLM Margin을 추가로 설명하는가 (CPU)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    step4_dir = project_path(config["step4_dir"]).resolve()
    output_dir = project_path(args.output_dir or config["output_dir"])

    report, tables = run_evidence_margin(step4_dir, config)
    report.update(config=project_relative_path(args.config), step4_dir=project_relative_path(step4_dir),
                  output_dir=project_relative_path(output_dir))
    write_json(report, output_dir / "analysis.json", allow_nan=False)
    write_csv(report["coefficients"], output_dir / "coefficients.csv", list(report["coefficients"][0]))
    write_csv(report["fits"], output_dir / "fits.csv", list(report["fits"][0]))
    write_csv(report["diagnostics"], output_dir / "diagnostics.csv", list(report["diagnostics"][0]))
    rows = []
    for sample, items in tables.items():
        for item in items:
            row = {"sample": sample, **{key: item[key] for key in (
                "case_id", "mode", "class", "composition", "gt", "prediction", "n_records", "image_ids", "labels",
                "margin")}}
            for layer, terms in item["terms"].items():
                row.update({f"{name}_layer{layer}": value for name, value in terms.items()
                            if name in ("z_N", "z_P_mean", "z_P_min")})
            rows.append(row)
    write_csv(rows, output_dir / "inputs.csv", list(dict.fromkeys(key for row in rows for key in row)))

    report_dir = (PROJECT_ROOT / "reports" / output_dir.relative_to(PROJECT_ROOT / "results")
                  if output_dir.is_relative_to(PROJECT_ROOT / "results") else output_dir)
    print(f"그림: {plot_evidence_margin(report, tables, report_dir)}")
    counts, headline = report["counts"], report["headline"]
    print(f"저장: {output_dir} | 고유 입력 {counts['unique_inputs']} | N 1장 {counts['one_negative_inputs']} | "
          f"PPP {counts['all_positive_inputs']} | 부트스트랩 이미지 {counts['bootstrap_images']}")
    positive, negative, fit = headline["z_P_mean"], headline["z_N"], headline["delta_r2"]
    print(f"[주 분석, 탐색적] L{positive['layer']} {positive['model']}: "
          f"z_P_mean β={fmt(positive['estimate'])} [{fmt(positive['ci_low'])}, {fmt(positive['ci_high'])}] "
          f"(1 SD당 {fmt(positive['estimate_per_sd'])}) | z_N β={fmt(negative['estimate'])} "
          f"[{fmt(negative['ci_low'])}, {fmt(negative['ci_high'])}] (1 SD당 {fmt(negative['estimate_per_sd'])})")
    print(f"  ΔR² (neg_pos − neg) = {fmt(fit['delta_r2'], 4)} [{fmt(fit['ci_low'], 4)}, {fmt(fit['ci_high'], 4)}] | "
          f"R² {fmt(fit['r2'])} (유효 반복 {fit['ci_valid']})")
    for row in report["coefficients"]:
        if row["term"] in ("z_P_mean", "z_P_min") and not row["primary"]:
            print(f"  {row['sample']} L{row['layer']} {row['model']}: {row['term']} β={fmt(row['estimate'])} "
                  f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]")
    for row in report["diagnostics"]:
        print(f"  진단 L{row['layer']}: z_N·z_P 편상관 {fmt(row['partial_correlation_z_N_z_P_mean'])} | "
              f"같은 N 안의 z_P 분산 비율 {fmt(row['within_negative_image_share_of_z_P_mean_variance'])}")
    print("탐색적 분석: STEP 4는 H1용으로 사전 지정된 데이터입니다. 연관성일 뿐 인과가 아니며, 같은 N 비교는 STEP 5-2에서 합니다.")


if __name__ == "__main__":
    main()
