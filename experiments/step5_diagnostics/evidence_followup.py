import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import write_csv, write_json
from src.common.paths import PROJECT_ROOT, project_path, project_relative_path
from src.common.threads import configure_cpu_threads
from src.core.evidence_followup import VARIANTS, run_followup, summary_rows
from src.visualization.evidence_followup import plot_coefficients, plot_models, plot_terciles


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step5_evidence_followup.yaml"
SECTIONS = ("a1", "a2", "a3", "a4", "a5", "a5_rates", "a6", "a7")


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:+.{digits}f}"


def main():
    configure_cpu_threads()
    parser = argparse.ArgumentParser(description="STEP 5-1 보완 분석 A1~A7: 집계 규칙 진단 (CPU, exploratory)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    output_dir = project_path(args.output_dir or config["output_dir"])

    report, results = run_followup(config)
    summary = summary_rows(results, config["primary_layer"])
    report.update(config=project_relative_path(args.config), output_dir=project_relative_path(output_dir),
                  summary=summary, results={variant: {key: results[variant][key] for key in
                                                      (*SECTIONS, "n_inputs", "n_inputs_a7")}
                                            for variant in VARIANTS},
                  calibration=results["all"]["calibration"], reliability=results["all"]["reliability"])
    write_json(report, output_dir / "analysis.json", allow_nan=False)
    write_csv(summary, output_dir / "summary.csv", list(summary[0]))
    for section in SECTIONS:
        rows = [{"variant": variant, **row} for variant in VARIANTS for row in results[variant][section]]
        write_csv(rows, output_dir / f"{section}.csv", list(dict.fromkeys(key for row in rows for key in row)))
    write_csv(results["all"]["calibration"], output_dir / "calibration.csv", list(results["all"]["calibration"][0]))
    write_csv(results["all"]["reliability"], output_dir / "reliability.csv", list(results["all"]["reliability"][0]))

    report_dir = (PROJECT_ROOT / "reports" / output_dir.relative_to(PROJECT_ROOT / "results")
                  if output_dir.is_relative_to(PROJECT_ROOT / "results") else output_dir)
    for figure in (plot_coefficients(report, results, report_dir), plot_models(report, results, report_dir),
                   plot_terciles(report, results, report_dir)):
        print(f"그림: {figure}")
    print(f"저장: {output_dir}")
    for check in report["precheck"]:
        print(f"사전 점검 ({check['data']}): {check['n_inputs']}개 중 부호 불일치 {check['n_inconsistent']}개 {check['by_kind']}")
    print(f"{'분석':<4} {'항목':<52} {'전체 (n) 추정 [95% CI]':<40} 부호 일치만 (n) 추정 [95% CI]")
    for row in summary:
        cells = [f"({row[f'{p}_n']}) {fmt(row[f'{p}_estimate'])} [{fmt(row[f'{p}_ci_low'])}, {fmt(row[f'{p}_ci_high'])}]"
                 for p in ("all", "consistent")]
        print(f"{row['analysis']:<4} {row['quantity']:<52} {cells[0]:<40} {cells[1]}")
    print("탐색적 분석입니다. 확증은 STEP 5-2(통제 입력)에서 합니다.")


if __name__ == "__main__":
    main()
