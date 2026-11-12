import argparse
from pathlib import Path

from src.common.data import write_csv, write_json
from src.common.paths import PROJECT_ROOT, project_path, project_relative_path
from src.common.threads import configure_cpu_threads
from src.core.context_noise import run_context_noise


def main():
    configure_cpu_threads()
    parser = argparse.ArgumentParser(description="STEP 2 z_i 문맥 효과 vs bf16 수치 잡음 (위치별 코사인 거리, CPU, 읽기 전용)")
    parser.add_argument("--run-dir", type=Path, default=Path("results/step2/qwen2_5_vl_3b"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/step5_diagnostics/qwen2_5_vl_3b/context_noise"))
    args = parser.parse_args()
    config = {"atol": 1e-5, "rtol": 1e-5, "threshold": 0.001, "step5_pilot_max_cosine_distance": 3.4e-4}
    report, rows = run_context_noise(project_path(args.run_dir).resolve(), config)
    output_dir = project_path(args.output_dir)
    report.update(run_dir=project_relative_path(args.run_dir), status="diagnostic", rows=rows,
                  note="STEP 2 outputs are read only. Noise = same image and same preceding images in different inputs.")
    write_json(report, output_dir / "analysis.json", allow_nan=False)
    write_csv(rows, output_dir / "summary.csv", list(rows[0]))
    print(f"저장: {output_dir} | 슬롯 {report['counts']['slots']}")
    print(f"{'비교':<26}{'위치':>4}{'레이어':>6}{'n':>7}{'중앙값':>11}{'p95':>11}{'최대':>11}{'>0.001':>8}{'중앙값/잡음':>11}")
    for row in rows:
        ratio = row["median_over_noise_median"]
        print(f"{row['comparison']:<26}{row['position']:>4}{row['layer']:>6}{row['n']:>7}{row['median']:>11.2e}"
              f"{row['p95']:>11.2e}{row['max']:>11.2e}{row['share_above_threshold']:>8.2f}"
              f"{(f'{ratio:.0f}x' if ratio else 'NA'):>11}")


if __name__ == "__main__":
    main()
