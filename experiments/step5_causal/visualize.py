"""STEP 5-3 그림·탐색 표 (CPU, analysis.json만 읽는다. 새 평가 없음, 판정과 무관).

python -m experiments.step5_causal.visualize
→ reports/step5/qwen2_5_vl_3b/causal/
  descriptive_a_layer_curves.png, descriptive_b_band_curves.png,
  exploratory_b_answer_change_rates.csv, exploratory_b_subgroups.csv
"""
import argparse
from pathlib import Path

from src.common.data import read_json, write_csv
from src.common.paths import project_path, project_relative_path
from src.core.causal_freeze import experiment_config
from src.visualization.causal import b_answer_change_rows, b_subgroup_rows, plot_a_layer_curves, plot_b_band_curves


DEFAULT_OUTPUT = "reports/step5/qwen2_5_vl_3b/causal"


def main():
    parser = argparse.ArgumentParser(description="STEP 5-3 그림·탐색 표 (analysis.json만 읽음)")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = project_path(args.output_dir)
    reports = {e: read_json(project_path(experiment_config(e)[1]["evaluation_dir"]) / "analysis.json") for e in "AB"}
    paths = [plot_a_layer_curves(reports["A"], output / "descriptive_a_layer_curves.png"),
             plot_b_band_curves(reports["B"], output / "descriptive_b_band_curves.png")]
    for name, rows in (("exploratory_b_answer_change_rates.csv", b_answer_change_rows(reports["B"])),
                       ("exploratory_b_subgroups.csv", b_subgroup_rows(reports["B"]))):
        write_csv(rows, output / name, list(rows[0]))
        paths.append(output / name)
    for path in paths:
        print(f"저장: {project_relative_path(path)}")


if __name__ == "__main__":
    main()
