import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import read_json, write_csv, write_json
from src.common.paths import PROJECT_ROOT, project_path, project_relative_path
from src.common.threads import configure_cpu_threads
from src.core.probe_apply import apply_frozen_probe
from src.core.probe_freeze import verify_frozen
from src.core.probe_outputs import write_evaluation


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step4.yaml"


def write_hq_outputs(report, result, output_dir):
    hq, layers = report["hq"], result["layers"]
    write_csv(hq["readability"], output_dir / "hq_readability.csv", list(hq["readability"][0]))
    write_csv(hq["localization"], output_dir / "localization.csv", list(hq["localization"][0]))
    write_csv(hq["paired"], output_dir / "zi_vs_hq.csv", list(hq["paired"][0]))
    write_csv(hq["group_comparison"], output_dir / "hq_group_comparison.csv", [
        "feature", "level", "layer", "analysis", "metric", "group", "comparison_group", "n_cases",
        "n_comparison_cases", "n_units", "value", "comparison_value", "difference", "ci_low", "ci_high",
        "ci_target", "ci_valid", "greater", "equal", "less",
    ])
    rows = []
    for case in result["hq_cases"]:
        for position in range(3):
            slot = case["vectors"][position]
            row = {key: case[key] for key in ("mode", "case_id", "class", "composition", "gt", "prediction", "n_records")}
            row.update(position=position + 1, image_id=case["image_ids"][position], label=case["labels"][position])
            for feature, values in result["hq_features"].items():
                row.update({f"p_{feature}_layer{layer}": float(values[slot, i]) for i, layer in enumerate(layers)})
            rows.append(row)
    write_csv(rows, output_dir / "zi_hq_slots.csv", list(rows[0]))


def main():
    configure_cpu_threads()
    parser = argparse.ArgumentParser(description="STEP 4: 고정 z_i Probe·h_q readout을 독립 이미지에 적용해 검증 (CPU)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, help="STEP 4 추출 폴더 (기본: 설정 파일의 run_dir)")
    parser.add_argument("--output-dir", type=Path, help="기본: 설정 파일의 output_dir")
    args = parser.parse_args()

    config = read_yaml(project_path(args.config))
    run_dir = project_path(args.run_dir or config["run_dir"]).resolve()
    output_dir = project_path(args.output_dir or config["output_dir"])
    step3_path = project_path(config["step3_analysis"]) if config.get("step3_analysis") else None
    step3_report = read_json(step3_path) if step3_path and step3_path.exists() else None

    result = apply_frozen_probe(run_dir, config, step3_report)
    report, layers = result["report"], result["layers"]
    report["config"] = project_relative_path(args.config)
    report["output_dir"] = project_relative_path(output_dir)
    write_evaluation(report, result["vectors"], result["slots"], result["predictions"], layers,
                     result["strict_cases"], result["examples"], output_dir)
    write_csv(report["step3_comparison"], output_dir / "step3_comparison.csv",
              ["layer", "metric", "step3", "step4", "step4_ci_low", "step4_ci_high"])
    write_hq_outputs(report, result, output_dir)
    changed = verify_frozen(result["frozen_dir"], result["manifest"])
    write_json({"frozen_dir": report["preregistration"]["frozen_dir"], "verified_before": True,
                "verified_after": not changed, "changed_files": changed}, output_dir / "frozen_verification.json")
    if changed:
        write_json({"status": "invalid", "reason": f"Frozen files changed during evaluation: {changed}"},
                   output_dir / "INVALID.json")
        raise SystemExit(f"무효: 평가 중 고정 파일이 바뀌었습니다 {changed}. 이 실행 결과를 사용하지 마세요.")

    pre = report["preregistration"]
    trace = report["denominators"]
    print(f"저장: {output_dir} | 상태 {report['status']} | 임계값 {pre['threshold']} (고정)")
    print(f"분모: 레코드 {trace['records']} → 고유 입력 {trace['unique_inputs']} (제외 {sum(trace['excluded'].values())})")
    zi_layer, hq_layer = pre["z_i"]["primary_layer"], pre["h_q"]["primary_layer"]
    scope = f"h_q_{pre['h_q']['primary_scope']}"
    key = next(row for row in report["headline"] if row["layer"] == zi_layer)
    rate = key["strict_discrepancy_rate"]
    rate_text = "N/A" if rate is None else (
        f"{rate:.3f} [{key['strict_discrepancy_rate_ci_low']:.3f}, {key['strict_discrepancy_rate_ci_high']:.3f}]")
    print(f"H1-A z_i L{zi_layer}: VLM False-Yes {key['vlm_false_yes']} 중 Strict {key['fy_strict']} = {rate_text}")
    for row in report["hq"]["localization"]:
        if row["feature"] == scope and row["layer"] == hq_layer and row["scope"] in ("all", "vlm_group"):
            accuracy = "N/A" if row["accuracy"] is None else f"{row['accuracy']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
            print(f"H1-B {scope} L{hq_layer} N 위치 정확도 ({row['value']}, n={row['n']}): {accuracy} (우연 0.333)")
    print("짝 비교(z_i vs h_q, 같은 입력)는 zi_vs_hq.csv. McNemar p는 케이스 독립 가정이므로 CI와 함께 봅니다.")


if __name__ == "__main__":
    main()
