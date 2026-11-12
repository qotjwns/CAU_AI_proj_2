import argparse
from pathlib import Path

from src.common.config import load_config, read_yaml
from src.common.paths import PROJECT_ROOT, project_path
from src.data_manipulate.build_accumulation import save_accumulation_cases


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "step5_accumulation.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 5-2 입력 수준 실험 케이스 생성 (STEP 4 이미지만 사용)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    data_config = load_config(project_path(config["source_config"]))
    target, summary = save_accumulation_cases(args.config, data_config)
    print(f"저장: {target['cases']}")
    print(f"설계: {target['design']} | 사용 이미지 목록(STEP 1~5): {target['used']}")
    print(f"레코드 {summary['records']}개 → 고유 입력 {summary['unique_inputs']}개 (중복 {summary['duplicate_records']}) | "
          f"클래스 {summary['classes']} | N {summary['negatives']}장 | 계열 {summary['series']}개")
    print(f"조건 {summary['by_condition']} | 장수 {summary['by_n_images']}")
    print(f"P {summary['P_images_used']}장 사용, P 한 장당 계열 수 {summary['series_per_P_image']} | "
          f"층 크기 {summary['stratum_sizes']} | Probe가 N으로 읽어 제외한 P {summary['excluded_read_as_N']}장")


if __name__ == "__main__":
    main()
