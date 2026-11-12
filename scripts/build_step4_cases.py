import argparse
from pathlib import Path

from src.common.config import load_config, read_yaml
from src.common.paths import PROJECT_ROOT, project_path
from src.data_manipulate.build_independent import save_independent_cases


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "step4_independent.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 4 독립 검증 케이스 생성 (STEP 1~3 이미지 전역 제외)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    data_config = load_config(project_path(config["source_config"]))
    cases_path, summary = save_independent_cases(args.config, data_config)
    print(f"저장: {cases_path}")
    print(f"클래스 {summary['classes']}개 | 이미지 {summary['images']}장 (STEP 1~3 이미지 {summary['excluded_step1_3_images']}장 제외)")
    print(f"레코드 {summary['records']}개 → 고유 입력 {summary['unique_inputs']}개 | 모드 {summary['by_mode']}")
    print(f"고유 입력 구성: {summary['by_composition_unique']} | 중복: {summary['duplicate_groups']}")


if __name__ == "__main__":
    main()
