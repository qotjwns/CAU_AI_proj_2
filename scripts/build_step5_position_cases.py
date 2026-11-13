import argparse
from pathlib import Path

from src.common.paths import PROJECT_ROOT
from src.data_manipulate.build_position import save_position_cases


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "step5_position.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 5-2b 위치 격자 케이스 생성 (5-2 이미지만, 덮어쓰기 금지)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    target, summary = save_position_cases(args.config)
    print(f"저장: {target['cases']}")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
