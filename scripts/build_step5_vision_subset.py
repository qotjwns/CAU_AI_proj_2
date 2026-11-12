import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.paths import PROJECT_ROOT, project_path
from src.data_manipulate.build_accumulation import save_vision_subset


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "step5_accumulation.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 5-2 비전 float32 민감도 분석용 부분집합 (seed로만 선택)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    output, summary = save_vision_subset(read_yaml(project_path(args.config)))
    print(f"저장: {output} | {summary}")


if __name__ == "__main__":
    main()
