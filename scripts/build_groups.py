import argparse
from pathlib import Path

from src.common.config import load_config
from src.common.paths import DEFAULT_CONFIG
from src.data_manipulate.build_groups import generate_groups


def main():
    parser = argparse.ArgumentParser(
        description="클래스별 양성 8장·음성 후보 8장에서 양성 4장·음성 8장의 그룹 5개 생성"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--category", dest="categories", nargs="+",
        help="클래스 이름 또는 폴더명 목록. 지정하면 이 클래스들만 사용합니다.",
    )
    selection.add_argument(
        "--num-classes", type=int,
        help="자동 선택할 클래스 수 (기본: YAML 설정)",
    )
    parser.add_argument("--seed", type=int, help="샘플링 시드 (기본: YAML 설정)")
    parser.add_argument("--pool-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.num_classes is not None:
        config["num_classes"] = args.num_classes
        config["categories"] = []
    if args.categories is not None:
        config["categories"] = args.categories
    generate_groups(config, args.pool_dir, args.output_dir)


if __name__ == "__main__":
    main()
