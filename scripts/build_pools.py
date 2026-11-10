import argparse
from pathlib import Path

from src.common.config import load_config
from src.common.data import read_json, write_json
from src.common.paths import DEFAULT_CONFIG
from src.data_manipulate.build_cases import save_cases
from src.data_manipulate.build_groups import generate_groups
from src.data_manipulate.build_pools import build_pool_ids, save_pools, summarize_pools


def main():
    parser = argparse.ArgumentParser(description="설정에 따라 클래스별 양성·음성 후보 목록 생성")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("A", "B"), help="지정하면 그룹과 해당 모드의 케이스까지 생성")
    args = parser.parse_args()
    config = load_config(args.config)
    data = read_json(config["annotation_path"])
    pools = build_pool_ids(data, config)
    summary = summarize_pools(data, pools, config)
    save_pools(data, pools, config)

    print(f"전체 이미지: {summary['image_count']:,}장")
    print(f"카테고리: {summary['category_count']:,}개")
    print(f"양성 기준: bbox 면적 / 이미지 면적 >= {config['min_bbox_area_ratio']:.0%}")

    output_path = config["output_dir"] / "pool_counts.json"
    write_json(summary, output_path)
    print(f"\n집계 저장: {output_path}")
    print(f"양성·음성 후보 이미지 목록 저장: {config['output_dir'] / 'pools'}")
    if args.mode:
        groups_json = generate_groups(config)
        save_cases(groups_json, args.mode)


if __name__ == "__main__":
    main()
