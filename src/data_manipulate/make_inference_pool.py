# python src/data_manipulate/make_inference_pool.py --config configs/data/balanced3.yaml

import argparse
import hashlib
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import normalize_category, read_json, write_json
from src.common.config import load_config
from src.common.paths import DEFAULT_CONFIG


IMAGES_PER_LABEL = 8
POSITIVE_GROUPS = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 2, 4, 6),
    (1, 3, 5, 7),
    (0, 3, 4, 7),
)


def select_category_dirs(pool_dir, categories, num_classes):
    directories = [path.parent for path in sorted(pool_dir.glob("*/positive.json"))]
    if categories:
        by_name = {}
        for directory in directories:
            by_name[normalize_category(directory.name)] = directory
            by_name[normalize_category(directory.name.partition("_")[2])] = directory
        selected = [by_name[normalize_category(category)] for category in categories]
        if len(set(selected)) != len(selected):
            raise ValueError("같은 클래스가 중복 지정되었습니다.")
        return selected
    if num_classes < 1:
        raise ValueError("--num-classes는 1 이상이어야 합니다.")

    available = []
    for category_dir in directories:
        positive = read_json(category_dir / "positive.json")
        negative = read_json(category_dir / "negative_candidates.json")
        positive_count = len(positive["images"])
        negative_count = len(negative["images"])
        if min(positive_count, negative_count) < IMAGES_PER_LABEL:
            continue
        available.append((positive_count, positive["category_id"], category_dir))

    available.sort(key=lambda item: (-item[0], item[1]))
    if len(available) < num_classes:
        raise ValueError(
            f"양성·음성 후보가 각각 8장 이상인 클래스는 {len(available)}개입니다. "
            f"요청한 {num_classes}개를 만들 수 없습니다."
        )
    return [item[2] for item in available[:num_classes]]


def sample_images(pool, seed):
    rng = random.Random(f"{seed}:{pool['category_id']}:{pool['pool_type']}")
    candidates = sorted(pool["images"], key=lambda image: image["image_id"])
    return rng.sample(candidates, k=IMAGES_PER_LABEL)


def build_groups(category_dir, seed):
    positive = read_json(category_dir / "positive.json")
    negative = read_json(category_dir / "negative_candidates.json")
    positive_images = sample_images(positive, seed)
    negative_images = sample_images(negative, seed)
    positive_names = [Path(image["image_path"]).name for image in positive_images]
    negative_names = [Path(image["image_path"]).name for image in negative_images]
    image_ids = {Path(image["image_path"]).name: image["image_id"] for image in positive_images + negative_images}
    category_name = positive["category_name"].replace(" ", "_")
    groups = {}
    for group_index, indices in enumerate(POSITIVE_GROUPS, start=1):
        names = [positive_names[index] for index in indices]
        groups[f"group_{group_index:02d}_{category_name}"] = {
            "class": positive["category_name"],
            "pos_image": names,
            "neg_image": negative_names,
            "image_id_by_name": {name: image_ids[name] for name in names + negative_names},
        }
    return groups


def generate_groups(config, pool_dir=None, output_dir=None):
    pool_dir = (pool_dir or config["output_dir"] / "pools").expanduser().resolve()
    output_dir = (output_dir or config["output_dir"] / "inference_pools").expanduser().resolve()
    category_dirs = select_category_dirs(pool_dir, config["categories"], config["num_classes"])
    groups = {}
    for category_dir in category_dirs:
        groups.update(build_groups(category_dir, config["seed"]))
    scope = f"top{config['num_classes']}"
    if config["categories"]:
        names = "\n".join(path.name for path in category_dirs)
        scope = f"classes{len(category_dirs)}_{hashlib.sha256(names.encode()).hexdigest()[:8]}"
    output_path = output_dir / f"inference_groups_{scope}_seed{config['seed']}.json"
    write_json({
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in config.items()},
        "groups": groups,
    }, output_path)
    print(f"전체: {len(category_dirs)}클래스, {len(groups)}그룹")
    print(f"저장: {output_path}")
    return output_path


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
