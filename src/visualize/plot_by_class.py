import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from src.common.data import normalize_category
from src.common.paths import PROJECT_ROOT, project_path


def plot_pair(cases, output):
    first = cases[0]
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 3 * len(cases)), squeeze=False, constrained_layout=True)
    fig.suptitle(f"Mode {first['mode']} | {first['class']}\n{first['pair_id']}", fontsize=11)
    for row, case in enumerate(cases):
        image_dir = project_path(case["image_dir"])
        for column, (name, label) in enumerate(zip(case["image_order"], case["image_labels"])):
            axis = axes[row, column]
            axis.imshow(plt.imread(image_dir / name))
            title = f"Image {column + 1}: {label} | {name}"
            if column == 0:
                title = f"{case['composition']} | GT: {case['gt']}\n{title}"
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="A/B 케이스를 클래스별 폴더에 pair 단위로 시각화")
    parser.add_argument("--cases-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--category", help="지정하지 않으면 모든 클래스")
    parser.add_argument("--limit", type=int, default=0, help="클래스별 pair 수 (기본 0: 전체)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "class_plots")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    for source in args.cases_jsonl:
        source = source.expanduser().resolve()
        classes = {}
        with source.open(encoding="utf-8") as input_file:
            for line in input_file:
                if not line.strip():
                    continue
                case = json.loads(line)
                category = case["class"]
                if args.category and normalize_category(category) != normalize_category(args.category):
                    continue
                pairs = classes.setdefault(category, {})
                pairs.setdefault(case["pair_id"], []).append(case)

        for category, pairs in classes.items():
            folder = output_dir / source.stem / normalize_category(category).replace(" ", "_")
            folder.mkdir(parents=True, exist_ok=True)
            selected = list(pairs.items())
            if args.limit:
                selected = selected[:args.limit]
            for pair_id, cases in selected:
                plot_pair(cases, folder / f"{pair_id}.png")
            print(f"{source.stem} | {category}: {len(selected)}개 저장 → {folder}")


if __name__ == "__main__":
    main()
