import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import normalize_category, read_jsonl
from src.common.paths import PROJECT_ROOT
from src.visualization.cases import group_cases_by_class, plot_pair


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
        classes = group_cases_by_class(read_jsonl(source), args.category)
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
