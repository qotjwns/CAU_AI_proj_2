import argparse
from pathlib import Path

from src.common.data import normalize_category, read_jsonl
from src.common.paths import PROJECT_ROOT, project_path
from src.visualization.token_spans import group_span_records, plot_token_spans


def main():
    parser = argparse.ArgumentParser(description="추출된 케이스의 이미지와 토큰 구간을 PNG로 시각화")
    parser.add_argument("--manifest-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--category", help="지정한 클래스만 시각화")
    parser.add_argument("--limit", type=int, help="전체 입력에서 저장할 최대 그림 수 (그림 하나 = A/B별 pair 하나)")
    parser.add_argument("--output-dir", type=Path, help="기본: results/ 경로에 대응하는 reports/ 경로 아래 token_spans/")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다.")

    written = 0
    for path in args.manifest_jsonl:
        if args.limit is not None and written >= args.limit:
            break
        source = project_path(path).resolve()
        if args.output_dir:
            output_dir = project_path(args.output_dir)
        else:
            relative_dir = source.parent.relative_to(PROJECT_ROOT / "results")
            output_dir = PROJECT_ROOT / "reports" / relative_dir / "token_spans"
        groups = group_span_records(read_jsonl(source), args.category)
        if args.limit is not None:
            groups = groups[:args.limit - written]
        for group in groups:
            case = group[0]["case"]
            category = normalize_category(case["class"]).replace(" ", "_")
            output = output_dir / case["mode"] / category / f"{case['pair_id']}.png"
            plot_token_spans(group, output)
            written += 1
            print(f"저장: {output}")
    if not written:
        raise ValueError("No saved cases matched. Check that extraction has written records to manifest.jsonl.")
    print(f"총 {written}개 그림 저장")


if __name__ == "__main__":
    main()
