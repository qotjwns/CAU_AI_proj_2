import argparse
from pathlib import Path

from src.data_manipulate.build_cases import save_cases


def main():
    parser = argparse.ArgumentParser(description="A: 한 장 교체 비교, B: 같은 이미지의 위치 비교")
    parser.add_argument("--groups-json", type=Path, required=True)
    parser.add_argument("--mode", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    save_cases(args.groups_json, args.mode, args.output)


if __name__ == "__main__":
    main()
