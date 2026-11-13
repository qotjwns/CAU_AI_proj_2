"""STEP 5-3 사전 등록 고정 (CPU, 연습 실행·본 실행 전, 덮어쓰기 금지).

python -m experiments.step5_causal.freeze --experiment A
python -m experiments.step5_causal.freeze --experiment B
"""
import argparse

from src.core.causal_freeze import freeze_causal


def main():
    parser = argparse.ArgumentParser(description="STEP 5-3 사전 등록 고정 (CPU)")
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    args = parser.parse_args()
    frozen_dir, manifest = freeze_causal(args.experiment)
    print(f"저장: {frozen_dir / 'manifest.json'} (실험 {args.experiment}, 버전 {manifest['version']}, {manifest['created_at']})")
    print(f"계획: {manifest['plan_counts']}")
    for path, digest in manifest["files_sha256"].items():
        print(f"  {path}: sha256 {digest[:16]}…")
    print("이후 가설·결과 변수·판정 기준·계획·코드를 바꾸지 않습니다.")


if __name__ == "__main__":
    main()
