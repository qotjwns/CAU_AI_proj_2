import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.paths import PROJECT_ROOT, project_path
from src.core.accumulation_freeze import freeze_accumulation


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step5_accumulation.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 5-2 사전 등록 고정: 파일럿·본 추출 전에 케이스·가설·판정 기준 해시 고정 (CPU)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    frozen_dir, manifest = freeze_accumulation(args.config, config)
    effect = manifest["preregistration"]["minimum_effect"]
    print(f"저장: {frozen_dir / 'manifest.json'} (버전 {manifest['version']}, {manifest['created_at']})")
    print(f"레코드 {manifest['counts']['records']} | 고유 입력 {manifest['counts']['unique_inputs']} | "
          f"N {manifest['counts']['negatives']} | 계열 {manifest['counts']['series']}")
    print(f"최소 의미 효과: False-Yes +{effect['false_yes_rate']} | margin +{effect['margin']} (주 margin: fp32 변형)")
    for path, digest in manifest["files_sha256"].items():
        print(f"  {path}: sha256 {digest[:16]}…")
    print("이후 가설·결과 변수·판정 기준·케이스를 바꾸지 않습니다.")


if __name__ == "__main__":
    main()
