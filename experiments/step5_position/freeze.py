import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.paths import PROJECT_ROOT, project_path
from src.core.position_freeze import freeze_position


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step5_position.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 5-2b 사전 등록 고정: 파일럿·본 추출 전에 케이스·가설·판정 기준 해시 고정 (CPU)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    frozen_dir, manifest = freeze_position(args.config, config)
    effect = manifest["preregistration"]["minimum_effect"]
    print(f"저장: {frozen_dir / 'manifest.json'} (버전 {manifest['version']}, {manifest['created_at']})")
    print(f"레코드 {manifest['counts']['records']} {manifest['counts']['by_role']} | 계열 {manifest['counts']['series']} | "
          f"N {manifest['counts']['n_images_anchor']}")
    print(f"최소 의미 효과: False-Yes +{effect['false_yes_rate']} (H1b·H2b, 97.5% CI) | margin: 판정 없음")
    for path, digest in manifest["files_sha256"].items():
        print(f"  {path}: sha256 {digest[:16]}…")
    print("이후 가설·결과 변수·판정 기준·케이스·코드를 바꾸지 않습니다.")


if __name__ == "__main__":
    main()
