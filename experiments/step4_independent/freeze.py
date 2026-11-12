import argparse
from pathlib import Path

from src.common.config import read_yaml
from src.common.paths import PROJECT_ROOT, project_path
from src.core.probe_freeze import freeze_probes


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step4.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 4 GPU 추출 전에 z_i Probe와 h_q readout을 학습·고정 (CPU)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reason", help="이전 고정 버전이 있을 때 새 버전을 만드는 이유 (필수)")
    args = parser.parse_args()
    config = read_yaml(project_path(args.config))
    step3_config_path = Path(config["training"]["step3_config"])
    step3_config = read_yaml(project_path(step3_config_path))
    output_dir, manifest = freeze_probes(args.config, config, step3_config_path, step3_config, args.reason)
    print(f"저장: {output_dir} (버전 {manifest['version']}, 이유: {manifest['reason']})")
    print(f"평가 케이스 고정: {manifest['evaluation']['cases']} sha256 {manifest['evaluation']['cases_sha256'][:12]}… "
          f"({manifest['evaluation']['records']}개) | 학습 데이터 파일 {manifest['training_data']['files_listed']}개 해시 기록")
    for name, digest in manifest["files"].items():
        print(f"  {name}: sha256 {digest}")
    print(f"z_i Probe {manifest['zi_probes']}개 (미수렴 {manifest['zi_not_converged']}) | "
          f"h_q readout {manifest['hq_readouts']}개 (미수렴 {manifest['hq_not_converged']}) | "
          f"학습 이미지 {manifest['training_images']}장 | 임계값 {manifest['threshold']}")
    print("이후 STEP 4 데이터를 본 뒤에는 재학습·임계값·레이어·위치·범위를 바꾸지 않습니다.")


if __name__ == "__main__":
    main()
