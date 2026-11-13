"""STEP 5-2b 추출 진입점 (GPU).

STEP 2 추출(run_extraction)을 그대로 호출하고, 실행 환경(GPU 종류·드라이버·CUDA·라이브러리 버전)을 run.json의
`environment`에 기록한다. experiments/step2_hidden_states/run.py는 STEP 5-2 고정본(v2)의 해시 대상이므로 수정하지 않는다.
"""
import argparse
from pathlib import Path

from experiments.step2_hidden_states.run import run_extraction
from src.common.data import read_json, write_json
from src.common.environment import collect_environment
from src.common.paths import PROJECT_ROOT


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step5_position_extract.yaml"


def main():
    parser = argparse.ArgumentParser(description="STEP 5-2b Hidden State 추출 + 실행 환경 기록 (GPU)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, help="파일럿: 케이스 파일 앞 N개")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-repeat", action="store_true")
    parser.add_argument("--verify-reference", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다.")
    before, packages = collect_environment()
    destination = run_extraction(args.config, None, None, args.limit, args.output_dir, False,
                                 args.verify_repeat, args.verify_reference)
    after, _ = collect_environment()
    metadata = read_json(destination / "run.json")
    metadata["environment"] = {**before, "same_gpu_at_end": before.get("gpus") == after.get("gpus")
                               and before.get("nvidia_smi") == after.get("nvidia_smi")}
    write_json(metadata, destination / "run.json", allow_nan=False)
    (destination / "environment_packages.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    gpus = ", ".join(gpu["name"] for gpu in before.get("gpus", [])) or "no GPU"
    drivers = ", ".join(sorted({row["driver_version"] for row in before["nvidia_smi"] or []})) or "unknown"
    print(f"환경 기록: {gpus} | driver {drivers} | torch {before['torch']} (CUDA {before['torch_cuda']}) | "
          f"transformers {before['packages']['transformers']}")


if __name__ == "__main__":
    main()
