import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from src.common.config import read_yaml
from src.common.paths import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "cache" / "models"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models" / "qwen2_5_vl_3b.yaml"


def download_model(model_name: str) -> Path:
    save_dir = MODEL_DIR / model_name

    print(f"다운로드 모델: {model_name}", flush=True)
    print(f"저장 경로: {save_dir}", flush=True)

    model_path = snapshot_download(
        repo_id=model_name,
        repo_type="model",
        local_dir=str(save_dir),
    )

    return Path(model_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = read_yaml(args.config)
    model_path = download_model(config["model_name"])
    print(f"다운로드 완료: {model_path}", flush=True)


if __name__ == "__main__":
    main()
