# python3 scripts/download_model.py
# python3 scripts/download_model.py --config configs/models/qwen2_5_vl_3b.yaml

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "cache" / "models"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def download_model(model_name: str = DEFAULT_MODEL) -> Path:
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
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    config = {}
    if args.config:
        with args.config.expanduser().open(encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

    model_name = config.get("model_name") or DEFAULT_MODEL
    model_path = download_model(model_name=model_name)
    print(f"다운로드 완료: {model_path}", flush=True)


if __name__ == "__main__":
    main()
