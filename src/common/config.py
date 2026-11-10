from pathlib import Path

import yaml

from src.common.paths import DEFAULT_CONFIG, project_path


def read_yaml(path):
    with Path(path).expanduser().open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def load_config(path=DEFAULT_CONFIG):
    config = read_yaml(path)
    for key in ("annotation_path", "image_dir", "output_dir"):
        config[key] = project_path(config[key]).resolve()
    return config


def load_inference_config(path):
    config = read_yaml(project_path(path))
    model_config = read_yaml(project_path(config["model_config"]))
    config["model_name"] = model_config["model_name"]
    config["cases_jsonl"] = [project_path(path) for path in config["cases_jsonl"]]
    config["output_dir"] = project_path(config["output_dir"])
    return config
