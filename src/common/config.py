import yaml

from src.common.paths import DEFAULT_CONFIG, project_path


def load_config(path=DEFAULT_CONFIG):
    with path.expanduser().open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    for key in ("annotation_path", "image_dir", "output_dir"):
        config[key] = project_path(config[key]).resolve()
    return config
