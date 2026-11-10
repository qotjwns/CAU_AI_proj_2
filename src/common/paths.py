import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "balanced3.yaml"


def project_path(path):
    return PROJECT_ROOT / Path(path).expanduser()


def project_relative_path(path):
    return Path(os.path.relpath(project_path(path), PROJECT_ROOT)).as_posix()
