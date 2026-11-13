"""실행 환경 기록: GPU 종류·드라이버·CUDA·라이브러리 버전 (실행 간 수치 차이 추적용)."""
import hashlib
from importlib import metadata
import platform
import subprocess
import sys


PACKAGES = ("torch", "transformers", "accelerate", "tokenizers", "safetensors", "pillow", "numpy", "qwen-vl-utils",
            "torchvision", "flash-attn")


def nvidia_smi():
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    keys = ("index", "name", "driver_version", "memory_total", "compute_capability")
    return [dict(zip(keys, (part.strip() for part in line.split(",")))) for line in output.strip().splitlines()]


def installed_packages():
    """설치된 전체 패키지 'name==version' 목록 (정렬)."""
    names = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            names[name.lower()] = f"{name}=={distribution.version}"
    return [names[key] for key in sorted(names)]


def collect_environment():
    import torch

    packages = installed_packages()
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    environment = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(), "packages": versions,
        "installed_packages_sha256": hashlib.sha256("\n".join(packages).encode()).hexdigest(),
        "nvidia_smi": nvidia_smi(),
    }
    if torch.cuda.is_available():
        environment["gpus"] = [
            {"index": i, "name": torch.cuda.get_device_name(i),
             "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
             "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())]
    return environment, packages
