# python scripts/download_data.py

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile


DATASETS = {
    "images": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "archive": "val2017.zip",
        "destination": "coco",
        "expected": "val2017/",
    },
    "annotations": {
        "url": (
            "https://huggingface.co/datasets/kdexd/coco-rem/"
            "resolve/main/instances_valrem.json.zip"
        ),
        "archive": "instances_valrem.json.zip",
        "destination": "coco_rem",
        "expected": "instances_valrem.json",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "cache" / "downloads"


def download_data(dataset_name: str) -> Path:
    dataset = DATASETS[dataset_name]
    save_dir = DATA_DIR / dataset["destination"]
    data_path = save_dir / dataset["expected"]

    print(f"다운로드 데이터: {dataset_name}", flush=True)
    print(f"저장 경로: {data_path}", flush=True)

    if data_path.exists():
        print("이미 존재하는 데이터입니다.", flush=True)
        return data_path

    save_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive = CACHE_DIR / dataset["archive"]

    if not archive.exists():
        with TemporaryDirectory(prefix=".download-", dir=CACHE_DIR) as work:
            partial = Path(work) / dataset["archive"]
            urlretrieve(dataset["url"], partial)
            partial.replace(archive)

    with TemporaryDirectory(prefix=".extract-", dir=save_dir) as work:
        with ZipFile(archive) as zipped:
            zipped.extractall(work)
        (Path(work) / dataset["expected"]).rename(data_path)

    return data_path


def main() -> None:
    for dataset_name in DATASETS:
        data_path = download_data(dataset_name=dataset_name)
        print(f"다운로드 완료: {data_path}", flush=True)


if __name__ == "__main__":
    main()
