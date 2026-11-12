import csv
import json


def read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def iter_jsonl(source):
    for line in source:
        if line.strip():
            yield json.loads(line)


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        yield from iter_jsonl(source)


def write_json(data, output_path, *, allow_nan=True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=allow_nan)
        output.write("\n")


def normalize_category(category):
    return category.strip().casefold().replace("_", " ")


def write_csv(rows, output_path, columns):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })
