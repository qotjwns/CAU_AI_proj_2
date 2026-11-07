import json


def read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_json(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.write("\n")


def normalize_category(category):
    return category.strip().casefold().replace("_", " ")
