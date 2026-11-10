from src.common.paths import project_path


LABELS = ("Yes", "No")


def build_messages(case):
    image_dir = project_path(case["image_dir"])
    content = [
        {"type": "image", "image": str(image_dir / name)}
        for name in case["image_order"]
    ]
    content.append({"type": "text", "text": case["prompt"]})
    return [{"role": "user", "content": content}]


def parse_prediction(raw_response):
    answer = raw_response.strip().rstrip(".!?").strip().casefold()
    return {"yes": "Yes", "no": "No"}.get(answer)
