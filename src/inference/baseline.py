import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm
import yaml

from src.common.paths import PROJECT_ROOT, project_path


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "inference" / "baseline.yaml"


def load_inference_config(path):
    with project_path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if config["inference_method"] != "baseline":
        raise ValueError("inference_method must be baseline")
    with project_path(config["model_config"]).open(encoding="utf-8") as source:
        config["model_name"] = yaml.safe_load(source)["model_name"]
    config["cases_jsonl"] = [project_path(path) for path in config["cases_jsonl"]]
    config["output_dir"] = project_path(config["output_dir"])
    return config


def load_model(model_name):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_dir = PROJECT_ROOT / "cache" / "models" / model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    # Unpacking also checks that each answer is exactly one token.
    (yes_id,) = processor.tokenizer.encode("Yes", add_special_tokens=False)
    (no_id,) = processor.tokenizer.encode("No", add_special_tokens=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, local_files_only=True,
    ).to(device).eval()
    return model, processor, (yes_id, no_id)


def parse_prediction(raw_response):
    answer = raw_response.strip().rstrip(".!?").strip().casefold()
    return {"yes": "Yes", "no": "No"}.get(answer)


def infer_case(case, model, processor, answer_token_ids):
    import torch

    image_dir = project_path(case["image_dir"])
    content = [
        {"type": "image", "image": str(image_dir / name)}
        for name in case["image_order"]
    ]
    content.append({"type": "text", "text": case["prompt"]})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=8,
            return_dict_in_generate=True,
            output_logits=True,
        )

    input_length = inputs["input_ids"].shape[1]
    raw_response = processor.tokenizer.decode(
        generated.sequences[0, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    # Raw logits at the FIRST generated position, before decoding processors.
    first_logits = generated.logits[0][0]
    yes_logit, no_logit = [first_logits[token_id].item() for token_id in answer_token_ids]
    image_tokens = (
        (inputs["input_ids"] == model.config.image_token_id)
        & inputs["attention_mask"].bool()
    )
    return {
        "prediction": parse_prediction(raw_response),
        "raw_response": raw_response,
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "logit_margin": yes_logit - no_logit,
        "input_token_count": inputs["attention_mask"].sum().item(),
        "image_token_count": image_tokens.sum().item(),
    }


def build_result(case, inference, config):
    image_dir = Path(case["image_dir"])
    return {
        "case_id": case["case_id"],
        "group_id": case["group_id"],
        "pair_id": case["pair_id"],
        "mode": case["mode"],
        "inference_method": config["inference_method"],
        "model_name": config["model_name"],
        "object": case["class"],
        "composition": "".join(case["image_labels"]),
        "images": [(image_dir / name).as_posix() for name in case["image_order"]],
        "per_image_gt": [int(label == "P") for label in case["image_labels"]],
        "negative_indices": case["negative_positions"],
        "prompt": case["prompt"],
        "gt": case["gt"],
        **inference,
        "correct": inference["prediction"] == case["gt"],
    }


def run_baseline(config_path):
    config = load_inference_config(config_path)
    model, processor, answer_token_ids = load_model(config["model_name"])
    config["output_dir"].mkdir(parents=True, exist_ok=True)

    for cases_path in config["cases_jsonl"]:
        output_name = f"{cases_path.stem.removesuffix('_cases')}_results.jsonl"
        output_path = config["output_dir"] / output_name
        with (
            cases_path.open(encoding="utf-8") as source,
            output_path.open("x", encoding="utf-8") as destination,
        ):
            total = sum(1 for line in source if line.strip())
            source.seek(0)
            cases = (json.loads(line) for line in source if line.strip())
            for case in tqdm(cases, total=total, desc=cases_path.stem, unit="case"):
                inference = infer_case(case, model, processor, answer_token_ids)
                result = build_result(case, inference, config)
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                destination.flush()
        print(f"저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="JSONL 케이스를 순서대로 baseline 추론")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run_baseline(args.config)


if __name__ == "__main__":
    main()
