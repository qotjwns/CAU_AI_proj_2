import argparse
import hashlib
from itertools import islice
import json
from pathlib import Path

import torch
from tqdm import tqdm
import transformers

from src.common.config import load_inference_config
from src.common.data import normalize_category, read_jsonl, write_json
from src.common.paths import PROJECT_ROOT, project_path, project_relative_path
from src.core.features import compare_features, compare_tensors, extract_case_features
from src.core.inference import answer_variant_ids
from src.core.model import load_model


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "step2.yaml"


def select_cases(paths, category=None, limit=None):
    def records():
        for path in paths:
            for case in read_jsonl(path):
                if category and normalize_category(case["class"]) != normalize_category(category):
                    continue
                yield path, case

    selected = list(islice(records(), limit))
    if not selected:
        raise ValueError("No cases matched the input and category selection")
    keys = [(case["mode"], case["case_id"]) for _, case in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate (mode, case_id) in the selected cases")
    return selected


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_inference(left, right, atol, rtol):
    logits = compare_tensors(
        torch.tensor([left["yes_logit"], left["no_logit"], left["logit_margin"]]),
        torch.tensor([right["yes_logit"], right["no_logit"], right["logit_margin"]]),
        atol, rtol,
    )
    same_response = left["raw_response"] == right["raw_response"]
    return {"same_response": same_response, "logits": logits, "passed": same_response and logits["allclose"]}


def run_extraction(
    config_path, cases_jsonl=None, category=None, limit=None, output_dir=None,
    save_token_states=False, verify_repeat=False, verify_reference=False,
):
    config = load_inference_config(config_path)
    if cases_jsonl is not None:
        config["cases_jsonl"] = [project_path(path) for path in cases_jsonl]
    selected = select_cases(config["cases_jsonl"], category, limit)
    destination = project_path(output_dir) if output_dir else config["output_dir"]
    if output_dir is None:
        if category:
            destination /= normalize_category(category).replace(" ", "_")
        if limit is not None:
            destination /= f"pilot_{limit}"
    destination.mkdir(parents=True, exist_ok=True)
    raw_dir = destination / "raw"
    raw_dir.mkdir()

    saved_config = {
        **config,
        "cases_jsonl": [project_relative_path(path) for path in config["cases_jsonl"]],
        "output_dir": project_relative_path(destination),
    }
    source_paths = list(dict.fromkeys(path for path, _ in selected))
    metadata = {
        "status": "incomplete", "config": saved_config, "category": category,
        "limit": limit, "selected_cases": len(selected), "written_cases": 0,
        "save_token_states": save_token_states,
        "verify_repeat": verify_repeat, "verify_reference": verify_reference,
        "layer_convention": "huggingface_hidden_states",
        "pooling": f"mean_{config['pooling_dtype']}",
        "sources": [{"path": project_relative_path(path), "sha256": sha256_file(path)} for path in source_paths],
        "versions": {"torch": str(torch.__version__), "transformers": transformers.__version__},
    }
    write_json(metadata, destination / "run.json", allow_nan=False)
    model, processor, answer_token_ids = load_model(config["model_name"], config.get("vision_dtype"))
    metadata["vision_dtype"] = str(model.model.visual.dtype)
    metadata.update(
        model_config=model.config.to_dict(),
        image_processor=processor.image_processor.to_dict(),
        chat_template=processor.chat_template,
        device=str(model.device), dtype=str(next(model.get_decoder().parameters()).dtype),
        attention_implementation=model.get_decoder().config._attn_implementation,
        final_layer_after_norm=len(model.get_decoder().layers),
    )
    write_json(metadata, destination / "run.json", allow_nan=False)
    # STEP 5-2 옵션: 설정에 없으면 STEP 2~4와 같은 기본값(이미지 3장, 변형 margin·fp32 margin 없음)
    answer_variants = answer_variant_ids(processor.tokenizer) if config.get("answer_variants", False) else None
    if answer_variants is not None:
        metadata["answer_variants"] = {
            label: [{"id": token, "text": processor.tokenizer.decode([token])} for token in ids]
            for label, ids in answer_variants.items()}
        write_json(metadata, destination / "run.json", allow_nan=False)
    options = {
        "layers": config["layers"],
        "expected_images": config.get("expected_images", 3),
        "answer_variants": answer_variants,
        "fp32_margin": config.get("fp32_margin", False),
        "expected_hidden_size": config["expected_hidden_size"],
        "pooling_dtype": config["pooling_dtype"],
        "save_token_states": save_token_states,
        **config["verification"],
    }
    verification_passed = True
    with (destination / "manifest.jsonl").open("x", encoding="utf-8") as manifest:
        for index, (source, case) in enumerate(tqdm(selected, desc="Extract hidden states", unit="case")):
            features, inference, reference = extract_case_features(
                case, model, processor, answer_token_ids,
                verify_reference=verify_reference, **options,
            )
            checks = {}
            if reference is not None:
                checks["reference"] = reference
            if verify_repeat:
                repeated, repeated_inference, _ = extract_case_features(
                    case, model, processor, answer_token_ids,
                    **{**options, "save_token_states": False},
                )
                checks["repeat_features"] = compare_features(features, repeated, **config["verification"])
                checks["repeat_inference"] = compare_inference(inference, repeated_inference, **config["verification"])
                del repeated
            verification_passed &= all(check["passed"] for check in checks.values())

            filename = f"{index:06d}.pt"
            torch.save({"case": case, **features, "inference": inference}, raw_dir / filename)
            record = {
                "index": index, "source_file": project_relative_path(source),
                "case": case, "feature_file": f"raw/{filename}",
                "image_spans": features["image_spans"], "query_index": features["query_index"],
                **inference, "correct": inference["prediction"] == case["gt"],
                "verification": checks,
            }
            manifest.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            manifest.flush()
            metadata["written_cases"] += 1
            write_json(metadata, destination / "run.json", allow_nan=False)
            del features

    if torch.cuda.is_available():
        metadata["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    metadata["verification_passed"] = verification_passed if verify_repeat or verify_reference else None
    metadata["status"] = "completed" if verification_passed else "verification_failed"
    write_json(metadata, destination / "run.json", allow_nan=False)
    print(f"저장: {destination}")
    if not verification_passed:
        raise RuntimeError("Feature verification failed; see manifest.jsonl")
    return destination


def main():
    parser = argparse.ArgumentParser(description="이미지별 Decoder Hidden State와 최종 입력 토큰 특징 추출")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases-jsonl", type=Path, nargs="+", help="기존 케이스 파일을 직접 지정")
    parser.add_argument("--category", help="지정한 클래스만 추출")
    parser.add_argument("--limit", type=int, help="전체 입력에서 추출할 최대 케이스 수")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-token-states", action="store_true", help="pooling 전 이미지 토큰 벡터도 저장")
    parser.add_argument("--verify-repeat", action="store_true", help="동일 케이스를 두 번 처리해 특징·응답 비교")
    parser.add_argument("--verify-reference", action="store_true", help="HF hidden_states 출력과 비교 (추가 메모리 사용)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다.")
    run_extraction(
        args.config, args.cases_jsonl, args.category, args.limit, args.output_dir,
        args.save_token_states, args.verify_repeat, args.verify_reference,
    )


if __name__ == "__main__":
    main()
