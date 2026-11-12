from src.common.paths import PROJECT_ROOT


def load_model(model_name, vision_dtype=None):
    """vision_dtype='float32'이면 비전 인코더만 float32로 둔다 (STEP 5-2 민감도 분석). 기본값은 STEP 1~4와 같다."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_dir = PROJECT_ROOT / "cache" / "models" / model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    (yes_id,) = processor.tokenizer.encode("Yes", add_special_tokens=False)
    (no_id,) = processor.tokenizer.encode("No", add_special_tokens=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, local_files_only=True,
    ).to(device).eval()
    if vision_dtype is not None:
        if vision_dtype != "float32":
            raise ValueError("vision_dtype must be None or 'float32'")
        model.model.visual.to(torch.float32)
    return model, processor, (yes_id, no_id)
