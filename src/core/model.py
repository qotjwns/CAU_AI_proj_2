"""Load the model and single-token Yes/No answer IDs for baseline inference."""

from src.common.paths import PROJECT_ROOT


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
