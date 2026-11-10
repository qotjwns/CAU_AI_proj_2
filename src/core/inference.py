"""Inference for one case, including first-generated-token logits."""

from src.core.prompts import build_messages, parse_prediction


def infer_case(case, model, processor, answer_token_ids):
    import torch

    inputs = processor.apply_chat_template(
        build_messages(case),
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
