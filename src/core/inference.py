from src.core.prompts import build_messages, parse_prediction


def prepare_inputs(case, model, processor):
    return processor.apply_chat_template(
        build_messages(case),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


def generate_response(inputs, model, output_hidden_states=False):
    import torch

    with torch.inference_mode():
        return model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=8,
            return_dict_in_generate=True,
            output_logits=True,
            output_hidden_states=output_hidden_states,
        )


def answer_variant_ids(tokenizer):
    """앞뒤 공백을 빼고 대소문자를 무시했을 때 yes / no가 되는 단일 토큰 ID (STEP 5-2 표기 변형 margin)."""
    variants = {"Yes": [], "No": []}
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id]).strip().casefold()
        if text in ("yes", "no"):
            variants[text.capitalize()].append(token_id)
    if not variants["Yes"] or not variants["No"]:
        raise ValueError("No Yes/No answer tokens found in the vocabulary")
    return variants


def variant_margin(logits, answer_variants):
    yes = logits[answer_variants["Yes"]].float().logsumexp(dim=0).item()
    no = logits[answer_variants["No"]].float().logsumexp(dim=0).item()
    return yes, no, yes - no


def summarize_response(inputs, generated, model, processor, answer_token_ids, answer_variants=None):
    input_length = inputs["input_ids"].shape[1]
    raw_response = processor.tokenizer.decode(
        generated.sequences[0, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    first_logits = generated.logits[0][0]
    yes_logit, no_logit = [first_logits[token_id].item() for token_id in answer_token_ids]
    image_tokens = (
        (inputs["input_ids"] == model.config.image_token_id)
        & inputs["attention_mask"].bool()
    )
    summary = {
        "prediction": parse_prediction(raw_response),
        "raw_response": raw_response,
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "logit_margin": yes_logit - no_logit,
        "input_token_count": inputs["attention_mask"].sum().item(),
        "image_token_count": image_tokens.sum().item(),
    }
    if answer_variants is not None:
        yes, no, margin = variant_margin(first_logits, answer_variants)
        summary.update(variant_yes_logsumexp=yes, variant_no_logsumexp=no, variant_logit_margin=margin)
    return summary


def infer_case(case, model, processor, answer_token_ids):
    inputs = prepare_inputs(case, model, processor)
    generated = generate_response(inputs, model)
    return summarize_response(inputs, generated, model, processor, answer_token_ids)
