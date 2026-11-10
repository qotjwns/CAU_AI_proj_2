import torch

from src.core.inference import generate_response, prepare_inputs, summarize_response


def map_image_tokens(inputs, model_config, merge_size, image_ids):
    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("Feature extraction expects one case at a time")
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    mask = inputs["attention_mask"][0].detach().cpu().bool().tolist()
    grids = inputs["image_grid_thw"].detach().cpu().tolist()
    if len(grids) != len(image_ids) or len(mask) != len(ids):
        raise ValueError("Image grids, image IDs and input mask do not match")

    spans = []
    position = 0
    while position < len(ids):
        if not mask[position] or ids[position] != model_config.image_token_id:
            position += 1
            continue
        start = position
        while position < len(ids) and mask[position] and ids[position] == model_config.image_token_id:
            position += 1
        end = position
        if (start == 0 or end == len(ids)
                or ids[start - 1] != model_config.vision_start_token_id
                or ids[end] != model_config.vision_end_token_id
                or not mask[start - 1] or not mask[end]):
            raise ValueError("Image tokens must be enclosed by vision start/end tokens")
        spans.append({"start": start, "end": end, "token_count": end - start})

    if len(spans) != len(image_ids):
        raise ValueError(f"Found {len(spans)} image spans for {len(image_ids)} images")
    for index, (span, grid, image_id) in enumerate(zip(spans, grids, image_ids)):
        t, h, w = grid
        if min(t, h, w, merge_size) <= 0 or h % merge_size or w % merge_size:
            raise ValueError(f"Invalid grid or spatial merge size for image {index}")
        expected = t * (h // merge_size) * (w // merge_size)
        if span["token_count"] != expected:
            raise ValueError(f"Image {index}: {span['token_count']} tokens, expected {expected}")
        span.update(image_index=index, image_id=image_id, grid_thw=grid)
    valid_positions = [index for index, valid in enumerate(mask) if valid]
    if not valid_positions:
        raise ValueError("The input has no valid tokens")
    return spans, valid_positions[-1]


def pool_hidden_state(hidden, spans, query_index, pooling_dtype="float32"):
    def mean(span):
        values = hidden[0, span["start"]:span["end"]]
        if pooling_dtype == "float32":
            values = values.float()
        return values.mean(dim=0).float()

    image_features = torch.stack([
        mean(span) for span in spans
    ]).detach().cpu()
    query_feature = hidden[0, query_index].detach().float().cpu().clone()
    return image_features, query_feature


def compare_tensors(left, right, atol, rtol):
    if left.shape != right.shape:
        return {"exact": False, "allclose": False, "max_abs_diff": None}
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    return {
        "exact": bool(torch.equal(left, right)),
        "allclose": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
        "max_abs_diff": (left.float() - right.float()).abs().max().item() if finite else None,
    }


def compare_features(left, right, atol=1e-5, rtol=1e-5):
    checks = {
        "same_layers": left["layers"] == right["layers"],
        "same_pooling": left["pooling"] == right["pooling"],
        "same_spans": left["image_spans"] == right["image_spans"],
        "same_query_index": left["query_index"] == right["query_index"],
        "same_input_ids": torch.equal(left["input_ids"], right["input_ids"]),
        "same_attention_mask": torch.equal(left["attention_mask"], right["attention_mask"]),
        "same_image_grid": torch.equal(left["image_grid_thw"], right["image_grid_thw"]),
    }
    tensors = {
        name: compare_tensors(left[name], right[name], atol, rtol)
        for name in ("image_features", "query_features")
    }
    return {
        **checks, **tensors,
        "passed": all(checks.values()) and all(value["allclose"] for value in tensors.values()),
    }


def extract_case_features(
    case, model, processor, answer_token_ids, layers=(9, 18, 27, 36),
    expected_hidden_size=2048, save_token_states=False, verify_reference=False,
    pooling_dtype="float32", atol=1e-5, rtol=1e-5,
):
    layers = list(layers)
    decoder = model.get_decoder()
    layer_count = len(decoder.layers)
    hidden_size = decoder.config.hidden_size
    if not layers or len(set(layers)) != len(layers) or any(layer < 1 or layer > layer_count for layer in layers):
        raise ValueError(f"Choose distinct decoder layers between 1 and {layer_count}")
    if hidden_size != expected_hidden_size:
        raise ValueError(f"Hidden dimension is {hidden_size}, expected {expected_hidden_size}")
    if pooling_dtype not in ("float32", "model"):
        raise ValueError("pooling_dtype must be float32 or model")
    if not (len(case["image_ids"]) == len(case["image_order"]) == len(case["image_labels"]) == 3):
        raise ValueError("Each case must contain three image IDs, filenames and labels")
    model.eval()
    inputs = prepare_inputs(case, model, processor)
    merge_size = processor.image_processor.merge_size
    spans, query_index = map_image_tokens(inputs, model.config, merge_size, case["image_ids"])
    sequence_length = inputs["input_ids"].shape[1]
    captured = {}
    token_states = {}
    handles = []

    def capture(layer):
        def hook(module, args, output):
            if layer in captured:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if tuple(hidden.shape) != (1, sequence_length, hidden_size):
                raise ValueError(f"Layer {layer} did not receive the full prefill sequence")
            captured[layer] = pool_hidden_state(hidden, spans, query_index, pooling_dtype)
            if save_token_states:
                token_states[layer] = [
                    hidden[0, span["start"]:span["end"]].detach().cpu().clone()
                    for span in spans
                ]
        return hook

    try:
        for layer in layers:
            module = decoder.norm if layer == layer_count else decoder.layers[layer - 1]
            handles.append(module.register_forward_hook(capture(layer)))
        generated = generate_response(inputs, model, output_hidden_states=verify_reference)
    finally:
        for handle in handles:
            handle.remove()

    if set(captured) != set(layers):
        raise ValueError("Not all requested layers were captured")
    features = {
        "layers": layers,
        "layer_convention": "huggingface_hidden_states",
        "final_layer_after_norm": layer_count,
        "pooling": f"mean_{pooling_dtype}",
        "image_spans": spans,
        "query_index": query_index,
        "input_ids": inputs["input_ids"].detach().cpu().clone(),
        "attention_mask": inputs["attention_mask"].detach().cpu().clone(),
        "image_grid_thw": inputs["image_grid_thw"].detach().cpu().clone(),
        "image_features": torch.stack([captured[layer][0] for layer in layers]),
        "query_features": torch.stack([captured[layer][1] for layer in layers]),
    }
    if not all(torch.isfinite(features[key]).all() for key in ("image_features", "query_features")):
        raise ValueError("Non-finite hidden-state features")
    if save_token_states:
        features["visual_token_states"] = token_states

    reference_check = None
    if verify_reference:
        prefill = generated.hidden_states[0]
        reference = [pool_hidden_state(prefill[layer], spans, query_index, pooling_dtype) for layer in layers]
        reference_check = {
            "image_features": compare_tensors(
                features["image_features"], torch.stack([row[0] for row in reference]), atol, rtol,
            ),
            "query_features": compare_tensors(
                features["query_features"], torch.stack([row[1] for row in reference]), atol, rtol,
            ),
        }
        reference_check["passed"] = all(value["allclose"] for value in reference_check.values())

    inference = summarize_response(inputs, generated, model, processor, answer_token_ids)
    return features, inference, reference_check
