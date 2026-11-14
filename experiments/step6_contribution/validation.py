"""STEP 6-1 검증: 재구성 오차와 개입 위치. 작은 무작위 모델과 실제 3B가 같은 함수를 쓴다.

허용 오차는 실행 전에 여기서 고정했다. 실패하면 허용 오차를 늘리지 말고 원인을 추적한다.
torch.allclose 기준: |a − b| ≤ atol + rtol·|b| (b = 기준 값).
- identity: Σ c_g + b_O 대 분해하지 않은 참조. 같은 A·v·W_O로 계산하고 compute dtype은 float32다.
  분해(묶음 나누기) 자체의 정확성을 본다.
- attention: 다시 계산한 o_proj 입력 z 대 모델 커널(sdpa 등)이 o_proj에 넣은 값. A·v 재계산이 모델과 같은지 본다.
- output: Σ c_g + b_O 대 모델의 attention 출력 o. 전체 재구성 오차다.
- o_proj: W_O·(모델 z) + b 대 모델 o. o_proj 자체의 반올림 크기를 본다.
float32 모델은 반올림 수준의 오차만 허용한다. bfloat16 모델은 bf16 반올림(상대 2⁻⁸)을 몇 번 거치므로 상대·절대 2⁻⁶을 둔다.
"""
import torch

from src.core.feature_validation import require
from experiments.step6_contribution.contribution import (
    AttentionRecorder, OutputAddition, OutputProbe, attention_modules, contributions, key_groups, question_positions,
    reconstruct)


TOLERANCES = {
    "identity": (1e-5, 1e-5),
    torch.float32: {"attention": (1e-4, 1e-5), "output": (1e-4, 1e-5)},
    torch.bfloat16: {"attention": (2 ** -6, 2 ** -6), "output": (2 ** -6, 2 ** -6)},
}
# 재구성 판정 (STEP 6-2 고정본 v3, 사용자 승인 2026-11-14, 실제 3B fp32 검증 실패를 본 뒤):
# - 항등식: 원소별 allclose (위 identity, float64 재계산) 그대로
# - attention·output: 층마다 상대 오차 ‖추정 − 모델 커널‖ / ‖모델 커널‖ ≤ RELATIVE_MAX[모델 dtype]
#   원소별 allclose는 0 근처 원소에서 모델 커널 자체의 반올림(fp32 상대 2e-6)만으로 절대 허용 오차를 넘을 수 있어
#   판정에서 뺐다 (값은 계속 보고). 마스크·rope·scaling 불일치 같은 실제 오류는 상대 10% 이상이다.
RELATIVE_MAX = {torch.float32: 1e-5, torch.bfloat16: 2 ** -6}
RULE = ("identity: elementwise allclose (rtol = atol = 1e-5, float64 recompute); attention and output: per-layer "
        "relative Frobenius error <= 1e-5 (float32 model) or 2^-6 (bfloat16 model) [frozen v3]")


def reconstruction_passes(per_layer, dtype):
    """per_layer: {층: {"identity": stats, "attention": stats, "output": stats, ...}} → 판정 (RULE)."""
    limit = RELATIVE_MAX[dtype]
    return all(checks["identity"]["allclose"] and checks["attention"]["relative"] <= limit
               and checks["output"]["relative"] <= limit for checks in per_layer.values())


def model_dtype(model):
    return next(model.get_decoder().parameters()).dtype


def stats(estimate, reference, tolerance):
    rtol, atol = tolerance
    a, b = estimate.double(), reference.double()
    diff = (a - b).abs()
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "rmse": float(diff.pow(2).mean().sqrt()),
            "relative": float(diff.norm() / b.norm().clamp_min(1e-300)), "reference_rms": float(b.pow(2).mean().sqrt()),
            "allclose": bool(torch.allclose(a, b, rtol=rtol, atol=atol)), "rtol": rtol, "atol": atol,
            "worst_excess": float((diff - (atol + rtol * b.abs())).max())}


def reconstruction(model, inputs, spans, layers, compute_dtype=torch.float32):
    """무개입 forward 한 번(batch 1, use_cache=False)에서 지정 층·질문 토큰의 재구성 오차."""
    dtype = model_dtype(model)
    length = inputs["input_ids"].shape[1]
    queries = question_positions(spans, length)
    modules = attention_modules(model, layers)
    groups = key_groups(length, spans)
    recorder = AttentionRecorder(modules, length)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        recorder.remove()
    per_layer, collected = {}, {name: [] for name in ("identity", "attention", "output", "o_proj")}
    token_max = torch.zeros(len(queries), dtype=torch.float64)
    with torch.inference_mode():
        for layer in layers:
            record, module = recorder.records[layer], modules[layer]
            require("o" in record, f"Layer {layer}: the attention output was not recorded")
            result = contributions(module, record, queries, groups, compute_dtype)
            recon = reconstruct(result["parts"], result["bias"])
            kernel_z = record["z"][0, queries].to(compute_dtype)
            kernel_o = record["o"][0, queries].to(compute_dtype)
            weight = module.o_proj.weight.detach().to(compute_dtype)
            o_proj_only = kernel_z @ weight.T + (result["bias"] if result["bias"] is not None else 0)
            pairs = {"identity": (recon, result["reference"], TOLERANCES["identity"]),
                     "attention": (result["z_all"], kernel_z, TOLERANCES[dtype]["attention"]),
                     "output": (recon, kernel_o, TOLERANCES[dtype]["output"]),
                     "o_proj": (o_proj_only, kernel_o, TOLERANCES[dtype]["output"])}
            per_layer[layer] = {name: stats(*pair) for name, pair in pairs.items()}
            for name, (a, b, _) in pairs.items():
                collected[name].append((a, b))
            token_max = torch.maximum(token_max, (recon.double() - kernel_o.double()).abs().amax(dim=1).cpu())
            per_layer[layer]["row_sum_max_error"] = float((result["row_sums"].double() - 1).abs().max())
            per_layer[layer]["attention_share"] = {k: round(v, 6) for k, v in result["share"].items()}
            per_layer[layer]["bias"] = None if result["bias"] is None else float(result["bias"].abs().max())
    overall = {}
    for name, pairs in collected.items():
        tolerance = TOLERANCES["identity"] if name == "identity" else TOLERANCES[dtype][
            "attention" if name == "attention" else "output"]
        overall[name] = stats(torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs]), tolerance)
    return {"dtype": str(dtype), "compute_dtype": str(compute_dtype), "length": length, "queries": queries,
            "layers": list(layers), "groups": [(name, len(keys)) for name, keys in groups],
            "overall": overall, "per_layer": per_layer,
            "per_token_output_max_abs": {q: float(v) for q, v in zip(queries, token_max)},
            "rule": RULE, "passed": reconstruction_passes(per_layer, dtype)}


def _forward(model, inputs):
    with torch.inference_mode():
        return model(**inputs, use_cache=False).logits[0].float().cpu()


def intervention(model, inputs, spans, layers, target_image, scale, answer_ids=None):
    """별도 무개입 실행에서 c_N을 뽑고, 같은 입력에 o' = o + λ c_N (층 layers × 질문 토큰)을 더해 위치를 검사한다."""
    length = inputs["input_ids"].shape[1]
    queries = question_positions(spans, length)
    n_layers = len(model.get_decoder().layers)
    every = attention_modules(model, range(n_layers))
    targets = {layer: every[layer] for layer in layers}
    groups = key_groups(length, spans)

    # 1. 무개입 실행: 기준 출력(모든 층)과 c_N
    probe, recorder = OutputProbe(every, length), AttentionRecorder(targets, length)
    try:
        baseline_logits = _forward(model, inputs)
    finally:
        probe.remove()
        recorder.remove()
    baseline = probe.pre
    with torch.inference_mode():
        vectors = {layer: contributions(targets[layer], recorder.records[layer], queries, groups)["parts"][
            f"image_{target_image}"].float() for layer in layers}

    def run(lam):
        probe = OutputProbe(every, length)
        addition = OutputAddition(targets, vectors, queries, lam, length)
        probe.attach_post()
        try:
            logits = _forward(model, inputs)
        finally:
            addition.remove()
            probe.remove()
        return probe.pre, probe.post, logits

    rows = torch.as_tensor(queries)
    other = torch.ones(length, dtype=torch.bool)
    other[rows] = False
    out = {"layers": list(layers), "queries": [queries[0], queries[-1]], "target_image": target_image, "scale": scale}

    # 2. λ = 0: 모델 출력이 그대로여야 한다
    pre, post, logits = run(0.0)
    out["lambda_zero"] = {
        "logits_identical": bool(torch.equal(logits, baseline_logits)),
        "logits_max_abs_diff": float((logits - baseline_logits).abs().max()),
        "attention_outputs_identical_all_layers": all(torch.equal(post[i], pre[i]) and torch.equal(pre[i], baseline[i])
                                                       for i in range(n_layers))}
    # 3. λ = scale
    pre, post, logits = run(scale)
    first = min(layers)
    target_exact, target_rounding, untouched_tokens = True, 0.0, True
    for layer in layers:
        expected = (pre[layer][:, rows, :].float() + scale * vectors[layer].to(pre[layer].device)).to(pre[layer].dtype)
        target_exact &= bool(torch.equal(post[layer][:, rows, :], expected))
        added = post[layer][:, rows, :].float() - pre[layer][:, rows, :].float()
        target_rounding = max(target_rounding, float((added - scale * vectors[layer].to(added.device)).abs().max()))
        untouched_tokens &= bool(torch.equal(post[layer][:, other, :], pre[layer][:, other, :]))
    other_layers = [i for i in range(n_layers) if i not in layers]
    downstream = [i for i in range(n_layers) if i > first and not torch.equal(pre[i], baseline[i])]
    out["lambda_scale"] = {
        "target_rows_equal_pre_plus_lambda_v": target_exact,
        "target_rows_max_abs(added - lambda_v)": target_rounding,
        "non_question_tokens_unmodified_in_target_layers": untouched_tokens,
        "non_target_layers_unmodified": all(torch.equal(post[i], pre[i]) for i in other_layers),
        "layers_before_first_target_unchanged_vs_baseline": all(torch.equal(pre[i], baseline[i]) for i in range(first)),
        "first_target_layer_pre_unchanged_vs_baseline": bool(torch.equal(pre[first], baseline[first])),
        "tokens_before_question_unchanged_all_layers": all(
            torch.equal(pre[i][:, :queries[0], :], baseline[i][:, :queries[0], :]) for i in range(n_layers)),
        "downstream_layers_changed_(expected)": downstream,
        "vector_norm_mean": {layer: float(vectors[layer].norm(dim=-1).mean()) for layer in layers},
        "logits_max_abs_change": float((logits - baseline_logits).abs().max()),
    }
    if answer_ids is not None:
        yes, no = answer_ids
        margin = lambda x: float(x[-1, yes] - x[-1, no])
        out["lambda_scale"]["yes_no_margin"] = {"baseline": margin(baseline_logits), "intervened": margin(logits)}
    checks = [out["lambda_zero"]["logits_identical"], out["lambda_zero"]["attention_outputs_identical_all_layers"],
              *[out["lambda_scale"][k] for k in ("target_rows_equal_pre_plus_lambda_v",
                                                 "non_question_tokens_unmodified_in_target_layers",
                                                 "non_target_layers_unmodified",
                                                 "layers_before_first_target_unchanged_vs_baseline",
                                                 "first_target_layer_pre_unchanged_vs_baseline",
                                                 "tokens_before_question_unchanged_all_layers")]]
    out["passed"] = all(checks)
    return out


def real_cases(plan, count=3):
    """검증용 실제 입력 (규칙 고정): 실험 B 선택 절반 항목에서 (3장·middle), (4장·first), (5장·last)를 item_id 순 첫 번째로."""
    wanted = [(3, "middle"), (4, "first"), (5, "last")][:count]
    chosen = []
    for n_images, position in wanted:
        item = min((i for i in plan["b_items"] if i["half"] == "selection" and i["population"] in ("block", "boost")
                    and i["n_images"] == n_images and i["position"] == position), key=lambda i: i["item_id"])
        chosen.append(item)
    return chosen


def prepare_real(model, processor, plan, item):
    from src.core.features import map_image_tokens
    from src.core.inference import prepare_inputs

    entry = plan["inputs"][item["input_id"]]
    case = entry["case"]
    inputs = prepare_inputs(case, model, processor)
    spans, query_index = map_image_tokens(inputs, model.config, processor.image_processor.merge_size, case["image_ids"])
    keys = ("start", "end", "token_count")
    same = ([{k: s[k] for k in keys} for s in spans] == [{k: s[k] for k in keys} for s in entry["baseline"]["image_spans"]]
            and query_index == entry["baseline"]["query_index"])
    require(query_index == inputs["input_ids"].shape[1] - 1, "Padded input")
    return inputs, spans, {"input_id": item["input_id"], "n_images": case["n_images"], "position": item["position"],
                           "target_index": item["target_index"], "length": int(inputs["input_ids"].shape[1]),
                           "spans_match_stored_baseline": same, "suffix_start_matches_plan":
                               spans[-1]["end"] + 1 == item["suffix_start"]}
