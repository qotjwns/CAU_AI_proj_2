"""STEP 6-1 자체 시험 (CPU, 실제 3B 가중치를 쓰지 않음).

1. 가짜 attention: o_proj bias 있음, GQA, mrope 함수, mask 세 종류(없음 = 인과, bool, 가산). 독립 커널(sdpa)과 비교해
   분해 항등식을 확인하고, bias를 묶음마다 더하면 틀린다는 것도 확인한다. float64에서는 정확히 맞아야 한다.
2. 작은 무작위 Qwen2.5-VL: 같은 구조·같은 코드 경로이고 4층이다. 합성 이미지 3장, float32·bfloat16, sdpa.
   eager 구현으로는 다시 계산한 A를 모델 attention 가중치와 직접 비교한다.
3. 같은 작은 모델에 실제 입력 3개를 넣는다.
   - 입력: 실제 processor·COCO 이미지, 3·4·5장, 저장된 기준 실행과 같은 토큰 경계인지 확인
   - 시험: 재구성, 그리고 개입 위치 (층 1–2, λ = 0과 3)
4. 다섯 비교 조건과 P 기저의 수학 성질 (합성 데이터).

python -m experiments.step6_contribution.selftest
→ results/step6/contribution_validation/selftest.json
"""
import json
import tempfile
from pathlib import Path

import torch

from src.common.config import read_yaml
from src.common.paths import project_path
from src.core.causal_plan import load_plan
from src.core.feature_validation import require
from experiments.step6_contribution import directions as D
from experiments.step6_contribution.contribution import (
    AttentionRecorder, attention_modules, contributions, key_groups, question_positions, recompute_attention,
    reconstruct)
from experiments.step6_contribution.validation import (
    TOLERANCES, intervention, prepare_real, real_cases, reconstruction, stats)


MODEL_DIR = "cache/models/Qwen/Qwen2.5-VL-3B-Instruct"     # 설정·processor만 읽는다 (가중치는 쓰지 않음)
OUTPUT = Path("results/step6/contribution_validation/selftest.json")


# ---------------------------------------------------------------- 1. 가짜 attention

class FakeAttention(torch.nn.Module):
    """Qwen2.5-VL attention과 같은 속성 이름을 갖되 o_proj에 bias를 둔 작은 attention. 계산은 sdpa(독립 커널)로."""

    def __init__(self, dtype, hidden=48, heads=6, kv_heads=2, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.head_dim, self.num_heads = hidden // heads, heads
        self.num_key_value_groups, self.scaling, self.sliding_window = heads // kv_heads, (hidden // heads) ** -0.5, None
        self.q_proj = torch.nn.Linear(hidden, heads * self.head_dim, bias=True, dtype=dtype)
        self.k_proj = torch.nn.Linear(hidden, kv_heads * self.head_dim, bias=True, dtype=dtype)
        self.v_proj = torch.nn.Linear(hidden, kv_heads * self.head_dim, bias=True, dtype=dtype)
        self.o_proj = torch.nn.Linear(heads * self.head_dim, hidden, bias=True, dtype=dtype)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64).to(dtype) * 0.5)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, past_key_values=None):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb

        _, length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(1, length, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(1, length, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(1, length, -1, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=attention_mask is None, scale=self.scaling)
        return self.o_proj(out.transpose(1, 2).reshape(1, length, -1)), None


def fake_test():
    results = {}
    length, spans = 40, [{"start": 3, "end": 10}, {"start": 12, "end": 20}, {"start": 25, "end": 30}]
    queries = list(range(32, 40))
    generator = torch.Generator().manual_seed(3)
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    blocked = causal & (torch.rand(length, length, generator=generator) > 0.15)
    blocked |= torch.eye(length, dtype=torch.bool)                    # 모든 행에 허용 key가 하나 이상
    for dtype in (torch.float64, torch.float32):
        module = FakeAttention(dtype)
        hidden = torch.randn(1, length, 48, generator=generator, dtype=torch.float64).to(dtype)
        angles = torch.rand(length, module.head_dim // 2, generator=generator, dtype=torch.float64) * 6.0
        emb = torch.cat([angles, angles], dim=-1)[None]
        position = (emb.cos().to(dtype), emb.sin().to(dtype))
        additive = torch.where(causal, torch.zeros((), dtype=dtype), torch.tensor(torch.finfo(dtype).min, dtype=dtype))
        additive = additive + (torch.rand(length, length, generator=generator) > 0.8).to(dtype) * float(torch.tensor(4.0).log())
        masks = {"none_causal": None, "bool": blocked[None, None], "additive_with_boost": additive[None, None]}
        groups = key_groups(length, spans)
        for name, mask in masks.items():
            recorder = AttentionRecorder({0: module}, length)
            try:
                with torch.inference_mode():
                    module(hidden_states=hidden, attention_mask=mask, position_embeddings=position)
            finally:
                recorder.remove()
            with torch.inference_mode():
                result = contributions(module, recorder.records[0], queries, groups, compute_dtype=dtype)
                recon = reconstruct(result["parts"], result["bias"])
                kernel = recorder.records[0]["o"][0, queries]
                tolerance = (1e-12, 1e-12) if dtype == torch.float64 else TOLERANCES[torch.float32]["output"]
                wrong = torch.stack([part + result["bias"] for part in result["parts"].values()]).sum(dim=0)
            results[f"{str(dtype).split('.')[-1]}/{name}"] = {
                "output": stats(recon, kernel, tolerance),
                "identity": stats(recon, result["reference"], TOLERANCES["identity"]),
                "bias_added_per_group_error": float((wrong - kernel).abs().max()),
                "expected_error_(groups-1)*max|b|": float((len(groups) - 1) * result["bias"].abs().max()),
                "o_proj_bias_max_abs": float(result["bias"].abs().max())}
    passed = all(r["output"]["allclose"] and r["identity"]["allclose"] for r in results.values())
    return {"passed": passed, "cases": results}


# ---------------------------------------------------------------- 2·3. 작은 무작위 Qwen2.5-VL

def tiny_model(dtype, attn="sdpa", seed=0):
    """STEP 5-3 자체 시험과 같은 축소 설정(4층, hidden 64, head 4, KV head 2, 비전 2블록). 패치·병합 크기는 실제와 같다."""
    from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    directory = project_path(MODEL_DIR)
    config = Qwen2_5_VLConfig.from_pretrained(directory)
    text, vision = config.text_config, config.vision_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 64, 128, 4
    text.num_attention_heads, text.num_key_value_heads = 4, 2
    text.layer_types = ["full_attention"] * 4
    text.rope_parameters = {**text.rope_parameters, "mrope_section": [2, 3, 3]}
    vision.depth, vision.hidden_size, vision.num_heads, vision.intermediate_size = 2, 32, 2, 64
    vision.out_hidden_size, vision.fullatt_block_indexes = 64, [1]
    torch.manual_seed(seed)
    model = Qwen2_5_VLForConditionalGeneration._from_config(config, attn_implementation=attn, dtype=dtype).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2 and "embed" not in name:
                parameter.normal_(0, 0.2)
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    return model, processor


def synthetic_inputs(model, processor, directory):
    from PIL import Image
    from src.core.features import map_image_tokens
    from src.core.inference import prepare_inputs

    generator = torch.Generator().manual_seed(1)
    names = []
    for i, (w, h) in enumerate([(56, 56), (84, 56), (56, 112)]):
        array = (torch.rand(h, w, 3, generator=generator) * 255).to(torch.uint8).numpy()
        Image.fromarray(array).save(directory / f"img{i}.png")
        names.append(f"img{i}.png")
    case = {"case_id": "synthetic", "image_dir": str(directory), "image_order": names, "image_ids": [100, 101, 102],
            "image_labels": ["P", "N", "P"],
            "prompt": "Is the object category 'bed' present in all 3 images? Please answer with Yes or No."}
    inputs = prepare_inputs(case, model, processor)
    spans, _ = map_image_tokens(inputs, model.config, processor.image_processor.merge_size, case["image_ids"])
    return inputs, spans


def eager_probability_check(model, inputs, spans, layers):
    """eager 구현이 돌려주는 attention 가중치와 다시 계산한 A의 차이 (질문 토큰 행)."""
    length = inputs["input_ids"].shape[1]
    queries = question_positions(spans, length)
    modules = attention_modules(model, layers)
    weights, handles = {}, []
    for index, module in modules.items():
        handles.append(module.register_forward_hook(
            lambda m, a, o, index=index: weights.__setitem__(index, o[1].detach())))
    recorder = AttentionRecorder(modules, length)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        recorder.remove()
        for handle in handles:
            handle.remove()
    out = {}
    with torch.inference_mode():
        for index in layers:
            probs, _ = recompute_attention(modules[index], recorder.records[index], queries, torch.float32)
            eager = weights[index][:, :, queries, :].float()
            out[index] = float((probs - eager).abs().max())
    return out


def tiny_tests():
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for dtype in (torch.float32, torch.bfloat16):
            model, processor = tiny_model(dtype)
            inputs, spans = synthetic_inputs(model, processor, Path(tmp))
            results[f"synthetic/{str(dtype).split('.')[-1]}/sdpa"] = reconstruction(model, inputs, spans, [0, 1, 2, 3])
        model, processor = tiny_model(torch.float32, attn="eager")
        inputs, spans = synthetic_inputs(model, processor, Path(tmp))
        results["synthetic/float32/eager"] = reconstruction(model, inputs, spans, [0, 1, 2, 3])
        results["synthetic/float32/eager"]["eager_probability_max_abs_diff"] = eager_probability_check(
            model, inputs, spans, [0, 1, 2, 3])
    return results


def real_input_tests():
    plan = load_plan(read_yaml(project_path("configs/data/step5_causal_split.yaml")))
    items = real_cases(plan)
    results = {}
    for dtype in (torch.float32, torch.bfloat16):
        model, processor = tiny_model(dtype)
        (yes,) = processor.tokenizer.encode("Yes", add_special_tokens=False)
        (no,) = processor.tokenizer.encode("No", add_special_tokens=False)
        for item in items:
            inputs, spans, info = prepare_real(model, processor, plan, item)
            key = f"{str(dtype).split('.')[-1]}/{info['n_images']}img_{info['position']}"
            results[key] = {"input": info,
                            "reconstruction": reconstruction(model, inputs, spans, [0, 1, 2, 3]),
                            "intervention": intervention(model, inputs, spans, [1, 2], info["target_index"], 3.0,
                                                         (yes, no))}
    return results


# ---------------------------------------------------------------- 4. 조건·기저 수학

def direction_tests():
    generator = torch.Generator().manual_seed(7)
    d, rank, n = 64, 5, 400
    true = torch.linalg.qr(torch.randn(d, rank, generator=generator, dtype=torch.float64))[0]
    data = torch.randn(n, rank, generator=generator, dtype=torch.float64) * torch.tensor([5., 4., 3., 2., 1.]) @ true.T
    data = data + 0.01 * torch.randn(n, d, generator=generator, dtype=torch.float64)
    out = {}
    for rule in (("variance", 0.90), ("variance", 0.95), ("rank", 3)):
        for centered in (False, True):
            fit = D.fit_basis(data, rule, centered)
            D.check_orthonormal(fit["basis"])
            out[f"{rule[0]}={rule[1]}/{'centered' if centered else 'uncentered'}"] = {"k": fit["k"],
                                                                                    "explained": round(fit["explained"], 4)}
    fit = D.fit_basis(data, ("variance", 0.95), False)
    basis = fit["basis"]
    c_n = torch.randn(10, d, generator=generator, dtype=torch.float64)
    inside = c_n @ basis @ basis.T                                   # 학습한 기저의 span 안 벡터
    planted = D.fit_basis(data, ("rank", rank), False)["basis"]      # 심은 차원 그대로면 심은 부분공간을 복원해야 한다
    in_planted = c_n @ true @ true.T
    par, orth = D.parallel(c_n, basis), D.orthogonal(c_n, basis)
    rnd = D.condition_vector("random", c_n, generator=torch.Generator().manual_seed(1))
    mu_n, mu_p = torch.randn(d, generator=generator, dtype=torch.float64), torch.randn(d, generator=generator, dtype=torch.float64)
    matched = D.condition_vector("mean_difference", c_n, mu_n=mu_n, mu_p=mu_p, mean_scale="match")
    checks = {
        "variance_rules_choose_planted_rank_or_less": all(v["k"] <= rank for k, v in out.items() if k.startswith("variance")),
        "fixed_rank_exact": all(v["k"] == 3 for k, v in out.items() if k.startswith("rank")),
        "parallel_plus_orthogonal_equals_full": float((par + orth - c_n).abs().max()) < 1e-12,
        "orthogonal_is_orthogonal_to_basis": float((orth @ basis).abs().max()) < 1e-12,
        "parallel_keeps_vectors_in_basis_span": float((D.parallel(inside, basis) - inside).abs().max()) < 1e-12,
        "rank5_basis_recovers_planted_subspace": float((D.parallel(in_planted, planted) - in_planted).abs().max()
                                                       / in_planted.abs().max()) < 1e-2,
        "random_same_norm": float((rnd.norm(dim=-1) - c_n.norm(dim=-1)).abs().max()) < 1e-12,
        "random_reproducible": bool(torch.equal(rnd, D.condition_vector("random", c_n,
                                                                         generator=torch.Generator().manual_seed(1)))),
        "mean_difference_raw": bool(torch.equal(D.condition_vector("mean_difference", c_n, mu_n=mu_n, mu_p=mu_p)[0],
                                                mu_n - mu_p)),
        "mean_difference_matched_norm": float((matched.norm(dim=-1) - c_n.norm(dim=-1)).abs().max()) < 1e-12,
    }
    return {"passed": all(checks.values()), "checks": checks, "basis_rules": out}


def summary_line(name, result):
    o = result["overall"]
    return (f"  {name}: 출력 max {o['output']['max_abs']:.2e} 상대 {o['output']['relative']:.2e} "
            f"({'통과' if o['output']['allclose'] else '실패'}) | 재계산 attention max {o['attention']['max_abs']:.2e} | "
            f"항등식 max {o['identity']['max_abs']:.2e}")


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    report = {"fake": fake_test()}
    print(f"[가짜 attention] {'통과' if report['fake']['passed'] else '실패'}")
    for name, case in report["fake"]["cases"].items():
        print(f"  {name}: 출력 max {case['output']['max_abs']:.2e} 상대 {case['output']['relative']:.2e} | "
              f"bias를 묶음마다 더하면 max {case['bias_added_per_group_error']:.3f} (예상 {case['expected_error_(groups-1)*max|b|']:.3f})")
    report["tiny"] = tiny_tests()
    print("[작은 Qwen2.5-VL, 합성 이미지]")
    for name, result in report["tiny"].items():
        print(summary_line(name, result))
    print(f"  eager A 대 다시 계산한 A: {report['tiny']['synthetic/float32/eager']['eager_probability_max_abs_diff']}")
    report["real_inputs"] = real_input_tests()
    print("[작은 Qwen2.5-VL, 실제 입력 3개]")
    for name, result in report["real_inputs"].items():
        info, inter = result["input"], result["intervention"]
        print(summary_line(f"{name} (T={info['length']}, 경계 일치 {info['spans_match_stored_baseline']})",
                           result["reconstruction"]))
        print(f"    개입: λ=0 동일 {inter['lambda_zero']['logits_identical']}, λ=3 위치 검사 "
              f"{'통과' if inter['passed'] else '실패'}, 후속 층 변화 {inter['lambda_scale']['downstream_layers_changed_(expected)']}")
    report["directions"] = direction_tests()
    print(f"[조건·기저 수학] {'통과' if report['directions']['passed'] else '실패'} {report['directions']['basis_rules']}")
    passed = (report["fake"]["passed"] and all(r["passed"] for r in report["tiny"].values())
              and all(r["reconstruction"]["passed"] and r["intervention"]["passed"] and r["input"]["spans_match_stored_baseline"]
                      and r["input"]["suffix_start_matches_plan"] for r in report["real_inputs"].values())
              and report["directions"]["passed"])
    report["passed"] = passed
    path = project_path(OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"저장: {OUTPUT} | 전체 {'통과' if passed else '실패'}")
    require(passed, "STEP 6-1 self-test failed (see the JSON for the failing item)")


if __name__ == "__main__":
    main()
