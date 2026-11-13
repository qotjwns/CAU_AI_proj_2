"""STEP 5-3 자체 시험 (CPU 가능, 실제 데이터·실제 모델 출력 없음).

--model: 무작위 가중치의 작은 Qwen2.5-VL(같은 구조, 4층)과 합성 이미지로 개입 경로를 시험한다.
  빠른 경로(prefix 1회 + suffix batch) = 참조 경로(model.generate 전체 prefill + 같은 개입), ℓ = 0 이식 = 무개입,
  ℓ = 마지막 이식 = donor margin, 크기 0 개입 = 무개입, 강화는 대상 attention 승산비를 정확히 4배로, 차단은 대상
  attention을 0으로 만드는지 확인한다. float32(정확성)와 bfloat16(실행 가능성) 두 번 돈다.
--eval: 계획 파일 구조로 가짜 결과(효과 0, 효과를 심은 경우)를 만들어 검사·평가 코드가 기대한 판정을 내는지 시험한다.
서버에서는 GPU 실행 전에 --model로 설치된 라이브러리에서 개입 경로가 맞는지 확인한다.
"""
import argparse
import tempfile
from pathlib import Path

import torch

from src.common.paths import project_path
from src.core.causal_runtime import Runner, reference_attention_mask
from src.core.feature_validation import require
from src.core.inference import answer_variant_ids


MODEL_DIR = "cache/models/Qwen/Qwen2.5-VL-3B-Instruct"


# ---------------------------------------------------------------- 작은 모델

def tiny_model(dtype, seed=0):
    from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

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
    model = Qwen2_5_VLForConditionalGeneration._from_config(config, attn_implementation="sdpa", dtype=dtype).eval()
    with torch.no_grad():                         # 무작위 초기화를 키워 조건 사이 차이가 뚜렷하게
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2 and "embed" not in name:
                parameter.normal_(0, 0.2)
    model.generation_config = GenerationConfig.from_pretrained(directory)
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    (yes_id,) = processor.tokenizer.encode("Yes", add_special_tokens=False)
    (no_id,) = processor.tokenizer.encode("No", add_special_tokens=False)
    return model, processor, (yes_id, no_id)


def synthetic_cases(directory):
    from PIL import Image

    generator = torch.Generator().manual_seed(1)
    sizes = [(56, 56), (84, 56), (56, 112), (112, 84)]
    names = []
    for i, (w, h) in enumerate(sizes):
        array = (torch.rand(h, w, 3, generator=generator) * 255).to(torch.uint8).numpy()
        name = f"img{i}.png"
        Image.fromarray(array).save(directory / name)
        names.append(name)
    prompt = "Is the object category 'bed' present in all 3 images? Please answer with Yes or No."

    def case(order, case_id):
        return {"case_id": case_id, "image_dir": str(directory), "image_order": [names[i] for i in order],
                "image_ids": [100 + i for i in order], "image_labels": ["N" if i == 0 else "P" for i in order],
                "prompt": prompt}
    return case([0, 1, 2], "first"), case([1, 2, 0], "last"), case([3, 1, 2], "other")


def close(left, right, tolerance):
    return (abs(left["margin"] - right["margin"]) <= tolerance and abs(left["margin_fp32"] - right["margin_fp32"])
            <= tolerance and abs(left["margin_bf16"] - right["margin_bf16"]) <= max(tolerance, 0.0)
            and left["raw"] == right["raw"])


def model_test(dtype):
    exact = dtype == torch.float32
    tolerance = 2e-4 if exact else 0.25
    identity = 1e-5 if exact else 0.25        # 같은 batch 안 동일 입력 행: CPU 커널은 행 위치에 따라 마지막 자리가 다를 수 있음
    model, processor, answer_ids = tiny_model(dtype)
    runner = Runner(model, processor, answer_ids, answer_variant_ids(processor.tokenizer))
    results = {}
    with tempfile.TemporaryDirectory() as folder:
        first, last, other = synthetic_cases(Path(folder))
        prepared = {c["case_id"]: runner.prepare(c) for c in (first, last, other)}
        n_layers = runner.n_layers

        # 1. 무개입: 빠른 경로(A형, B형) = 참조
        stacks, none = {}, {}
        for name, item in prepared.items():
            rows, stack = runner.run_suffix(item, runner.prefix(item, item["length"] - 1), 1, capture=True)
            stacks[name], none[name] = stack[0], rows[0]
            reference = runner.reference(item)
            require(close(rows[0], reference, tolerance), f"A-type no-intervention differs from reference: {name}")
            start = item["spans"][-1]["end"] + 1
            empty = {"layers": torch.zeros(1, n_layers, dtype=torch.bool),
                     "select": torch.zeros(1, item["length"] - start, item["length"], dtype=torch.bool),
                     "value": torch.zeros(1)}
            rows_b, _ = runner.run_suffix(item, runner.prefix(item, start), 1, attention=empty)
            require(close(rows_b[0], reference, tolerance), f"B-type no-intervention differs from reference: {name}")
        results["no_intervention_matches_reference"] = True

        # 2. 실험 A 이식: 같은 묶음 donor(last → first), 다른 묶음 donor(other → first)
        item = prepared["first"]
        layers = [-1] + list(range(n_layers + 1)) * 2
        donors = [stacks["first"]] + [stacks["last"]] * (n_layers + 1) + [stacks["other"]] * (n_layers + 1)
        rows, _ = runner.run_suffix(item, runner.prefix(item, item["length"] - 1), len(layers),
                                    patch={"layer": torch.tensor(layers), "donor": torch.stack(donors)})
        gaps = [abs(rows[i]["margin"] - rows[0]["margin"]) for i in (1, 2 + n_layers)]
        require(max(gaps) <= identity and rows[1]["raw"] == rows[0]["raw"], f"Patch at l = 0 is not an identity: {gaps}")
        results["l0_max_gap"] = max(gaps)
        for donor, row in (("last", rows[1 + n_layers]), ("other", rows[2 + 2 * n_layers])):
            require(abs(row["margin"] - none[donor]["margin"]) <= (1e-9 if exact else 1e-6),
                    f"Patch at l = {n_layers} does not reproduce the donor margin ({donor})")
        changed = sum(abs(r["margin"] - rows[0]["margin"]) > 1e-3 for r in rows[1:])
        require(changed >= n_layers, "Patching barely changes the output; the test model is not informative")
        for layer, donor, row in zip(layers[1:], donors[1:], rows[1:]):
            reference = runner.reference(item, layer, donor)
            require(close(row, reference, tolerance), f"Patch fast path differs from reference at l = {layer}")
        results["patch_matches_reference"] = True

        # 3. 실험 B attention: 차단·강화·크기 0, 두 query 범위, 두 구간
        start = item["spans"][-1]["end"] + 1
        suffix, length = item["length"] - start, item["length"]
        specs = [None]
        for kind, value in (("block", float(runner.block_value)), ("boost", float(runner.boost_value)), ("zero", 0.0)):
            for query in ("post_image", "last_token"):
                for band in ((1, 2), (3, 4)):
                    for target in (0, 1):
                        specs.append((kind, value, query, band, target))
        layer_mask = torch.zeros(len(specs), n_layers, dtype=torch.bool)
        select = torch.zeros(len(specs), suffix, length, dtype=torch.bool)
        values = torch.zeros(len(specs))
        for b, spec in enumerate(specs[1:], start=1):
            kind, value, query, band, target = spec
            layer_mask[b, band[0] - 1:band[1]] = True
            span = item["spans"][target]
            for q in (range(suffix) if query == "post_image" else [suffix - 1]):
                select[b, q, span["start"]:span["end"]] = True
            values[b] = value
        rows, _ = runner.run_suffix(item, runner.prefix(item, start), len(specs),
                                    attention={"layers": layer_mask, "select": select, "value": values})
        for b, spec in enumerate(specs[1:], start=1):
            kind, value, query, band, target = spec
            span = item["spans"][target]
            queries = [start + q for q in (range(suffix) if query == "post_image" else [suffix - 1])]
            reference = runner.reference(item, attention={"layers": list(range(band[0] - 1, band[1])),
                                                          "queries": queries, "targets": [(span["start"], span["end"])],
                                                          "value": value})
            require(close(rows[b], reference, tolerance), f"Attention fast path differs from reference: {spec}")
            if kind == "zero":
                gap = abs(rows[b]["margin"] - rows[0]["margin"])
                results["zero_max_gap"] = max(results.get("zero_max_gap", 0.0), gap)
                require(gap <= identity and rows[b]["raw"] == rows[0]["raw"],
                        f"Zero-size intervention changed the output: {gap}")
        moved = sum(abs(r["margin"] - rows[0]["margin"]) > 1e-4 for r, s in zip(rows[1:], specs[1:]) if s[0] != "zero")
        require(moved >= 8, "Block/boost barely change the output; the test model is not informative")
        results["attention_matches_reference"] = True

        # 4. attention 확률 직접 확인 (첫 층, float32에서만 정확히)
        if exact:
            results["attention_probabilities"] = attention_probability_test(runner, item, start)
    return results


def attention_probability_test(runner, item, start):
    """첫 층 sdpa 입력을 가로채 확률을 계산: 강화 행의 대상 승산비 = 무개입의 4배, 차단 행의 대상 확률 = 0."""
    import torch.nn.functional as functional

    n_layers = runner.n_layers
    suffix, length = item["length"] - start, item["length"]
    span = item["spans"][1]
    layer_mask = torch.zeros(3, n_layers, dtype=torch.bool)
    layer_mask[1:, 0] = True
    select = torch.zeros(3, suffix, length, dtype=torch.bool)
    select[1:, :, span["start"]:span["end"]] = True
    values = torch.tensor([0.0, float(runner.boost_value), float(runner.block_value)])
    captured = []
    original = functional.scaled_dot_product_attention

    def spy(query, key, value, attn_mask=None, **kwargs):
        if not captured and attn_mask is not None:
            captured.append((query.detach().clone(), key.detach().clone(), attn_mask.detach().clone(),
                             kwargs.get("scale")))
        return original(query, key, value, attn_mask=attn_mask, **kwargs)

    prefix = runner.prefix(item, start)
    functional.scaled_dot_product_attention = spy
    try:
        runner.run_suffix(item, prefix, 3,
                          attention={"layers": layer_mask, "select": select, "value": values})
    finally:
        functional.scaled_dot_product_attention = original
    query, key, mask, scale = captured[0]
    if key.shape[1] != query.shape[1]:
        key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    probabilities = torch.softmax(query.double() @ key.double().transpose(-1, -2) * scale + mask.double(), dim=-1)
    target = probabilities[..., span["start"]:span["end"]].sum(-1)            # [B, heads, S]
    odds = target / (1 - target)
    ratio = (odds[1] / odds[0])
    require(torch.allclose(ratio, torch.full_like(ratio, 4.0), rtol=1e-5), f"Boost odds ratio is not 4: {ratio}")
    require(float(target[2].max()) == 0.0, "Block left attention on the target")
    reference = reference_attention_mask(length, list(range(start, length)), [(span["start"], span["end"])],
                                         float(runner.boost_value), mask.dtype, mask.device)[0, 0, start:]
    require(torch.equal(mask[1, 0], reference), "Fast-path mask differs from the independently built reference mask")
    return {"boost_odds_ratio": [float(ratio.min()), float(ratio.max())], "block_target_max": float(target[2].max())}


# ---------------------------------------------------------------- 실행 흐름 전체 (run_a, run_b → 검사)

def pipeline_test(dtype=torch.float32):
    """작은 계획(합성 이미지 5장, 계열 2개)으로 run.py의 실행 흐름과 검사(check)를 연습 실행 단계 그대로 돈다."""
    from PIL import Image

    from experiments.step5_causal.run import run_a, run_b
    from src.common.data import write_json
    from src.core.causal_check import check_a, check_b, load_run
    from src.core.causal_conditions import a_work, b_work
    from src.core.causal_plan import control_p

    model, processor, answer_ids = tiny_model(dtype)
    runner = Runner(model, processor, answer_ids, answer_variant_ids(processor.tokenizer))
    verification = {"answer_agreement_min": 0.99, "near_tie_bf16_margin": 0.125, "margin_abs_tolerance": 0.25,
                    "margin_within_share_min": 0.99, "final_layer_donor_tolerance": 0.01}
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        generator = torch.Generator().manual_seed(3)
        sizes = [(56, 56), (84, 56), (56, 84), (112, 56), (56, 56)]
        for i, (w, h) in enumerate(sizes):
            array = (torch.rand(h, w, 3, generator=generator) * 255).to(torch.uint8).numpy()
            Image.fromarray(array).save(folder / f"img{i}.png")
        prompt = "Is the object category 'bed' present in all 3 images? Please answer with Yes or No."
        inputs = {}

        def add(input_id, order, labels, series, anchor, a, condition="one_negative", placement=None):
            case = {"case_id": input_id, "image_dir": str(folder), "image_order": [f"img{i}.png" for i in order],
                    "image_ids": [500 + i for i in order], "image_labels": labels, "prompt": prompt, "n_images": 3,
                    "n_before": a, "pair_id": series, "anchor_negative_id": anchor, "class": "bed",
                    "strength": "High", "condition": condition, "placement": placement}
            prepared = runner.prepare(case)
            reference = runner.reference(prepared)
            baseline = {"prediction": reference["answer"], "fp32_variant_logit_margin": reference["margin"],
                        "logit_margin": reference["margin_bf16"], "query_index": prepared["length"] - 1,
                        "image_spans": [{**s, "image_index": i, "image_id": case["image_ids"][i]}
                                        for i, s in enumerate(prepared["spans"])]}
            inputs[input_id] = {"input_id": input_id, "case": case, "baseline": baseline, "half": "selection"}

        for series, anchor, n_image in (("s1", 1, 0), ("s2", 2, 3)):
            orders = {0: [n_image, 1, 2], 1: [1, n_image, 2], 2: [1, 2, n_image]}
            for a, order in orders.items():
                add(f"{series}a{a}", order, ["N" if i == n_image else "P" for i in order], series, anchor, a)
        add("allp_first", [4, 1, 2], ["P"] * 3, "s1", 1, None, "all_positive", "N-first")
        add("allp_last", [1, 2, 4], ["P"] * 3, "s1", 1, None, "all_positive", "N-last")
        units = []
        for a, position in ((0, "first"), (1, "middle")):
            units.append({"unit_id": f"A::s1::n3::a{a}", "series": "s1", "anchor": 1, "class": "bed", "strength": "High",
                          "n_images": 3, "a": a, "position": position, "half": "selection",
                          "false_yes_input": f"s1a{a}", "correct_no_input": "s1a2",
                          "restore": {"recipient": f"s1a{a}", "same_donor": "s1a2", "c1_donor": "s2a2"},
                          "contaminate": {"recipient": "s1a2", "same_donor": f"s1a{a}", "c1_donor": f"s2a{a}"}})
        pairs = [{"pair_id": "A-AllP::s1::n3", "series": "s1", "anchor": 1, "class": "bed", "strength": "High",
                  "n_images": 3, "half": "selection", "inputs": {"N-first": "allp_first", "N-last": "allp_last"}}]
        items = []
        for input_id, entry in inputs.items():
            case, spans = entry["case"], entry["baseline"]["image_spans"]
            one = case["condition"] == "one_negative"
            target = case["n_before"] if one else (0 if case["placement"] == "N-first" else 2)
            items.append({"item_id": f"B::{input_id}", "input_id": input_id, "kind": case["condition"],
                          "population": ("block" if case["n_before"] % 2 == 0 else "boost") if one else "all_positive",
                          "series": case["pair_id"], "anchor": case["anchor_negative_id"], "class": "bed",
                          "strength": "High", "n_images": 3, "target_index": target,
                          "position": ("first", "middle", "last")[target], "half": "selection",
                          "control_index": control_p(spans, case["image_labels"], target),
                          "suffix_start": spans[-1]["end"] + 1, "last_index": entry["baseline"]["query_index"]})
        plan = {"inputs": inputs, "a_units": units, "a_allp": pairs, "b_items": items}
        outcome = {}
        for experiment in ("A", "B"):
            output_dir = folder / experiment
            output_dir.mkdir()
            if experiment == "A":
                work = a_work(plan, "pilot", runner.n_layers)
                run_a(runner, plan, work, output_dir, {})
            else:
                work = b_work(plan, "pilot")
                run_b(runner, plan, work, output_dir, {})
            write_json({"status": "completed", "n_layers": runner.n_layers}, output_dir / "run.json")
            run = load_run(output_dir)
            checks = (check_a("pilot", plan, run, verification, runner.n_layers) if experiment == "A"
                      else check_b("pilot", plan, run, verification))
            failed = [name for name, check in checks.items() if not check["passed"]]
            require(not failed, f"Pipeline {experiment}: integrity checks failed {failed}: "
                                f"{ {k: checks[k]['detail'] for k in failed} }")
            outcome[experiment] = {"batches": len(run["batches"]), "reference": len(run["reference"]),
                                   "checks_passed": len(checks)}
    return outcome


def main():
    parser = argparse.ArgumentParser(description="STEP 5-3 자체 시험 (가짜 모델·가짜 데이터)")
    parser.add_argument("--model", action="store_true", help="작은 무작위 모델로 개입 경로 시험")
    parser.add_argument("--eval", action="store_true", help="가짜 결과로 검사·평가 시험")
    args = parser.parse_args()
    if not (args.model or args.eval):
        args.model = args.eval = True
    if args.model:
        for dtype in (torch.float32, torch.bfloat16):
            print(f"[model {dtype}]", model_test(dtype))
        print("[pipeline float32]", pipeline_test(torch.float32))
    if args.eval:
        from experiments.step5_causal.selftest_eval import eval_test
        for name, outcome in eval_test().items():
            print(f"[eval] {name}: {outcome}")
    print("자체 시험 통과")


if __name__ == "__main__":
    main()
