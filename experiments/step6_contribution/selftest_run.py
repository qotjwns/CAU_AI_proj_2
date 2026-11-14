"""STEP 6-2 실행 경로 자체 시험 (CPU, 실제 3B 가중치를 쓰지 않음). 서버에서도 GPU 단계 전에 --model로 돌린다.

--model: 무작위 가중치의 작은 Qwen2.5-VL(4층, STEP 5-3 자체 시험과 같은 축소 설정)과 합성 이미지.
  1. 기여 추출 재구성 검사 통과, BasisAccumulator·fit_bases = directions.fit_basis(SVD)
  2. 빠른 경로 개입(SuffixAddition) = 참조 경로(model.generate 전체 prefill + OutputAddition), 모든 조건 행
  3. 개입 위치: 대상 층 suffix 행 = 개입 전 + λv (비트 동일), λ = 0 행·대상 앞 층은 그대로
  4. 크기 0 행 = 무개입 행, 무개입 행은 다른 행의 벡터와 무관 (같은 크기 batch의 전부 무개입 실행과 비트 동일, v2),
     개입은 출력을 바꾼다
  5. 벡터: Parallel + Orthogonal = Full, Random·크기 맞춤 노름, seed 재현·항목별 차이, 평균 차이(raw)는 토큰마다 같음
  float32(정확성)와 bfloat16(실행 가능성) 두 번 돈다.
--simulate: 서버 단계 모의 실행. 작은 모델로 basis(풀 입력 일부) → 연습 실행(실제 연습 항목 12개, 실제 COCO 이미지)
  → 본 실행(확증 절반 일부) → 검사 함수 → 평가 분석까지 같은 코드를 돈다. 무작위 모델이라 기준 실행 재현 검사는
  실패해야 정상이고, 기준 답을 작은 모델 자신의 무개입 답으로 바꾸면 통과해야 한다.
"""
import argparse
import copy
import statistics
import tempfile
from pathlib import Path

import torch

from src.common.config import read_yaml
from src.common.data import write_json
from src.common.paths import project_path
from src.core.causal_plan import load_plan
from src.core.causal_runtime import Runner
from src.core.feature_validation import require
from src.core.inference import answer_variant_ids
from experiments.step5_causal.selftest import tiny_model
from experiments.step6_contribution import directions as D
from experiments.step6_contribution.conditions import condition_rows, stage_work
from experiments.step6_contribution.contribution import attention_modules
from experiments.step6_contribution.runtime import (
    BasisAccumulator, Directions, SuffixAddition, build_vectors, extract, fit_bases, reference_rows, run_batches)
from experiments.step6_contribution.validation import reconstruction


TINY_LAYERS = (1, 2)


def synthetic_cases(directory):
    from PIL import Image

    generator = torch.Generator().manual_seed(7)
    sizes = [(56, 56), (84, 56), (56, 112), (112, 84), (84, 84)]
    names = []
    for i, (w, h) in enumerate(sizes):
        array = (torch.rand(h, w, 3, generator=generator) * 255).to(torch.uint8).numpy()
        Image.fromarray(array).save(directory / f"img{i}.png")
        names.append(f"img{i}.png")
    prompt = "Is the object category 'bed' present in all 3 images? Please answer with Yes or No."

    def case(order, labels, case_id):
        return {"case_id": case_id, "image_dir": str(directory), "image_order": [names[i] for i in order],
                "image_ids": [100 + i for i in order], "image_labels": labels, "prompt": prompt}
    return [case([0, 1, 2], ["P", "N", "P"], "pnp"), case([3, 4, 1], ["P", "P", "N"], "ppn"),
            case([2, 0, 3], ["N", "P", "P"], "npp"), case([4, 2, 0], ["P", "P", "P"], "ppp")]


class Probe:
    """suffix pass의 지정 층 attention 출력: SuffixAddition 앞(pre)과 뒤(post)."""

    def __init__(self, modules, length):
        self.pre, self.post, self.handles, self.length = {}, {}, [], length
        for index, module in modules.items():
            self.handles.append(module.register_forward_hook(self._store(index, self.pre), prepend=True))

    def attach_post(self, modules):
        for index, module in modules.items():
            self.handles.append(module.register_forward_hook(self._store(index, self.post)))

    def _store(self, index, target):
        def hook(module, args, output):
            if output[0].shape[1] == self.length:
                target[index] = output[0].detach().clone()
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()


def model_test(dtype):
    exact = dtype == torch.float32
    tolerance = 2e-4 if exact else 0.25
    identity = 1e-5 if exact else 0.25
    model, processor, answer_ids = tiny_model(dtype)
    runner = Runner(model, processor, answer_ids, answer_variant_ids(processor.tokenizer))
    results = {}
    with tempfile.TemporaryDirectory() as folder:
        cases = synthetic_cases(Path(folder))
        prepared = {c["case_id"]: runner.prepare(c) for c in cases}

        # 1. 추출·기저
        accumulator = BasisAccumulator(model.config.text_config.hidden_size, runner.device, TINY_LAYERS)
        stacked = {layer: [] for layer in TINY_LAYERS}
        extracted = {}
        for case in cases:
            item = prepared[case["case_id"]]
            parts, report = extract(model, item["inputs"], item["spans"], TINY_LAYERS)
            require(report["valid"], f"Reconstruction check failed on the tiny model ({case['case_id']}, {dtype})")
            identity = max(layer["identity_max_abs"] for layer in report["layers"].values())
            require(identity < 1e-9, f"float64 decomposition identity is not exact: {identity}")
            recon = reconstruction(model, item["inputs"], item["spans"], TINY_LAYERS, compute_dtype=torch.float64)
            require(recon["passed"], f"validate_real's reconstruction (float64) fails on the tiny model")
            mixed = "P" in case["image_labels"] and "N" in case["image_labels"]
            accumulator.add(parts, case["image_labels"], mixed)
            extracted[case["case_id"]] = (parts, report)
            for layer in TINY_LAYERS:
                stacked[layer] += [parts[layer][f"image_{i}"] for i, l in enumerate(case["image_labels"]) if l == "P"]
        payload = fit_bases(accumulator)
        for layer in TINY_LAYERS:
            data = torch.cat(stacked[layer]).double()
            for name, rule in (("v90", ("variance", 0.90)), ("v95", ("variance", 0.95))):
                reference = D.fit_basis(data, rule, centered=False)
                require(reference["k"] == payload["k"][name][layer],
                        f"Eigendecomposition k differs from SVD ({name}, layer {layer}): "
                        f"{payload['k'][name][layer]} vs {reference['k']}")
                ours = payload["bases"][name][layer].double()
                theirs = reference["basis"]
                gap = float((ours @ ours.T - theirs @ theirs.T).abs().max())
                require(gap < 1e-4, f"Projector differs from SVD ({name}, layer {layer}): {gap}")
        results["basis_matches_svd"] = True
        results["k"] = payload["k"]
        directions = Directions(payload, runner.device, TINY_LAYERS)

        # 2~5. 개입
        case = prepared["pnp"]
        parts, report = extracted["pnp"]
        start = case["spans"][-1]["end"] + 1
        prefix = runner.prefix(case, start)
        rows = condition_rows(pilot=True)
        batch = {"batch": "synthetic#s1", "slot": 1, "rows": rows}
        modules = attention_modules(model, TINY_LAYERS)
        addition = SuffixAddition(modules)
        probe = Probe(attention_modules(model, range(len(model.get_decoder().layers))), case["length"] - start)
        probe.attach_post(modules)
        try:
            (output,) = run_batches(runner, addition, case, prefix, parts, {"item_id": "synthetic"}, [batch], directions)
        finally:
            probe.remove()
        fast = output["rows"]
        vectors, scales = output["vectors"], output["scales"]
        for layer in TINY_LAYERS:
            pre, post = probe.pre[layer], probe.post[layer]
            for b in range(len(rows)):
                if float(scales[b]) == 0.0:
                    require(torch.equal(post[b], pre[b]), f"Row {b} with lambda 0 changed at layer {layer}")
                else:
                    expected = (pre[b].float() + scales[b] * vectors[layer][b]).to(pre.dtype)
                    require(torch.equal(post[b], expected), f"Row {b} is not pre + lambda v at layer {layer}")
        first = min(TINY_LAYERS)
        for layer in range(first):
            gap = float((probe.pre[layer] - probe.pre[layer][0:1]).abs().max())
            require(gap <= (1e-6 if exact else 0.05), f"Layer {layer} before the first target differs across rows")
        results["locality"] = True
        reference = reference_rows(runner, case, batch, vectors, scales)
        gaps = []
        for f_row, r_row in zip(fast, reference):
            require(f_row["answer"] == r_row["answer"] or not exact, f"Fast and reference answers differ: {r_row}")
            gaps.append(abs(f_row["margin"] - r_row["margin"]))
        require(max(gaps) <= tolerance, f"Fast path differs from the reference path: max margin gap {max(gaps)}")
        results["fast_vs_reference_max_gap"] = max(gaps)
        zero = fast[-1]
        require(abs(zero["margin"] - fast[0]["margin"]) <= identity and zero["raw"] == fast[0]["raw"],
                "The zero-size row differs from the no-intervention row")
        # 무개입 행이 다른 행의 벡터와 무관한지: 같은 크기 batch에서 모든 행이 무개입인 실행과 비트 단위로 같아야 한다.
        # (v2) batch 1 실행과 비교하면 batch 크기에 따라 CPU 커널 반올림이 달라 float32 1e-5를 넘을 수 있다 (서버에서 발견).
        idle, _ = runner.run_suffix(case, prefix, len(rows))
        keys = ("margin", "margin_fp32", "margin_bf16", "margin_bf16_variant", "raw")
        require(all(idle[0][k] == fast[0][k] for k in keys),
                "The no-intervention row depends on the other rows' vectors (same batch size, not bit-identical)")
        changes = [abs(r["margin"] - fast[0]["margin"]) for r, s in zip(fast, rows) if s["lam"] != 0]
        require(statistics.median(changes) > (1e-4 if exact else 1e-3), "Interventions barely change the output")
        results["median_abs_margin_change"] = statistics.median(changes)
        checks = output["vector_checks"]
        require(checks["sum_identity_relative_error_max"] <= 1e-5 and checks["norm_relative_error_max"] <= 1e-5,
                f"Vector identities fail: {checks}")
        again, _, _ = build_vectors(parts, 1, rows, directions, "synthetic")
        other, _, _ = build_vectors(parts, 1, rows, directions, "synthetic-other")
        random_rows = [b for b, s in enumerate(rows) if s["condition"] == "random"]
        for layer in TINY_LAYERS:
            require(torch.equal(again[layer], vectors[layer]), "Vectors are not reproducible")
            require(not torch.equal(other[layer][random_rows], vectors[layer][random_rows]),
                    "Random vectors do not depend on the item ID")
            require(not torch.equal(vectors[layer][random_rows[0]], vectors[layer][random_rows[1]]),
                    "Random draws are identical")
            md = vectors[layer][[b for b, s in enumerate(rows) if s["condition"] == "mean_difference"
                                 and s["size"] == "raw"][0]]
            require(torch.equal(md, md[0:1].expand_as(md)), "The raw mean-difference vector varies across tokens")
        results["vectors"] = checks
        addition.remove()
    return results


# ---------------------------------------------------------------- 서버 단계 모의 실행

def simulate():
    from experiments.step6_contribution import check as C
    from experiments.step6_contribution import run as R
    from experiments.step6_contribution.analysis import analyze
    from experiments.step6_contribution.freeze import load_config
    from experiments.step6_contribution.plan import load_step6_plan

    config = load_config()
    plan6 = load_step6_plan(config["plan_dir"])
    plan53 = load_plan(read_yaml(project_path(config["split_config"])))
    model, processor, answer_ids = tiny_model(torch.float32)
    runner = Runner(model, processor, answer_ids, answer_variant_ids(processor.tokenizer))
    results = {}
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        sim = copy.deepcopy(config)
        sim["dirs"] = {name: str(root / name) for name in ("basis", "pilot", "main", "evaluation")}
        manifest = {"version": "simulation", "verification": config["verification"],
                    "plan_sufficiency": {"criteria": plan6["check"]["sufficiency"]["criteria"]}}
        meta = {"status": "completed", "frozen": {"version": "simulation"}, "environment": {"simulated": True}}
        # basis: 구성마다 2개씩
        chosen, seen = [], {}
        for row in plan6["pool"]:
            if seen.get(row["composition"], 0) < 2:
                chosen.append(row)
                seen[row["composition"]] = seen.get(row["composition"], 0) + 1
        small = {**plan6, "pool": chosen}
        (root / "basis").mkdir()
        R.run_basis(runner, small, root / "basis", {}, TINY_LAYERS)
        write_json(meta, root / "basis" / "run.json")
        report, _ = C.check_basis(sim, small, manifest, expected_layers=TINY_LAYERS)
        failed = set(report["failed"])
        require(failed <= {"mean_difference_ready"}, f"Unexpected basis check failures: {failed}")
        results["basis_check_failed_expected_only"] = sorted(failed)
        write_json({**report, "status": "passed"}, root / "basis" / "basis_check.json")
        directions = Directions(torch.load(root / "basis" / "basis.pt"), runner.device, TINY_LAYERS)
        # 연습 실행: 실제 연습 항목 12개
        (root / "pilot").mkdir()
        R.run_items(runner, plan6, plan53, "pilot", root / "pilot", directions)
        write_json({**meta, "basis_sha256": report["basis_sha256"]}, root / "pilot" / "run.json")
        pilot = C.check_items(sim, plan6, plan53, manifest, "pilot")
        require(set(pilot["failed"]) == {"reproduction_no_intervention"},
                f"Pilot checks: expected only the reproduction check to fail, got {pilot['failed']}")
        own = copy.deepcopy(plan53)
        run = C.load_run(root / "pilot")
        for batch in run["batches"].values():
            none = batch["rows"][0]
            base = own["inputs"][batch["input_id"]]["baseline"]
            base.update(prediction=none["answer"], fp32_variant_logit_margin=none["margin"],
                        logit_margin=none["margin_bf16"])
        pilot_own = C.check_items(sim, plan6, own, manifest, "pilot")
        require(pilot_own["status"] == "passed", f"Pilot checks with own baselines failed: {pilot_own['failed']}")
        results["pilot_checks"] = {name: c["passed"] for name, c in pilot_own["checks"].items()}
        results["pilot_equivalence"] = {k: pilot_own["checks"]["fast_path_equivalence"]["detail"][k]
                                        for k in ("n", "margin_gap_max", "answer_agreement_all_reported")}
        estimate = C.estimate(sim, plan6, plan53, 8)
        results["estimate_hours_tiny_cpu"] = estimate["total_hours_with_overhead"]
        # 본 실행 일부 → 검사 → 평가 분석
        main_items = {"one_negative": plan6["items"]["main"]["one_negative"][:6],
                      "all_positive": plan6["items"]["main"]["all_positive"][:2]}
        small6 = {**plan6, "items": {**plan6["items"], "main": main_items}}
        (root / "main").mkdir()
        R.run_items(runner, small6, plan53, "main", root / "main", directions)
        write_json({**meta, "basis_sha256": report["basis_sha256"]}, root / "main" / "run.json")
        run = C.load_run(root / "main")
        for batch in run["batches"].values():
            none = batch["rows"][0]
            base = own["inputs"][batch["input_id"]]["baseline"]
            base.update(prediction=none["answer"], fp32_variant_logit_margin=none["margin"],
                        logit_margin=none["margin_bf16"])
        main = C.check_items(sim, small6, own, manifest, "main")
        require(main["status"] == "passed", f"Main checks failed: {main['failed']}")
        preregistration = config["preregistration"]
        result = analyze(own, small6, run, preregistration, None, reps=50, seed=1)
        require(result["counts"]["planned"] == 6 and result["counts"]["analyzed"] == 6, f"{result['counts']}")
        require(len(result["primary"]) == 2 and all(r["label"] for r in result["primary"]), "Primary rows missing")
        allp = result["all_positive_side_effects"]["counts"]
        require(sum(allp.values()) == 2 and set(allp) <= {"valid", "none_answer_not_yes"}, f"All-P items: {allp}")
        results["main_analysis_counts"] = result["counts"]
        results["main_batches"] = len(run["batches"])
    return results


def main():
    parser = argparse.ArgumentParser(description="STEP 6-2 실행 경로 자체 시험 (CPU)")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    require(args.model or args.simulate, "Choose --model and/or --simulate")
    if args.model:
        for dtype in (torch.float32, torch.bfloat16):
            print(f"[model {dtype}]", model_test(dtype))
    if args.simulate:
        print("[simulate]", simulate())
    print("통과")


if __name__ == "__main__":
    main()
