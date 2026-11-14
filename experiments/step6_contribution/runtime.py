"""STEP 6-2 실행 핵심 (GPU, 기준선과 같은 bf16·sdpa). STEP 6-1 코드와 STEP 5-3 Runner를 고치지 않고 쓴다.

입력 하나의 순서:
1. 기여 추출: 무개입 전체 prefill 한 번(batch 1, use_cache=False)에서 L21–24 질문 토큰의 이미지별 기여 c_i를
   다시 계산한다 (contribution.contributions, STEP 6-1에서 검증). 재구성 검사(항등식·attention·출력, 허용 오차는
   validation.TOLERANCES)를 함께 기록한다.
2. 조건 벡터: 대상 자리의 c_i에서 condition_rows의 행마다 v를 만든다 (directions.py 정의).
3. 개입 실행: STEP 5-3B와 같은 빠른 경로. prefix(마지막 vision_end까지)를 KV cache로 한 번 계산하고,
   질문 구간(suffix)을 batch 행으로 다시 계산하면서 L21–24 attention 출력에 행마다 λ·v를 더한다 (SuffixAddition).
   0번 행은 같은 batch의 무개입 행이다. 생성은 Runner.greedy(기준 실행과 같은 greedy·반복 패널티)로 이어 간다.
   - 질문 구간의 모든 행이 suffix이므로 suffix 전체가 개입 위치다. 생성 단계(길이 1)는 직접 바꾸지 않는다.
4. 참조 경로(연습 실행 전용): model.generate 전체 prefill에 contribution.OutputAddition으로 같은 벡터를 더한다.

P 기저·평균 차이 (basis 단계, STEP 3 풀):
- 기저: 모든 풀 입력의 P 자리 기여 (질문 토큰 전부, 층별). 중심화 없이 2차 모멘트 M = Σ c cᵀ (float64)를 누적해
  고유분해한다 (SVD 오른쪽 특이벡터와 같다). 설명 비율 = 누적 고윳값 / 전체.
- 평균 차이: N·P가 함께 있는 풀 입력(1,539개)의 N 자리 평균 μ_N − P 자리 평균 μ_P (층별, 질문 토큰·자리 전체 평균).
"""
from collections import Counter

import torch

from src.core.feature_validation import require
from experiments.step6_contribution import directions as D
from experiments.step6_contribution.conditions import BASES, LAYERS, random_seed
from experiments.step6_contribution.contribution import (
    AttentionRecorder, OutputAddition, attention_modules, contributions, key_groups, question_positions, reconstruct)
from experiments.step6_contribution.validation import TOLERANCES, model_dtype, reconstruction_passes, stats


MATCH_EPSILON = 1e-12            # 크기 맞춤: ‖v‖ ≤ ε·‖c_N‖이면 0 벡터로 두고 개수를 센다


# ---------------------------------------------------------------- 개입 (빠른 경로)

class SuffixAddition:
    """빠른 경로 suffix pass에서 지정 층 attention 출력의 suffix 전체 행에 행마다 λ_b·v_b를 더한다.

    vectors: {층: [B, S, H] float32}, scales: [B] float32. λ_b = 0인 행은 건드리지 않는다 (무개입·크기 0 행).
    길이가 S인 forward(suffix pass)에서만 작동하고 생성 단계(길이 1)는 건드리지 않는다.
    """

    def __init__(self, modules):
        self.modules = dict(modules)
        self.vectors = self.scales = self.length = None
        self.applied = Counter()
        self.handles = [module.register_forward_hook(self._hook(index)) for index, module in self.modules.items()]

    def set(self, vectors, scales, length):
        require(set(vectors) == set(self.modules), "Vectors must cover every intervention layer")
        self.vectors, self.scales, self.length = vectors, scales, length

    def clear(self):
        self.vectors = self.scales = self.length = None

    def _hook(self, index):
        def hook(module, args, output):
            if self.vectors is None:
                return None
            out = output[0]
            if out.shape[1] != self.length:
                return None
            require(out.shape[0] == self.scales.shape[0], "Batch size differs from the number of condition rows")
            rows = (self.scales != 0).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                return None
            changed = out.clone()
            added = self.scales[rows, None, None] * self.vectors[index][rows]
            changed[rows] = (out[rows].to(torch.float32) + added).to(out.dtype)
            self.applied[index] += 1
            return (changed, *output[1:])
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


# ---------------------------------------------------------------- 기여 추출

def extract(model, inputs, spans, layers=LAYERS, compute_dtype=torch.float64):
    """무개입 전체 prefill 한 번: 층별 {묶음: c_g [S, H] float32}와 재구성 검사.

    재계산(softmax, 가중합, W_O)은 float64로 한다. float32로 하면 항등식 검사(Σ c_g 대 분해하지 않은 계산)가 합산 순서
    반올림만으로 절대 허용 오차 1e-5를 넘을 수 있다 (작은 모델 모의 실행에서 확인: 0 근처 원소 1.06e-5).
    허용 오차는 바꾸지 않았다. 검사 뒤 기여는 float32로 바꿔 돌려준다 (개입 덧셈은 float32).
    """
    length = inputs["input_ids"].shape[1]
    queries = question_positions(spans, length)
    modules = attention_modules(model, layers)
    groups = key_groups(length, spans)
    recorder = AttentionRecorder(modules, length)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        recorder.remove()
    dtype = model_dtype(model)
    parts, report, valid = {}, {}, True
    with torch.inference_mode():
        for layer in layers:
            record = recorder.records.get(layer)
            require(record is not None and "o" in record, f"Layer {layer}: attention was not recorded")
            result = contributions(modules[layer], record, queries, groups, compute_dtype)
            recon = reconstruct(result["parts"], result["bias"])
            kernel_z = record["z"][0, queries].to(compute_dtype)
            kernel_o = record["o"][0, queries].to(compute_dtype)
            checks = {"identity": stats(recon, result["reference"], TOLERANCES["identity"]),
                      "attention": stats(result["z_all"], kernel_z, TOLERANCES[dtype]["attention"]),
                      "output": stats(recon, kernel_o, TOLERANCES[dtype]["output"])}
            ok = reconstruction_passes({layer: checks}, dtype)          # 판정 규칙: validation.RULE (v3)
            valid &= ok
            report[layer] = {
                "passed": ok, **{f"{name}_max_abs": check["max_abs"] for name, check in checks.items()},
                "attention_relative": checks["attention"]["relative"], "output_relative": checks["output"]["relative"],
                "output_worst_excess": checks["output"]["worst_excess"],
                "elementwise_allclose": {name: check["allclose"] for name, check in checks.items()},
                "row_sum_max_error": float((result["row_sums"].double() - 1).abs().max()),
                "o_norm_mean": float(kernel_o.norm(dim=-1).mean()),
                "share": {name: round(value, 6) for name, value in result["share"].items()}}
            parts[layer] = {name: value.to(torch.float32) for name, value in result["parts"].items()}
            recorder.records.pop(layer)
    return parts, {"valid": bool(valid), "length": length, "question": [queries[0], queries[-1]],
                   "n_question": len(queries), "layers": report}


# ---------------------------------------------------------------- 조건 벡터

class Directions:
    """basis 단계 출력(basis.pt)의 기저 U [H, k] (기저 이름·층별)와 평균 μ_N, μ_P (층별)."""

    def __init__(self, payload, device, expected_layers=LAYERS):
        self.layers = tuple(payload["layers"])
        require(self.layers == tuple(expected_layers), "The basis file was made for different layers")
        require(set(payload["bases"]) == set(BASES), "The basis file does not have every preregistered basis")
        self.bases = {name: {layer: U.to(device, torch.float32) for layer, U in by_layer.items()}
                      for name, by_layer in payload["bases"].items()}
        self.mu_n = {layer: v.to(device, torch.float32) for layer, v in payload["mu_n"].items()}
        self.mu_p = {layer: v.to(device, torch.float32) for layer, v in payload["mu_p"].items()}
        self.k = payload["k"]


def match_norm(vector, reference):
    """행(토큰)마다 vector 방향을 유지하고 노름을 reference에 맞춘다. 거의 0인 행은 0으로 두고 개수를 돌려준다."""
    norm = vector.double().norm(dim=-1, keepdim=True)
    target = reference.double().norm(dim=-1, keepdim=True)
    degenerate = norm <= MATCH_EPSILON * target
    scaled = torch.where(degenerate, torch.zeros_like(norm), target / norm.clamp_min(1e-300))
    return (vector.double() * scaled).to(vector.dtype), int(degenerate.sum())


def condition_vector(row, c_n, layer, directions, generator):
    name = row["condition"]
    if name == "full":
        return c_n, 0
    if name in ("parallel", "orthogonal"):
        basis = directions.bases[row["basis"]][layer]
        vector = D.parallel(c_n, basis) if name == "parallel" else D.orthogonal(c_n, basis)
        return match_norm(vector, c_n) if row["size"] == "match" else (vector, 0)
    if name == "random":
        return D.random_same_norm(c_n.cpu(), generator).to(c_n.device), 0
    if name == "mean_difference":
        mu_n, mu_p = directions.mu_n[layer], directions.mu_p[layer]
        if row["size"] == "raw":
            return D.mean_difference(mu_n, mu_p).expand_as(c_n), 0
        return D.mean_difference(mu_n, mu_p, c_n, "match"), 0
    raise ValueError(f"Unknown condition {name}")


def build_vectors(parts, slot, rows, directions, item_id):
    """batch 행마다 개입 벡터 {층: [B, S, H] float32}, λ [B], 벡터 검사 값.

    Random은 행(추첨)마다 random_seed(항목, 자리, 추첨)로 생성기를 만들고 층 순서대로 뽑는다 (재현 가능).
    """
    layers = directions.layers
    target = {layer: parts[layer][f"image_{slot}"] for layer in layers}
    first = target[layers[0]]
    count, length, hidden = len(rows), first.shape[0], first.shape[1]
    checks = {"degenerate_match_rows": 0, "norm_relative_error_max": 0.0, "sum_identity_relative_error_max": 0.0}
    with torch.inference_mode():
        vectors = {layer: torch.zeros(count, length, hidden, dtype=torch.float32, device=first.device)
                   for layer in layers}
        scales = torch.tensor([float(row["lam"]) for row in rows], dtype=torch.float32, device=first.device)
        for b, row in enumerate(rows):
            if row["condition"] == "none":
                continue
            generator = (torch.Generator().manual_seed(random_seed(item_id, slot, row["draw"]))
                         if row["condition"] == "random" else None)
            for layer in layers:
                vector, degenerate = condition_vector(row, target[layer], layer, directions, generator)
                vectors[layer][b] = vector
                checks["degenerate_match_rows"] += degenerate
                if row["condition"] == "random" or row["size"] == "match":
                    reference = target[layer].double().norm(dim=-1)
                    error = (vector.double().norm(dim=-1) - reference).abs() / reference.clamp_min(1e-300)
                    keep = vector.double().norm(dim=-1) > 0
                    if bool(keep.any()):
                        checks["norm_relative_error_max"] = max(checks["norm_relative_error_max"],
                                                                float(error[keep].max()))
        for name in BASES:
            for layer in layers:
                c_n, basis = target[layer], directions.bases[name][layer]
                total = D.parallel(c_n, basis) + D.orthogonal(c_n, basis)
                error = float((total - c_n).double().norm(dim=-1).max() / c_n.double().norm(dim=-1).max())
                checks["sum_identity_relative_error_max"] = max(checks["sum_identity_relative_error_max"], error)
    return vectors, scales, checks


def vector_summary(parts, slot, directions, extraction):
    """기술 통계: 층별 c_N 크기, 기저별 평행 성분 에너지 비율·직교 잔차 비율, 평균 차이 방향과의 cos."""
    summary = {}
    with torch.inference_mode():
        for layer in directions.layers:
            c_n = parts[layer][f"image_{slot}"].double()
            norm = c_n.norm(dim=-1).clamp_min(1e-300)
            difference = (directions.mu_n[layer] - directions.mu_p[layer]).double()
            row = {"c_norm_mean": float(norm.mean()),
                   "c_over_o_norm": float(norm.mean()) / max(extraction["layers"][layer]["o_norm_mean"], 1e-300),
                   "target_attention_share": extraction["layers"][layer]["share"][f"image_{slot}"],
                   "cos_mean_difference": float((c_n @ difference / (norm * difference.norm())).mean())}
            for name in BASES:
                basis = directions.bases[name][layer].double()
                parallel = (c_n @ basis) @ basis.T
                orthogonal = c_n - parallel
                row[f"parallel_energy_{name}"] = float((parallel.norm(dim=-1).pow(2) / norm.pow(2)).mean())
                row[f"orthogonal_ratio_{name}"] = float((orthogonal.norm(dim=-1) / norm).mean())
            summary[layer] = row
    return summary


# ---------------------------------------------------------------- 실행

def run_batches(runner, addition, prepared, prefix, parts, item, batches, directions):
    """한 입력의 batch들을 빠른 경로로 실행한다. 행 결과와 벡터 검사를 batch마다 돌려준다."""
    from src.core.causal_runtime import timed

    length = prepared["length"] - prefix["length"]
    outputs = []
    for batch in batches:
        vectors, scales, checks = build_vectors(parts, batch["slot"], batch["rows"], directions, item["item_id"])
        addition.set(vectors, scales, length)
        before = dict(addition.applied)
        try:
            (rows, _), seconds = timed(runner.run_suffix, prepared, prefix, len(batch["rows"]))
        finally:
            addition.clear()
        applied = {layer: addition.applied[layer] - before.get(layer, 0) for layer in addition.modules}
        require(all(count == 1 for count in applied.values()), f"The suffix addition ran {applied} times")
        outputs.append({"rows": rows, "seconds": seconds, "vector_checks": checks, "vectors": vectors,
                        "scales": scales})
    return outputs


def reference_rows(runner, prepared, batch, vectors, scales):
    """연습 실행 전용: batch의 행마다 model.generate 전체 prefill에 같은 벡터를 더한 결과 (한 행씩)."""
    layers = tuple(vectors)
    modules = attention_modules(runner.model, layers)
    queries = question_positions(prepared["spans"], prepared["length"])
    results = []
    for b, row in enumerate(batch["rows"]):
        addition = None
        if float(scales[b]) != 0.0:
            addition = OutputAddition(modules, {layer: vectors[layer][b] for layer in layers}, queries,
                                      float(scales[b]), prepared["length"])
        try:
            results.append({**row, **runner.reference(prepared)})
        finally:
            if addition is not None:
                addition.remove()
    return results


# ---------------------------------------------------------------- P 기저·평균 차이 (basis 단계)

class BasisAccumulator:
    """층별 P 기여 2차 모멘트(기저용, 모든 풀 입력)와 N·P 기여 합(평균 차이용, N·P가 함께 있는 입력)."""

    KEYS = ("p_basis", "p_mixed", "n_mixed")

    def __init__(self, hidden, device, layers=LAYERS):
        self.layers = tuple(layers)
        self.moment = {layer: torch.zeros(hidden, hidden, dtype=torch.float64, device=device) for layer in layers}
        self.sums = {key: {layer: torch.zeros(hidden, dtype=torch.float64, device=device) for layer in layers}
                     for key in self.KEYS}
        self.vectors, self.slots, self.inputs = Counter(), Counter(), Counter()

    def add(self, parts, labels, mixed):
        with torch.inference_mode():
            for layer in self.layers:
                by_label = {label: [parts[layer][f"image_{i}"] for i, l in enumerate(labels) if l == label]
                            for label in ("P", "N")}
                if by_label["P"]:
                    data = torch.cat(by_label["P"]).double()
                    self.moment[layer] += data.T @ data
                    self.sums["p_basis"][layer] += data.sum(dim=0)
                    if mixed:
                        self.sums["p_mixed"][layer] += data.sum(dim=0)
                if mixed and by_label["N"]:
                    self.sums["n_mixed"][layer] += torch.cat(by_label["N"]).double().sum(dim=0)
        length = parts[self.layers[0]]["image_0"].shape[0]
        n_p, n_n = labels.count("P"), labels.count("N")
        self.inputs["basis"] += int(n_p > 0)
        self.slots["p_basis"] += n_p
        self.vectors["p_basis"] += n_p * length
        if mixed:
            self.inputs["mixed"] += 1
            self.slots["p_mixed"] += n_p
            self.slots["n_mixed"] += n_n
            self.vectors["p_mixed"] += n_p * length
            self.vectors["n_mixed"] += n_n * length


def fit_bases(accumulator):
    """층별 고유분해(float64, CPU)로 사전 등록 기저(v90, v95, r64)와 μ_N, μ_P를 만든다."""
    payload = {"layers": list(accumulator.layers), "bases": {name: {} for name in BASES},
               "k": {name: {} for name in BASES}, "explained": {name: {} for name in BASES},
               "eigenvalues": {}, "mu_n": {}, "mu_p": {}, "mu_p_basis": {},
               "counts": {"vectors": dict(accumulator.vectors), "slots": dict(accumulator.slots),
                          "inputs": dict(accumulator.inputs)}}
    for layer in accumulator.layers:
        moment = accumulator.moment[layer].cpu()
        moment = (moment + moment.T) / 2
        values, vectors = torch.linalg.eigh(moment)
        values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
        share = torch.cumsum(values, dim=0) / values.sum()
        payload["eigenvalues"][layer] = values
        for name, (kind, value) in BASES.items():
            if kind == "variance":
                k = int((share < value - 1e-12).sum().item()) + 1
            else:
                k = int(value)
            basis = vectors[:, :k].contiguous().to(torch.float32)
            D.check_orthonormal(basis, tolerance=1e-5)
            payload["bases"][name][layer] = basis
            payload["k"][name][layer] = k
            payload["explained"][name][layer] = float(share[k - 1])
        payload["mu_n"][layer] = (accumulator.sums["n_mixed"][layer] / accumulator.vectors["n_mixed"]).cpu().float()
        payload["mu_p"][layer] = (accumulator.sums["p_mixed"][layer] / accumulator.vectors["p_mixed"]).cpu().float()
        payload["mu_p_basis"][layer] = (accumulator.sums["p_basis"][layer] / accumulator.vectors["p_basis"]).cpu().float()
    return payload


def basis_report(payload):
    rows = {}
    for layer in payload["layers"]:
        values = payload["eigenvalues"][layer]
        share = torch.cumsum(values, dim=0) / values.sum()
        difference = payload["mu_n"][layer].double() - payload["mu_p"][layer].double()
        rows[layer] = {
            "k": {name: payload["k"][name][layer] for name in BASES},
            "explained": {name: payload["explained"][name][layer] for name in BASES},
            "explained_at": {str(k): float(share[k - 1]) for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
                             if k <= len(share)},
            "mu_n_norm": float(payload["mu_n"][layer].double().norm()),
            "mu_p_norm": float(payload["mu_p"][layer].double().norm()),
            "mean_difference_norm": float(difference.norm()),
            "mean_difference_parallel_energy_v90": float(
                ((difference @ payload["bases"]["v90"][layer].double()).norm() / difference.norm()) ** 2)}
    return {"layers": rows, "counts": payload["counts"]}
