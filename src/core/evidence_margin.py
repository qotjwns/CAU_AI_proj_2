"""STEP 5-1 탐색 분석: Positive 증거(z_i)가 Negative 증거 외에 VLM Yes−No Margin을 추가로 설명하는가 (CPU).

STEP 4 출력(고정 z_i Probe 확률, VLM 첫 토큰 logit margin)만 읽고 고유 입력 단위로 회귀한다.
계수의 신뢰구간은 STEP 4와 같은 이미지 ID 군집 부트스트랩(케이스 가중치 = 세 이미지 추출 횟수의 곱)으로 구한다.
"""
import csv
from collections import defaultdict

import torch

from src.common.data import read_jsonl
from src.core.feature_validation import require
from src.core.probe_cases import ImageBootstrap
from src.core.probe_freeze import sha256_file


SAMPLES = {
    "one_negative": "GT No with exactly one N image (PPN, PNP, NPP)",
    "all_positive": "PPP inputs (GT Yes)",
}
# (모형, 비교 기준 모형, 항). class·n_pos는 클래스 고정효과와 N 위치 더미, 나머지는 z_i logit 항.
MODELS = {
    "one_negative": (
        ("base", None, ("class", "n_pos")),
        ("neg", "base", ("class", "n_pos", "z_N")),
        ("neg_pos", "neg", ("class", "n_pos", "z_N", "z_P_mean")),
        ("neg_pos_min", "neg", ("class", "n_pos", "z_N", "z_P_min")),
        ("neg_flex", "base", ("class", "n_pos", "z_N", "z_N_sq", "z_N_cu")),
        ("neg_flex_pos", "neg_flex", ("class", "n_pos", "z_N", "z_N_sq", "z_N_cu", "z_P_mean")),
    ),
    "all_positive": (
        ("base", None, ("class",)),
        ("pos", "base", ("class", "z_P_mean")),
        ("pos_min", "base", ("class", "z_P_min")),
    ),
}
REPORTED_TERMS = ("n_pos2", "n_pos3", "z_N", "z_N_sq", "z_N_cu", "z_P_mean", "z_P_min")


def read_inputs(step4_dir):
    """zi_hq_slots.csv(고유 입력 × 위치)와 추출 manifest의 logit margin을 (클래스, 이미지 순서)로 맞춘다."""
    margins, pairs = {}, {}
    for record in read_jsonl(step4_dir / "extraction" / "manifest.jsonl"):
        case = record["case"]
        key = (case["class"], tuple(case["image_ids"]))
        require(margins.setdefault(key, record["logit_margin"]) == record["logit_margin"],
                f"Records sharing an input have different logit margins: {key}")
        require(pairs.setdefault(key, case["pair_id"]) == case["pair_id"], f"Input appears in two pairs: {key}")
    rows = defaultdict(list)
    with (step4_dir / "zi_hq_slots.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            rows[row["case_id"]].append(row)
    layers = sorted(int(name.rsplit("layer", 1)[1]) for name in next(iter(rows.values()))[0]
                    if name.startswith("p_z_i_layer"))
    inputs = []
    for case_id, members in rows.items():
        members.sort(key=lambda row: int(row["position"]))
        require([int(row["position"]) for row in members] == [1, 2, 3], f"Input {case_id} does not have positions 1-3")
        first = members[0]
        image_ids = [int(row["image_id"]) for row in members]
        key = (first["class"], tuple(image_ids))
        require(key in margins, f"No logit margin for input {case_id}")
        inputs.append({
            "case_id": case_id, "pair_id": pairs[key], "mode": first["mode"], "class": first["class"],
            "composition": first["composition"],
            "gt": first["gt"], "prediction": first["prediction"], "n_records": int(first["n_records"]),
            "image_ids": image_ids, "labels": [row["label"] for row in members], "margin": margins[key],
            "probabilities": torch.tensor([[float(row[f"p_z_i_layer{layer}"]) for layer in layers] for row in members],
                                          dtype=torch.float64),
        })
    require(len({(item["class"], tuple(item["image_ids"])) for item in inputs}) == len(inputs),
            "zi_hq_slots.csv lists the same input more than once")
    return inputs, layers


def select(inputs, sample):
    if sample == "one_negative":
        return [item for item in inputs if item["gt"] == "No" and item["labels"].count("N") == 1]
    if sample == "all_positive":
        return [item for item in inputs if item["labels"].count("N") == 0]
    raise ValueError(f"Unknown sample: {sample}")


def evidence(items, column, clip):
    """입력별 z_i logit 항: z_N(N 1장일 때), z_P_mean, z_P_min."""
    probability = torch.stack([item["probabilities"][:, column] for item in items]).clamp(clip, 1 - clip)
    z = torch.log(probability) - torch.log1p(-probability)
    positive = torch.tensor([[label == "P" for label in item["labels"]] for item in items])
    z_positive = z.masked_fill(~positive, float("nan"))
    terms = {"z_P_mean": z_positive.nanmean(dim=1),
             "z_P_min": z_positive.masked_fill(~positive, float("inf")).min(dim=1).values}
    if not bool(positive.all()):
        z_negative = z[~positive].reshape(len(items), -1)
        require(z_negative.shape[1] == 1, "z_N is defined only for inputs with exactly one N image")
        centered = z_negative[:, 0] - z_negative[:, 0].mean()
        terms.update(z_N=z_negative[:, 0], z_N_sq=centered ** 2, z_N_cu=centered ** 3)
    return terms


def design(items, terms, groups):
    """회귀 설계 행렬. 클래스 고정효과(절편 대신 클래스별 더미)와 N 위치 더미(1번 위치 기준)."""
    columns, names = [], []
    for group in groups:
        if group == "class":
            for name in sorted({item["class"] for item in items}):
                columns.append(torch.tensor([item["class"] == name for item in items], dtype=torch.float64))
                names.append(f"class={name}")
        elif group == "n_pos":
            position = torch.tensor([item["labels"].index("N") + 1 for item in items])
            for value in (2, 3):
                columns.append((position == value).double())
                names.append(f"n_pos{value}")
        else:
            columns.append(terms[group])
            names.append(group)
    return torch.stack(columns, dim=1), names


def weighted_fit(x, y, weights):
    """가중 최소제곱. weights [반복, n] → 계수 [반복, p], R² [반복]."""
    xtw = x.T.unsqueeze(0) * weights.unsqueeze(1)
    beta = torch.linalg.lstsq(xtw @ x, (xtw @ y).unsqueeze(-1)).solution.squeeze(-1)
    residual = y.unsqueeze(0) - beta @ x.T
    total = weights.sum(dim=1, keepdim=True)
    mean = (weights * y).sum(dim=1, keepdim=True) / total
    r2 = 1 - (weights * residual ** 2).sum(dim=1) / (weights * (y - mean) ** 2).sum(dim=1)
    return beta, r2


def fit_all(x, y, weights, chunk):
    observed, r2 = weighted_fit(x, y, torch.ones(1, len(y), dtype=torch.float64))
    if weights is None:
        return observed[0], float(r2[0]), None, None
    parts = [weighted_fit(x, y, weights[start:start + chunk]) for start in range(0, len(weights), chunk)]
    return observed[0], float(r2[0]), torch.cat([beta for beta, _ in parts]), torch.cat([value for _, value in parts])


def percentile(values):
    values = values[torch.isfinite(values)]
    if not len(values):
        return None, None, 0
    low, high = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(low), float(high), len(values)


def residualize(x, values):
    beta = torch.linalg.lstsq(x, values.unsqueeze(-1)).solution.squeeze(-1)
    return values - x @ beta


def diagnostics(items, terms, layer):
    """z_N·z_P 편상관(클래스·위치 통제)과 같은 N 이미지 안에서 z_P 분산 비율."""
    base, _ = design(items, terms, ("class", "n_pos"))
    negative = residualize(base, terms["z_N"])
    positive = residualize(base, terms["z_P_mean"])
    partial = float((negative @ positive) / (negative.norm() * positive.norm()))
    groups = defaultdict(list)
    for index, item in enumerate(items):
        groups[(item["class"], item["image_ids"][item["labels"].index("N")])].append(index)
    values = terms["z_P_mean"]
    within = sum(float(((values[members] - values[members].mean()) ** 2).sum()) for members in groups.values())
    total = float(((values - values.mean()) ** 2).sum())
    return {
        "layer": layer, "n_inputs": len(items), "n_negative_images": len(groups),
        "inputs_per_negative_image": sorted({len(members) for members in groups.values()}),
        "sd_z_N": float(terms["z_N"].std()), "sd_z_P_mean": float(values.std()),
        "partial_correlation_z_N_z_P_mean": partial,
        "within_negative_image_share_of_z_P_mean_variance": within / total if total else None,
    }


def run_evidence_margin(step4_dir, config):
    analysis, statistics = config["analysis"], config["statistics"]
    inputs, layers = read_inputs(step4_dir)
    require(analysis["primary_layer"] in layers, f"Primary layer {analysis['primary_layer']} is not in {layers}")
    bootstrap = ImageBootstrap([image_id for item in inputs for image_id in item["image_ids"]],
                               statistics["reps"], statistics["seed"])
    samples = [analysis["primary_sample"], *analysis["secondary_samples"]]
    coefficients, fits, checks, tables = [], [], [], {}
    for sample in samples:
        items = select(inputs, sample)
        require(items, f"No inputs in sample {sample}")
        y = torch.tensor([item["margin"] for item in items], dtype=torch.float64)
        weights = bootstrap.case_weights([item["image_ids"] for item in items])
        tables[sample] = items
        for column, layer in enumerate(layers):
            terms = evidence(items, column, analysis["logit_clip"])
            if "z_N" in terms:
                checks.append({"sample": sample, **diagnostics(items, terms, layer)})
            results = {}
            for model, baseline, groups in MODELS[sample]:
                x, names = design(items, terms, groups)
                beta, r2, boot_beta, boot_r2 = fit_all(x, y, weights, statistics["chunk"])
                results[model] = (r2, boot_r2)
                for index, name in enumerate(names):
                    if name not in REPORTED_TERMS:
                        continue
                    low, high, valid = percentile(boot_beta[:, index])
                    scale = float(x[:, index].std()) if name.startswith("z_") else None
                    coefficients.append({
                        "sample": sample, "layer": layer, "model": model, "term": name, "n": len(items),
                        "estimate": float(beta[index]), "ci_low": low, "ci_high": high, "ci_valid": valid,
                        "sd_term": scale, "estimate_per_sd": float(beta[index]) * scale if scale else None,
                        "primary": (sample == analysis["primary_sample"] and layer == analysis["primary_layer"]
                                    and model == analysis["primary_model"] and name == "z_P_mean"),
                    })
                row = {"sample": sample, "layer": layer, "model": model, "terms": "+".join(groups), "n": len(items),
                       "r2": r2, "compared_to": baseline, "delta_r2": None, "ci_low": None, "ci_high": None,
                       "ci_valid": 0}
                if baseline:
                    base_r2, base_boot = results[baseline]
                    low, high, valid = percentile(boot_r2 - base_boot)
                    row.update(delta_r2=r2 - base_r2, ci_low=low, ci_high=high, ci_valid=valid)
                fits.append(row)
            for index, item in enumerate(items):
                item.setdefault("terms", {})[layer] = {name: float(values[index]) for name, values in terms.items()}

    primary = next(row for row in coefficients if row["primary"])
    primary_fit = next(row for row in fits if row["sample"] == analysis["primary_sample"]
                       and row["layer"] == analysis["primary_layer"] and row["model"] == analysis["primary_model"])
    negative = next(row for row in coefficients if row["sample"] == analysis["primary_sample"]
                    and row["layer"] == analysis["primary_layer"] and row["model"] == analysis["primary_model"]
                    and row["term"] == "z_N")
    report = {
        "status": config["status"], "note": config["note"], "analysis": analysis, "limitations": config["limitations"],
        "statistics": statistics, "layers": layers, "samples": {name: SAMPLES[name] for name in samples},
        "models": {sample: [{"model": model, "compared_to": baseline, "terms": list(groups)}
                            for model, baseline, groups in MODELS[sample]] for sample in samples},
        "inputs_sha256": {name: sha256_file(step4_dir / name)
                          for name in ("zi_hq_slots.csv", "extraction/manifest.jsonl", "analysis.json")},
        "counts": {"unique_inputs": len(inputs), **{f"{name}_inputs": len(items) for name, items in tables.items()},
                   "bootstrap_images": len(bootstrap.index)},
        "headline": {"z_P_mean": primary, "z_N": negative, "delta_r2": primary_fit},
        "diagnostics": checks, "coefficients": coefficients, "fits": fits,
    }
    return report, tables
