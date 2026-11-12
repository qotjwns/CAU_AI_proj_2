"""STEP 5-1 보완 분석 A1~A7 (CPU, exploratory).

STEP 4 one-N 입력에서 고정 z_i Probe logit과 VLM Yes−No margin·False-Yes의 관계를 여러 방식으로 본다.
A7만 STEP 3 OOF 점수와 STEP 2 margin을 쓴다. 신뢰구간은 모두 이미지 ID 군집 부트스트랩이다.
"""
import csv
from collections import defaultdict
import random

import torch
import torch.nn.functional as F

from src.common.data import read_jsonl
from src.core.evidence_margin import design, fit_all, percentile, read_inputs, weighted_fit
from src.core.feature_validation import require
from src.core.probe import auroc, fit_logistic, predict
from src.core.probe_cases import ImageBootstrap
from src.core.probe_freeze import sha256_file


VARIANTS = ("all", "sign_consistent")
A2_MODELS = ("base", "minimum", "mean", "soft_and_calibrated", "soft_and_raw", "free")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def to_logit(probability, clip):
    probability = probability.clamp(clip, 1 - clip)
    return torch.log(probability) - torch.log1p(-probability)


def sign_consistent(margin, prediction):
    return (margin > 0 and prediction == "Yes") or (margin < 0 and prediction == "No")


def one_negative(items):
    return [item for item in items if item["gt"] == "No" and item["labels"].count("N") == 1]


def scores(items, column, clip, calibration=None):
    """입력별 z_i logit[n, 3]와 클래스별 표준화 점수(N·P 슬롯을 합쳐 평균 0, 표준편차 1)."""
    probability = torch.stack([item["probabilities"][:, column] for item in items])
    logits = to_logit(probability, clip)
    classes = [item["class"] for item in items]
    standardized = torch.empty_like(logits)
    for name in set(classes):
        rows = torch.tensor([value == name for value in classes])
        values = logits[rows]
        standardized[rows] = (values - values.mean()) / values.std(unbiased=False)
    positive = torch.tensor([[label == "P" for label in item["labels"]] for item in items])
    require(bool((positive.sum(dim=1) == 2).all()), "Expected exactly one N image per input")
    rows = torch.arange(len(items))
    negative = (~positive).double().argmax(dim=1)
    result = {
        "probability": probability, "logits": logits, "standardized": standardized, "positive": positive,
        "p_N": probability[rows, negative], "z_N": standardized[rows, negative],
        "z_P_mean": standardized.masked_fill(~positive, 0).sum(dim=1) / 2,
        "z_P_mean_raw": logits.masked_fill(~positive, 0).sum(dim=1) / 2,
    }
    if calibration is not None:
        slope = torch.tensor([calibration[name][0] for name in classes], dtype=torch.float64).unsqueeze(1)
        intercept = torch.tensor([calibration[name][1] for name in classes], dtype=torch.float64).unsqueeze(1)
        result["calibrated_logits"] = slope * logits + intercept
    return result


def outcomes(items):
    return (torch.tensor([item["margin"] for item in items], dtype=torch.float64),
            torch.tensor([item["prediction"] == "Yes" for item in items], dtype=torch.float64))


def interval_row(estimate, values, **labels):
    low, high, valid = percentile(values)
    return {**labels, "estimate": float(estimate), "ci_low": low, "ci_high": high, "ci_valid": valid}


# ---------- A1 ----------
def a1(items, sc, y, weights, chunk, layer):
    x, names = design(items, sc, ("class", "n_pos", "z_N", "z_P_mean"))
    beta, _, boot, _ = fit_all(x, y, weights, chunk)
    n, p = names.index("z_N"), names.index("z_P_mean")
    return [interval_row(beta[i] - (beta[j] if j is not None else 0),
                         boot[:, i] - (boot[:, j] if j is not None else 0), layer=layer, term=term, n=len(items))
            for term, i, j in (("beta_N", n, None), ("beta_P", p, None), ("beta_P_minus_beta_N", p, n))]


# ---------- A2 ----------
def platt(logits, labels, iterations=200):
    """클래스별 Platt 보정 a·logit + b (정규화 없음, LBFGS)."""
    a = torch.ones(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([a, b], lr=1, max_iter=iterations, tolerance_grad=1e-10, tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(a * logits + b, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(a.detach()), float(b.detach())


def fit_calibration(step3_dir, layers, clip):
    """STEP 3 OOF 문맥 벡터(vectors.csv)로만 클래스·레이어별 Platt 보정을 맞춘다."""
    rows = read_csv(step3_dir / "vectors.csv")
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)
    calibration, logits_all, labels_all, classes_all = {}, {}, None, []
    for layer in layers:
        calibration[layer] = {}
        logits, labels, classes = [], [], []
        for name, members in sorted(by_class.items()):
            x = to_logit(torch.tensor([float(row[f"p_layer{layer}"]) for row in members], dtype=torch.float64), clip)
            target = torch.tensor([row["label"] == "P" for row in members], dtype=torch.float64)
            calibration[layer][name] = platt(x, target)
            logits.append(x)
            labels.append(target)
            classes += [name] * len(members)
        logits_all[layer] = torch.cat(logits)
        labels_all = torch.cat(labels)
        classes_all = classes
    return calibration, {"logits": logits_all, "labels": labels_all, "classes": classes_all, "n_vectors": len(rows)}


def reliability(probability, labels, bins):
    index = (probability * bins).long().clamp(max=bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        mask = index == b
        n = int(mask.sum())
        if not n:
            continue
        confidence, observed = float(probability[mask].mean()), float(labels[mask].mean())
        ece += n / len(probability) * abs(observed - confidence)
        rows.append({"bin": b, "low": b / bins, "high": (b + 1) / bins, "n": n,
                     "mean_predicted": confidence, "observed_P_rate": observed})
    return ece, rows


def calibration_report(calibration, step3, sample_scores, layers, bins):
    rows, curves = [], []
    for column, layer in enumerate(layers):
        x3, labels3 = step3["logits"][layer], step3["labels"]
        slope = torch.tensor([calibration[layer][name][0] for name in step3["classes"]], dtype=torch.float64)
        intercept = torch.tensor([calibration[layer][name][1] for name in step3["classes"]], dtype=torch.float64)
        sc = sample_scores[layer]
        labels4 = sc["positive"].flatten().double()
        for source, logits, labels in (
                ("step3_oof_raw", x3, labels3), ("step3_oof_calibrated_in_sample", slope * x3 + intercept, labels3),
                ("step4_one_negative_raw", sc["logits"].flatten(), labels4),
                ("step4_one_negative_calibrated", sc["calibrated_logits"].flatten(), labels4)):
            ece, curve = reliability(torch.sigmoid(logits), labels, bins)
            rows.append({"layer": layer, "source": source, "n": len(labels), "ece": ece})
            curves += [{"layer": layer, "source": source, **row} for row in curve]
    return rows, curves


def assign_folds(items, folds, seed):
    by_class = defaultdict(set)
    for item in items:
        by_class[item["class"]].add(item["pair_id"])
    fold_of = {}
    for name, pairs in sorted(by_class.items()):
        ordered = sorted(pairs)
        random.Random(f"{seed}:{name}").shuffle(ordered)
        fold_of.update({pair: index % folds for index, pair in enumerate(ordered)})
    fold = torch.tensor([fold_of[item["pair_id"]] for item in items])
    owner = {}
    for item, value in zip(items, fold.tolist()):
        for image_id in item["image_ids"]:
            require(owner.setdefault(image_id, value) == value, f"Image {image_id} appears in two CV folds")
    return fold


def aggregators(sc):
    s = sc["standardized"]
    return {
        "base": [],
        "minimum": [s.min(dim=1).values],
        "mean": [s.mean(dim=1)],
        "soft_and_calibrated": [F.logsigmoid(sc["calibrated_logits"]).sum(dim=1)],
        "soft_and_raw": [F.logsigmoid(sc["logits"]).sum(dim=1)],
        "free": [sc["z_N"], sc["z_P_mean"]],
    }


def cross_validate(x, y, fy, fold, c):
    margin_hat, fy_hat = torch.empty_like(y), torch.empty_like(y)
    for k in fold.unique().tolist():
        train, test = fold != k, fold == k
        beta = torch.linalg.lstsq(x[train], y[train].unsqueeze(-1)).solution.squeeze(-1)
        margin_hat[test] = x[test] @ beta
        fy_hat[test] = predict(fit_logistic(x[train], fy[train], c=c, class_weight=None), x[test])
    return margin_hat, fy_hat


def heldout(y, fy, margin_hat, fy_hat, weights):
    total = weights.sum(dim=1)
    mean = (weights * y).sum(dim=1) / total
    r2 = 1 - (weights * (y - margin_hat) ** 2).sum(dim=1) / (weights * (y.unsqueeze(0) - mean.unsqueeze(1)) ** 2).sum(dim=1)
    p = fy_hat.clamp(1e-12, 1 - 1e-12)
    log_loss = -(weights * (fy * torch.log(p) + (1 - fy) * torch.log1p(-p))).sum(dim=1) / total
    return r2, log_loss


def a2(items, sc, y, fy, weights, settings, layer):
    fold = assign_folds(items, settings["folds"], settings["seed"])
    controls, _ = design(items, sc, ("class", "n_pos"))
    ones = torch.ones(1, len(y), dtype=torch.float64)
    results = {}
    for name, columns in aggregators(sc).items():
        x = torch.cat([controls, *[column.unsqueeze(1) for column in columns]], dim=1)
        margin_hat, fy_hat = cross_validate(x, y, fy, fold, settings["logistic_c"])
        observed = heldout(y, fy, margin_hat, fy_hat, ones)
        results[name] = {"observed": [float(value[0]) for value in observed], "boot": heldout(y, fy, margin_hat, fy_hat, weights),
                         "mse": float(((y - margin_hat) ** 2).mean()), "brier": float(((fy - fy_hat) ** 2).mean()),
                         "auroc": auroc(fy_hat, fy.bool())}
    rows = []
    reference = results["minimum"]
    for name, result in results.items():
        row = {"layer": layer, "model": name, "n": len(items), "folds": settings["folds"],
               "margin_r2": result["observed"][0], "margin_mse": result["mse"],
               "fy_log_loss": result["observed"][1], "fy_brier": result["brier"], "fy_auroc": result["auroc"]}
        for index, metric in enumerate(("margin_r2", "fy_log_loss")):
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"], _ = percentile(result["boot"][index])
            difference = result["boot"][index] - reference["boot"][index]
            row[f"{metric}_minus_minimum"] = result["observed"][index] - reference["observed"][index]
            row[f"{metric}_minus_minimum_ci_low"], row[f"{metric}_minus_minimum_ci_high"], _ = percentile(difference)
        rows.append(row)
    return rows


# ---------- A3 ----------
def a3(items, sc, y, weights, chunk, layer):
    terms = dict(sc)
    names = []
    for k in range(3):
        for role, mask in (("P", sc["positive"][:, k]), ("N", ~sc["positive"][:, k])):
            terms[f"{role}_pos{k + 1}"] = sc["standardized"][:, k] * mask.double()
            names.append(f"{role}_pos{k + 1}")
    x, columns = design(items, terms, ("class", "n_pos", *names))
    beta, _, boot, _ = fit_all(x, y, weights, chunk)
    return [interval_row(beta[columns.index(name)], boot[:, columns.index(name)], layer=layer, role=name[0],
                         position=int(name[-1]), term=name, n=len(items),
                         n_nonzero=int((terms[name] != 0).sum())) for name in names]


# ---------- A4 / A5 ----------
def tercile_masks(items, values):
    strata = defaultdict(list)
    for index, item in enumerate(items):
        strata[(item["class"], item["labels"].index("N") + 1)].append(index)
    top, bottom, members = [], [], []
    for _, indices in sorted(strata.items()):
        m = len(indices) // 3
        if not m:
            continue
        order = sorted(indices, key=lambda i: (float(values[i]), i))
        for target, chosen in ((bottom, order[:m]), (top, order[-m:]), (members, indices)):
            mask = torch.zeros(len(items), dtype=torch.float64)
            mask[chosen] = 1
            target.append(mask)
    return torch.stack(top), torch.stack(bottom), torch.stack(members)


def tercile_contrast(outcome, top, bottom, members, weights):
    def rate(mask):
        total = weights @ mask.T
        return (weights * outcome) @ mask.T / total.clamp_min(1e-300), total

    top_rate, top_total = rate(top)
    bottom_rate, bottom_total = rate(bottom)
    size = weights @ members.T
    valid = (top_total > 0) & (bottom_total > 0)
    size = torch.where(valid, size, torch.zeros_like(size))
    norm = size.sum(dim=1)

    def pool(values):
        return (size * torch.where(valid, values, torch.zeros_like(values))).sum(dim=1) / norm

    return pool(top_rate), pool(bottom_rate), pool(top_rate - bottom_rate)


def terciles(items, sc, y, fy, weights, analysis, **labels):
    top, bottom, members = tercile_masks(items, sc["z_P_mean_raw"])
    ones = torch.ones(1, len(items), dtype=torch.float64)
    rows = []
    for outcome_name, outcome in (("false_yes", fy), ("margin", y)):
        observed = [float(value[0]) for value in tercile_contrast(outcome, top, bottom, members, ones)]
        boot = tercile_contrast(outcome, top, bottom, members, weights)
        row = {"analysis": analysis, **labels, "outcome": outcome_name, "n_inputs": len(items),
               "n_strata": len(members), "n_per_group": int(top.sum()), "top_tercile": observed[0],
               "bottom_tercile": observed[1], "difference": observed[2]}
        row["ci_low"], row["ci_high"], row["ci_valid"] = percentile(boot[2])
        rows.append(row)
    return rows


def a5(items, sc, y, fy, weights, thresholds, layer):
    rows, rates = [], []
    for threshold in thresholds:
        keep = sc["p_N"] < threshold
        subset = [item for item, flag in zip(items, keep.tolist()) if flag]
        sub = {key: value[keep] for key, value in sc.items()}
        w = weights[:, keep]
        rows += terciles(subset, sub, y[keep], fy[keep], w, "A5", layer=layer, threshold=threshold)
        boot = (w * fy[keep]).sum(dim=1) / w.sum(dim=1)
        rates.append(interval_row(fy[keep].mean(), boot, layer=layer, threshold=threshold, n_inputs=int(keep.sum()),
                                  n_false_yes=int(fy[keep].sum()), quantity="false_yes_rate"))
    return rows, rates


# ---------- A6 ----------
def average_rank(values):
    unique, inverse, counts = torch.unique(values, return_inverse=True, return_counts=True)
    end = counts.cumsum(0).double()
    return ((end - counts.double() + 1 + end) / 2)[inverse]


def huber_fit(x, y, weights, k, scale, iterations):
    beta, _ = weighted_fit(x, y, weights)
    for _ in range(iterations):
        residual = y.unsqueeze(0) - beta @ x.T
        robust = (k * scale / residual.abs().clamp_min(1e-12)).clamp(max=1)
        beta, _ = weighted_fit(x, y, weights * robust)
    return beta


def weighted_partial_correlation(x, first, second, weights):
    def residual(values):
        beta, _ = weighted_fit(x, values, weights)
        return values.unsqueeze(0) - beta @ x.T

    a, b = residual(first), residual(second)
    return (weights * a * b).sum(dim=1) / torch.sqrt((weights * a * a).sum(dim=1) * (weights * b * b).sum(dim=1))


def a6(items, sc, y, weights, settings, chunk, layer):
    rows = []
    x, names = design(items, sc, ("class", "n_pos", "z_N", "z_P_mean"))
    p = names.index("z_P_mean")
    controls, _ = design(items, sc, ("class", "n_pos", "z_N"))
    residual = sc["z_P_mean"] - controls @ torch.linalg.lstsq(controls, sc["z_P_mean"].unsqueeze(-1)).solution.squeeze(-1)
    keep = residual.abs() <= settings["exclusion_sd"] * residual.std()
    beta, _, boot, _ = fit_all(x[keep], y[keep], weights[:, keep], chunk)
    rows.append(interval_row(beta[p], boot[:, p], layer=layer, method="exclude_z_P_residual_outliers",
                             quantity="beta_P", n=int(keep.sum()), excluded=int((~keep).sum())))

    ols = torch.linalg.lstsq(x, y.unsqueeze(-1)).solution.squeeze(-1)
    resid = y - x @ ols
    scale = float(1.4826 * (resid - resid.median()).abs().median())
    ones = torch.ones(1, len(y), dtype=torch.float64)
    observed = huber_fit(x, y, ones, settings["huber_k"], scale, settings["huber_iterations"])[0]
    boot = torch.cat([huber_fit(x, y, weights[start:start + chunk], settings["huber_k"], scale,
                                settings["huber_iterations"]) for start in range(0, len(weights), chunk)])
    rows.append(interval_row(observed[p], boot[:, p], layer=layer, method="huber", quantity="beta_P", n=len(items),
                             excluded=0, scale=scale))

    ranks = {"rank_z_N": average_rank(sc["z_N"])}
    rank_controls, _ = design(items, ranks, ("class", "n_pos", "rank_z_N"))
    first, second = average_rank(y), average_rank(sc["z_P_mean"])
    observed = weighted_partial_correlation(rank_controls, first, second, ones)[0]
    boot = torch.cat([weighted_partial_correlation(rank_controls, first, second, weights[start:start + chunk])
                      for start in range(0, len(weights), chunk)])
    rows.append(interval_row(observed, boot, layer=layer, method="partial_spearman", quantity="rho_margin_z_P",
                             n=len(items), excluded=0))
    return rows


# ---------- A7 ----------
def read_step3_inputs(step3_dir, step2_dir):
    records = list(read_jsonl(step2_dir / "manifest.jsonl"))
    by_record = defaultdict(list)
    rows = read_csv(step3_dir / "slots.csv")
    layers = sorted(int(name.rsplit("layer", 1)[1]) for name in rows[0] if name.startswith("p_layer"))
    for row in rows:
        by_record[int(row["record"])].append(row)
    inputs = {}
    for index, members in sorted(by_record.items()):
        members.sort(key=lambda row: int(row["position"]))
        record = records[index]
        case = record["case"]
        require(record["index"] == index and [int(row["image_id"]) for row in members] == case["image_ids"],
                f"STEP 3 slots and STEP 2 manifest disagree at record {index}")
        item = {"class": case["class"], "pair_id": case["pair_id"], "gt": case["gt"], "labels": case["image_labels"],
                "image_ids": case["image_ids"], "margin": record["logit_margin"], "prediction": record["prediction"],
                "probabilities": torch.tensor([[float(row[f"p_layer{layer}"]) for layer in layers] for row in members],
                                              dtype=torch.float64)}
        key = (case["class"], tuple(case["image_ids"]))
        if key in inputs:
            require(inputs[key]["margin"] == item["margin"] and inputs[key]["prediction"] == item["prediction"],
                    f"Records sharing an input disagree: {key}")
            require(torch.allclose(inputs[key]["probabilities"], item["probabilities"]),
                    f"OOF probabilities differ within an input: {key}")
            continue
        inputs[key] = item
    return list(inputs.values()), layers


def a7(items, sc, y, weights, chunk, layer):
    negatives = [(item["class"], item["image_ids"][item["labels"].index("N")]) for item in items]
    groups = sorted(set(negatives))
    dummies = torch.tensor([[value == group for group in groups] for value in negatives], dtype=torch.float64)
    base, _ = design(items, sc, ("n_pos",))
    x = torch.cat([dummies, base, sc["z_N"].unsqueeze(1), sc["z_P_mean"].unsqueeze(1)], dim=1)
    beta, _, boot, _ = fit_all(x, y, weights, chunk)
    psets = defaultdict(set)
    for item, negative in zip(items, negatives):
        psets[negative].add(frozenset(i for i, label in zip(item["image_ids"], item["labels"]) if label == "P"))
    within = sum(float(((sc["z_P_mean"][idx] - sc["z_P_mean"][idx].mean()) ** 2).sum())
                 for idx in [[i for i, value in enumerate(negatives) if value == group] for group in groups])
    total = float(((sc["z_P_mean"] - sc["z_P_mean"].mean()) ** 2).sum())
    detail = {"n_inputs": len(items), "n_negative_images": len(groups),
              "n_negative_images_with_2plus_P_sets": sum(len(value) > 1 for value in psets.values()),
              "within_negative_image_share_of_z_P_variance": within / total}
    rows = [interval_row(beta[-1], boot[:, -1], layer=layer, model="negative_image_FE", term="beta_P", **detail),
            interval_row(beta[-2], boot[:, -2], layer=layer, model="negative_image_FE", term="beta_N", **detail)]
    x, names = design(items, sc, ("class", "n_pos", "z_N", "z_P_mean"))
    beta, _, boot, _ = fit_all(x, y, weights, chunk)
    rows.append(interval_row(beta[names.index("z_P_mean")], boot[:, names.index("z_P_mean")], layer=layer,
                             model="class_FE_between", term="beta_P", **detail))
    return rows


# ---------- 실행 ----------
def precheck(items, records_label):
    flags = [sign_consistent(item["margin"], item["prediction"]) for item in items]
    kinds = defaultdict(int)
    for item, flag in zip(items, flags):
        if not flag:
            kinds["tie_margin_0" if item["margin"] == 0 else
                  "unparsed" if item["prediction"] not in ("Yes", "No") else "sign_disagrees"] += 1
    return {"data": records_label, "n_inputs": len(items), "n_inconsistent": len(items) - sum(flags),
            "by_kind": dict(kinds)}, flags


def run_followup(config):
    from src.common.paths import project_path

    step4_dir, step3_dir, step2_dir = (project_path(config[key]).resolve()
                                      for key in ("step4_dir", "step3_dir", "step2_run_dir"))
    clip, layer_main, statistics = config["logit_clip"], config["primary_layer"], config["statistics"]
    reps, seed, chunk = statistics["reps"], statistics["seed"], statistics["chunk"]
    inputs, layers = read_inputs(step4_dir)
    require(layer_main in layers, f"Primary layer {layer_main} missing")
    sample = one_negative(inputs)
    bootstrap = ImageBootstrap([i for item in inputs for i in item["image_ids"]], reps, seed)
    calibration, step3_oof = fit_calibration(step3_dir, layers, clip)
    step3_inputs, step3_layers = read_step3_inputs(step3_dir, step2_dir)
    require(step3_layers == layers, "STEP 3 and STEP 4 layers differ")
    step3_sample = one_negative(step3_inputs)
    bootstrap3 = ImageBootstrap([i for item in step3_inputs for i in item["image_ids"]], reps, seed)
    check4, flags4 = precheck(sample, "STEP 4 one-N unique inputs")
    check3, flags3 = precheck(step3_sample, "STEP 2/3 one-N unique inputs (A7)")

    results = {}
    for variant in VARIANTS:
        items = sample if variant == "all" else [item for item, flag in zip(sample, flags4) if flag]
        items3 = step3_sample if variant == "all" else [item for item, flag in zip(step3_sample, flags3) if flag]
        y, fy = outcomes(items)
        weights = bootstrap.case_weights([item["image_ids"] for item in items])
        y3, _ = outcomes(items3)
        weights3 = bootstrap3.case_weights([item["image_ids"] for item in items3])
        out = defaultdict(list)
        per_layer = {}
        for column, layer in enumerate(layers):
            sc = scores(items, column, clip, calibration[layer])
            per_layer[layer] = sc
            out["a1"] += a1(items, sc, y, weights, chunk, layer)
            out["a2"] += a2(items, sc, y, fy, weights, config["a2"], layer)
            out["a3"] += a3(items, sc, y, weights, chunk, layer)
        column = layers.index(layer_main)
        sc = per_layer[layer_main]
        out["a4"] = terciles(items, sc, y, fy, weights, "A4", layer=layer_main, threshold=None)
        out["a5"], out["a5_rates"] = a5(items, sc, y, fy, weights, config["a5"]["thresholds"], layer_main)
        out["a6"] = a6(items, sc, y, weights, config["a6"], chunk, layer_main)
        sc3 = scores(items3, column, clip)
        out["a7"] = a7(items3, sc3, y3, weights3, chunk, layer_main)
        out["n_inputs"], out["n_inputs_a7"] = len(items), len(items3)
        if variant == "all":
            out["calibration"], out["reliability"] = calibration_report(
                calibration, step3_oof, per_layer, layers, config["a2"]["ece_bins"])
        results[variant] = out
    report = {
        "status": config["status"], "note": config["note"], "sample": config["sample"],
        "primary_layer": layer_main, "layers": layers, "settings": {k: config[k] for k in (
            "logit_clip", "standardization", "sign_check", "a2", "a3", "a4", "a5", "a6", "a7", "statistics")},
        "limitations": config["limitations"],
        "inputs_sha256": {
            "step4/zi_hq_slots.csv": sha256_file(step4_dir / "zi_hq_slots.csv"),
            "step4/extraction/manifest.jsonl": sha256_file(step4_dir / "extraction" / "manifest.jsonl"),
            "step3/vectors.csv": sha256_file(step3_dir / "vectors.csv"),
            "step3/slots.csv": sha256_file(step3_dir / "slots.csv"),
            "step2/manifest.jsonl": sha256_file(step2_dir / "manifest.jsonl"),
        },
        "precheck": [check4, check3],
        "calibration_parameters": {layer: {name: {"slope": a, "intercept": b} for name, (a, b) in values.items()}
                                   for layer, values in calibration.items()},
        "counts": {"step4_unique_inputs": len(inputs), "step4_one_negative": len(sample),
                   "step3_one_negative": len(step3_sample), "step3_oof_vectors": step3_oof["n_vectors"],
                   "bootstrap_images_step4": len(bootstrap.index), "bootstrap_images_step3": len(bootstrap3.index)},
    }
    return report, results


def summary_rows(results, layer):
    """A1~A7 핵심 수치 한 표: 전체 입력과 부호 일치 입력을 나란히."""
    def pick(variant, section, **match):
        rows = [row for row in results[variant][section] if all(row.get(k) == v for k, v in match.items())]
        require(len(rows) == 1, f"Summary lookup failed: {section} {match}")
        return rows[0]

    specs = [
        ("A1", "beta_P - beta_N (per class-SD)", "a1", {"layer": layer, "term": "beta_P_minus_beta_N"}, "estimate"),
        ("A1", "beta_P", "a1", {"layer": layer, "term": "beta_P"}, "estimate"),
        ("A1", "beta_N", "a1", {"layer": layer, "term": "beta_N"}, "estimate"),
    ]
    for model in ("mean", "soft_and_calibrated", "free"):
        specs.append(("A2", f"held-out margin R2: {model} - minimum", "a2", {"layer": layer, "model": model},
                      "margin_r2_minus_minimum"))
        specs.append(("A2", f"held-out False-Yes log loss: {model} - minimum", "a2", {"layer": layer, "model": model},
                      "fy_log_loss_minus_minimum"))
    for model in ("minimum", "free"):
        specs.append(("A2", f"held-out margin R2: {model}", "a2", {"layer": layer, "model": model}, "margin_r2"))
    for position in (1, 2, 3):
        for role in ("P", "N"):
            specs.append(("A3", f"{role} slot coefficient, position {position}", "a3",
                          {"layer": layer, "term": f"{role}_pos{position}"}, "estimate"))
    specs += [
        ("A4", "False-Yes rate: top - bottom z_P tercile", "a4", {"outcome": "false_yes"}, "difference"),
        ("A4", "margin: top - bottom z_P tercile", "a4", {"outcome": "margin"}, "difference"),
    ]
    for threshold in (0.1, 0.05, 0.2):
        specs.append(("A5", f"p_N < {threshold}: False-Yes rate in subset", "a5_rates", {"threshold": threshold},
                      "estimate"))
        specs.append(("A5", f"p_N < {threshold}: False-Yes top - bottom tercile", "a5",
                      {"threshold": threshold, "outcome": "false_yes"}, "difference"))
    specs += [
        ("A6", "beta_P, z_P residual outliers excluded", "a6", {"method": "exclude_z_P_residual_outliers"}, "estimate"),
        ("A6", "beta_P, Huber", "a6", {"method": "huber"}, "estimate"),
        ("A6", "partial Spearman rho(margin, z_P)", "a6", {"method": "partial_spearman"}, "estimate"),
        ("A7", "STEP 3 beta_P with N-image fixed effects", "a7", {"model": "negative_image_FE", "term": "beta_P"},
         "estimate"),
        ("A7", "STEP 3 beta_N with N-image fixed effects", "a7", {"model": "negative_image_FE", "term": "beta_N"},
         "estimate"),
        ("A7", "STEP 3 beta_P, class FE (between N images)", "a7", {"model": "class_FE_between", "term": "beta_P"},
         "estimate"),
    ]
    rows = []
    for analysis, quantity, section, match, field in specs:
        row = {"analysis": analysis, "quantity": quantity, "layer": layer}
        for variant in VARIANTS:
            source = pick(variant, section, **match)
            prefix = "all" if variant == "all" else "consistent"
            low_key, high_key = (("ci_low", "ci_high") if field in ("estimate", "difference")
                                 else (f"{field}_ci_low", f"{field}_ci_high"))
            n = source.get("n", source.get("n_inputs"))
            row.update({f"{prefix}_n": n, f"{prefix}_estimate": source[field],
                        f"{prefix}_ci_low": source.get(low_key), f"{prefix}_ci_high": source.get(high_key)})
        rows.append(row)
    return rows
