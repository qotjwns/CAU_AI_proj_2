"""STEP 5-2 탐색 분석 (CPU, exploratory; 사전 등록 판정과 무관).

5-2 저장값(본 추출·비전 fp32 추출), 5-2 평가 출력, STEP 4 고정 Probe/readout, 모델 임베딩 행, COCO-REM 주석만 읽는다.
- z_n_by_a: N-last에서 N 앞의 P 수 a에 따른 z_N 변화 (기준 a = 0: 선행 이미지 없는 N), bf16 잡음 기준선
- p_control_by_a: 같은 위치·같은 선행 P를 둔 P 이미지의 Probe logit 변화 (z_N 변화의 대조군)
- n_influence: 장수별 margin의 z_N·z_P 기울기와 AUROC
- probe_and / position_accuracy: 장수별 Probe AND vs VLM, 위치별 P 이미지 Probe 정확도
- yes_gap: 장수별 (All-P Yes 비율 − one-N Yes 비율)
- posthoc_placement: 배치 상호작용·순서 효과 (결과를 본 뒤 정한 대비)
- precision_5_2b: 5-2b H1b·H2b 예상 CI 반폭 (5-2 분산 대리 계산)
- readout_geometry: h_q 위치 readout 방향의 기하와 L36 Yes−No 방향 정렬 (직교 투영 사전 점검)
- step4_position_rates, bbox: STEP 4 N 위치별 False-Yes, High/Low P 이미지의 bbox 면적
"""
from collections import defaultdict
import json
import math
import statistics

import torch
from tqdm import tqdm

from src.common.config import read_yaml
from src.common.data import read_json, read_jsonl
from src.common.paths import project_path, project_relative_path
from src.core.feature_validation import inspect_run, load_validated_feature, require
from src.core.probe import auroc
from src.core.probe_cases import ImageBootstrap


LAYERS = (9, 18, 27, 36)
STEPS = (1, 2, 3, 4)
PLACEMENTS = ("N-first", "N-last")
NOTES = {
    "status": "exploratory; not part of any STEP 5-2 decision",
    "posthoc": "Post-hoc contrasts chosen after seeing the STEP 5-2 results; they do not change the preregistered decision.",
    "probe_and_all_positive": ("All-P Probe AND accuracy follows from the design: STEP 5-2 P images were selected as "
                               "probe-positive (frozen z_i p(P) >= 0.5 at L36, position 1); not an unselected-P estimate."),
    "strength_and_z_p": ("H2 (High - Low) and z_P associations are not distinguished from object size and salience: "
                         "High-P images have larger target boxes (see bbox.csv)."),
    "z_n_by_a": ("Increasing a changes both the preceding image content and N's absolute token position (rotary); "
                 "the two are not separated. The P-image control shares the same preceding images and position."),
    "n_influence": "z_N varies in a narrow range (all N images read as N), which limits its slope and AUROC.",
    "readout_geometry": ("Readouts are the frozen STEP 4 h_q readouts (positions 1-3, trained on 3-image inputs), "
                         "not the STEP 5 decision readout s_i = f(h_q, z_i); L36 alignment uses the exact Yes - No "
                         "unembedding direction, L27 leverage needs a forward pass (GPU)."),
    "precision_5_2b": ("Proxy unit: within-series first minus last position, equal weight over 3-5 images; "
                       "conservative because the 5-2b middle mean averages 1-3 positions."),
    "vision_fp32_main_results": "P4 - P1 on the vision-fp32 subset is in the STEP 5-2 evaluation (vision_fp32.csv).",
}


# ---------------------------------------------------------------- 데이터

def probe_logits(probes, cls, features):
    """features [L, n, D] → 고정 z_i Probe logit [L, n]."""
    out = torch.empty(features.shape[:2], dtype=torch.float64)
    for i, layer in enumerate(LAYERS):
        probe = probes[(cls, layer)]
        out[i] = ((features[i].double() - probe["mean"]) / probe["scale"]) @ probe["weight"] + probe["bias"]
    return out


def load_run(run_dir, probes, tolerance):
    metadata, records, checks, _ = inspect_run(run_dir)
    require(all(checks.values()), f"Invalid extraction folder: {run_dir}")
    data = {}
    for item in tqdm(records, desc=f"Load {run_dir.name}", unit="record"):
        payload, _ = load_validated_feature(run_dir, item, metadata, tolerance["atol"], tolerance["rtol"],
                                            expected_images=None)
        case = item["case"]
        entry = {"case": case, "prediction": item["prediction"], "margin": item["fp32_variant_logit_margin"],
                 "logits": probe_logits(probes, case["class"], payload["image_features"])}
        if case["condition"] == "one_negative":
            entry["z_n"] = payload["image_features"][:, case["negative_positions"][0]].double().clone()
        data[case["case_id"]] = entry
        del payload
    return data


def index(data):
    cells, meta = {}, {}
    for entry in data.values():
        case = entry["case"]
        cells[(case["condition"], case["pair_id"], case["placement"], case["series_step"])] = entry
        meta[case["pair_id"]] = (case["anchor_negative_id"], case["class"], case["strength"])
    return cells, meta


class Clusters:
    """N 이미지 군집과 클래스 군집 부트스트랩 (같은 반복 수·seed)."""

    def __init__(self, meta, reps, seed):
        self.meta = meta
        self.anchor = ImageBootstrap(sorted({m[0] for m in meta.values()}), reps, seed)
        self.cls = ImageBootstrap(sorted({m[1] for m in meta.values()}), reps, seed)

    def interval(self, bootstrap, keys, values, level=0.95):
        weights = bootstrap.unit_weights(keys)
        totals = weights.sum(dim=1)
        valid = totals > 0
        estimates = weights[valid] @ torch.tensor(values, dtype=torch.float64) / totals[valid]
        tail = (1 - level) / 2
        low, high = torch.quantile(estimates, torch.tensor([tail, 1 - tail], dtype=torch.float64))
        return float(low), float(high)

    def summary(self, series, values, level=0.95, both=True):
        """series: 계열 이름 목록 (값과 같은 순서)."""
        if not values:
            return {"n": 0, "estimate": None}
        row = {"n": len(values), "estimate": math.fsum(values) / len(values)}
        row["ci_low"], row["ci_high"] = self.interval(self.anchor, [self.meta[s][0] for s in series], values, level)
        if both:
            row["class_ci_low"], row["class_ci_high"] = self.interval(self.cls, [self.meta[s][1] for s in series],
                                                                      values, level)
        return row


def sigmoid(value):
    return float(torch.sigmoid(torch.tensor(value, dtype=torch.float64)))


def cosine_distance(first, second):
    return (1 - torch.nn.functional.cosine_similarity(first, second, dim=-1)).tolist()


# ---------------------------------------------------------------- 분석

def z_n_by_a(cells, meta, clusters, label):
    """N-last k = a의 z_N을 같은 계열 N-first 2장 입력(a = 0)의 z_N과 비교. 잡음 기준선: N-first k = 2..4 vs k = 1."""
    names = sorted(s for s in meta if ("one_negative", s, "N-first", 1) in cells)
    rows = []
    for i, layer in enumerate(LAYERS):
        noise = [cosine_distance(cells[("one_negative", s, "N-first", k)]["z_n"][i],
                                 cells[("one_negative", s, "N-first", 1)]["z_n"][i])
                 for s in names for k in (2, 3, 4) if ("one_negative", s, "N-first", k) in cells]
        rows.append({"run": label, "layer": layer, "a": 0, "kind": "bf16 noise floor (same prefix, N-first k=2..4 vs 1)",
                     "n": len(noise), "cos_median": statistics.median(noise), "cos_max": max(noise)})
        for a in STEPS:
            used = [s for s in names if ("one_negative", s, "N-last", a) in cells]
            reference = [cells[("one_negative", s, "N-first", 1)] for s in used]
            current = [cells[("one_negative", s, "N-last", a)] for s in used]
            distance = [cosine_distance(c["z_n"][i], r["z_n"][i]) for c, r in zip(current, reference)]
            logit = [c["logits"][i, a].item() - r["logits"][i, 0].item() for c, r in zip(current, reference)]
            prob = [sigmoid(c["logits"][i, a].item()) - sigmoid(r["logits"][i, 0].item()) for c, r in zip(current, reference)]
            row = {"run": label, "layer": layer, "a": a, "kind": "N-last k=a vs a=0", "n": len(used),
                   "cos_median": statistics.median(distance), "cos_mean": statistics.fmean(distance),
                   "cos_max": max(distance)}
            logit_summary = clusters.summary(used, logit)
            prob_summary = clusters.summary(used, prob)
            row.update({f"logit_{k}": v for k, v in logit_summary.items() if k != "n"})
            row.update({f"p_{k}": v for k, v in prob_summary.items() if k != "n"})
            rows.append(row)
    return rows


def p_control_by_a(cells, meta, clusters):
    """대조군: P(a+1)을 선행 P1..Pa 뒤 위치 a+1(All-P N-last k=a)과 단독 위치 1(All-P N-first k=a)에서 비교.
    N-last k = a의 N과 선행 이미지·위치가 같다. N 변화 − P 변화도 계열 안 짝으로 낸다."""
    names = sorted(meta)
    rows = []
    for i, layer in enumerate(LAYERS):
        for a in STEPS:
            p_change, n_change = [], []
            for s in names:
                alone, after = cells[("all_positive", s, "N-first", a)], cells[("all_positive", s, "N-last", a)]
                require(alone["case"]["image_ids"][0] == after["case"]["image_ids"][a], f"{s}: control image differs")
                p_change.append(after["logits"][i, a].item() - alone["logits"][i, 0].item())
                n_change.append(cells[("one_negative", s, "N-last", a)]["logits"][i, a].item()
                                - cells[("one_negative", s, "N-first", 1)]["logits"][i, 0].item())
            for kind, values in (("P control: P(a+1) after P1..Pa vs alone", p_change),
                                 ("N: N after P1..Pa vs alone", n_change),
                                 ("N minus P control (paired)", [n - p for n, p in zip(n_change, p_change)])):
                rows.append({"layer": layer, "a": a, "kind": kind, **clusters.summary(names, values)})
    return rows


def n_influence(data, clusters, layer):
    """one-N, 배치 × 장수: margin의 z_N·z_P(평균) 기울기 (클래스 안 중심화, 1 SD당)와 False-Yes AUROC."""
    column = LAYERS.index(layer)
    groups = defaultdict(list)
    for entry in data.values():
        case = entry["case"]
        if case["condition"] == "one_negative":
            groups[(case["placement"], case["n_images"])].append(entry)
    rows = []
    for (placement, n), items in sorted(groups.items()):
        classes = [e["case"]["class"] for e in items]
        z_n = [e["logits"][column, e["case"]["negative_positions"][0]].item() for e in items]
        z_p = [statistics.fmean(e["logits"][column, j].item() for j, label in enumerate(e["case"]["image_labels"])
                                if label == "P") for e in items]
        margin = [e["margin"] for e in items]
        false_yes = [e["prediction"] == "Yes" for e in items]

        def centred(values):
            groups_ = defaultdict(list)
            for name, value in zip(classes, values):
                groups_[name].append(value)
            means = {name: statistics.fmean(vals) for name, vals in groups_.items()}
            return torch.tensor([value - means[name] for name, value in zip(classes, values)], dtype=torch.float64)

        y = centred(margin)
        keys = [e["case"]["anchor_negative_id"] for e in items]
        for name, raw in (("z_N", z_n), ("z_P_mean", z_p)):
            x = centred(raw)
            sd = float(x.std())
            weights = clusters.anchor.unit_weights(keys)
            slopes = (weights * x * y).sum(1) / (weights * x * x).sum(1) * sd
            low, high = torch.quantile(slopes, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
            rows.append({"placement": placement, "n_images": n, "evidence": name, "n": len(items),
                         "margin_per_sd": float((x * y).sum() / (x * x).sum()) * sd, "ci_low": low, "ci_high": high,
                         "auroc_false_yes": auroc(raw, false_yes)})
    return rows


def probe_and(data, clusters, layer):
    column = LAYERS.index(layer)
    groups = defaultdict(list)
    positions = defaultdict(lambda: [0, 0])
    for entry in data.values():
        case = entry["case"]
        groups[(case["condition"], case["placement"], case["n_images"])].append(entry)
        for position, label in enumerate(case["image_labels"]):
            if label == "P":
                positions[(case["n_images"], position + 1)][0] += entry["logits"][column, position].item() >= 0
                positions[(case["n_images"], position + 1)][1] += 1
    rows = []
    for (condition, placement, n), items in sorted(groups.items()):
        target_yes = condition == "all_positive"
        vlm = [float((e["prediction"] == "Yes") == target_yes) for e in items]
        anded = [float(bool((e["logits"][column] >= 0).all()) == target_yes) for e in items]
        series = [e["case"]["pair_id"] for e in items]
        difference = clusters.summary(series, [a - v for a, v in zip(anded, vlm)])
        row = {"condition": condition, "placement": placement, "n_images": n, "n": len(items),
               "vlm_accuracy": statistics.fmean(vlm), "probe_and_accuracy": statistics.fmean(anded),
               **{f"and_minus_vlm_{k}": v for k, v in difference.items() if k != "n"}}
        if condition == "one_negative":
            fy = [e for e in items if e["prediction"] == "Yes"]
            row["false_yes_with_probe_and_correct"] = (
                sum(not bool((e["logits"][column] >= 0).all()) for e in fy) / len(fy) if fy else None)
        else:
            row["note"] = NOTES["probe_and_all_positive"]
        rows.append(row)
    accuracy = [{"layer": layer, "n_images": n, "position": p, "p_images": total, "probe_accuracy": hit / total}
                for (n, p), (hit, total) in sorted(positions.items())]
    return rows, accuracy


def yes_gap(cells, meta, clusters):
    """계열 × 장수: All-P Yes − one-N Yes (같은 계열·배치·단계 짝). 강도: 합침/High/Low."""
    rows = []
    for placement in PLACEMENTS:
        for strength in ("pooled", "High", "Low"):
            names = sorted(s for s in meta if strength == "pooled" or meta[s][2] == strength)
            for k in STEPS:
                allp = [float(cells[("all_positive", s, placement, k)]["prediction"] == "Yes") for s in names]
                one = [float(cells[("one_negative", s, placement, k)]["prediction"] == "Yes") for s in names]
                rows.append({"placement": placement, "strength": strength, "series_step": k, "n_images": k + 1,
                             "all_positive_yes_rate": statistics.fmean(allp), "one_negative_yes_rate": statistics.fmean(one),
                             **{f"gap_{key}": v for key, v in
                                clusters.summary(names, [x - y for x, y in zip(allp, one)]).items()}})
    return rows


def posthoc_placement(cells, meta, clusters):
    names = sorted(meta)
    rows = []
    for outcome in ("false_yes", "margin"):
        def value(s, placement, k):
            entry = cells[("one_negative", s, placement, k)]
            return float(entry["prediction"] == "Yes") if outcome == "false_yes" else entry["margin"]
        contrasts = []
        for a, b in ((1, 4), (2, 4), (1, 2)):
            contrasts.append((f"interaction (N-last - N-first) of P{b} - P{a}",
                              [(value(s, "N-last", b) - value(s, "N-last", a))
                               - (value(s, "N-first", b) - value(s, "N-first", a)) for s in names]))
        for placement in PLACEMENTS:
            contrasts.append((f"{placement} P4 - P2", [value(s, placement, 4) - value(s, placement, 2) for s in names]))
        for k in STEPS:
            contrasts.append((f"same images, order only: N-first - N-last at {k + 1} images",
                              [value(s, "N-first", k) - value(s, "N-last", k) for s in names]))
        rows += [{"outcome": outcome, "contrast": name, **clusters.summary(names, values)} for name, values in contrasts]
    return rows


def precision_5_2b(cells, meta, clusters, level):
    names = sorted(meta)
    rows = []
    for outcome in ("false_yes", "margin"):
        def value(s, placement, k):
            entry = cells[("one_negative", s, placement, k)]
            return float(entry["prediction"] == "Yes") if outcome == "false_yes" else entry["margin"]
        sets = [("pooled 3-5 images", [statistics.fmean(value(s, "N-first", k) - value(s, "N-last", k) for k in (2, 3, 4))
                                      for s in names])]
        sets += [(f"{k + 1} images", [value(s, "N-first", k) - value(s, "N-last", k) for s in names]) for k in (2, 3, 4)]
        for name, values in sets:
            row = clusters.summary(names, values, level=level)
            rows.append({"outcome": outcome, "proxy": name, "level": level, **row,
                         "half_width": (row["ci_high"] - row["ci_low"]) / 2,
                         "class_half_width": (row["class_ci_high"] - row["class_ci_low"]) / 2})
    return rows


def readout_geometry(hq, model_dir):
    """위치 readout 방향(∂logit/∂h_q = w/scale)의 코사인, 직교 투영 후 크기 비율, L36 Yes−No 방향 정렬."""
    from safetensors import safe_open
    from transformers import AutoTokenizer

    directions = {(r["scope"], r["class"], r["layer"], r["position"]): (r["weight"] / r["scale"]).double()
                  for r in hq["readouts"]}
    index_map = read_json(model_dir / "model.safetensors.index.json")["weight_map"]
    key = next(name for name in index_map if name.endswith("embed_tokens.weight"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    ids = {word: tokenizer.convert_tokens_to_ids(word) for word in ("Yes", "No")}
    with safe_open(str(model_dir / index_map[key]), framework="pt") as source:
        embedding = source.get_slice(key)
        answer = (embedding[ids["Yes"]:ids["Yes"] + 1].double() - embedding[ids["No"]:ids["No"] + 1].double())[0]
    rows = []
    for scope, cls, layer in sorted({(s, c, l) for s, c, l, _ in directions}):
        d = [directions[(scope, cls, layer, p)] for p in (1, 2, 3)]
        cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0))
        for target in range(3):
            others = torch.stack([d[j] for j in range(3) if j != target], dim=1)
            q, _ = torch.linalg.qr(others)
            push = -d[target]
            orthogonal = push - q @ (q.T @ push)
            row = {"scope": scope, "class": cls, "layer": layer, "n_position": target + 1,
                   "cos_with_other_positions": [cos(d[target], d[j]) for j in range(3) if j != target],
                   "norm_ratio_after_projection": float(orthogonal.norm() / push.norm())}
            if layer == LAYERS[-1]:
                row.update(cos_readout_with_yes_minus_no=cos(d[target], answer),
                           margin_per_unit_step_unprojected=float(answer @ push / push.norm()),
                           margin_per_unit_step_orthogonal=float(answer @ orthogonal / orthogonal.norm()),
                           margin_per_unit_step_yes_minus_no=float(answer.norm()))
            rows.append(row)
    return rows, {"answer_token_ids": ids, "embedding_key": key}


def step4_position_rates(run_dir):
    counts = defaultdict(lambda: [0, 0])
    for item in read_jsonl(run_dir / "manifest.jsonl"):
        composition = "".join(item["case"]["image_labels"])
        if composition in ("NPP", "PNP", "PPN"):
            counts[composition][0] += item["prediction"] == "Yes"
            counts[composition][1] += 1
    return [{"composition": c, "false_yes": fy, "records": n, "false_yes_rate": fy / n}
            for c, (fy, n) in sorted(counts.items())]


def bbox(design_path, annotations_path):
    design = read_json(design_path)["design"]
    coco = read_json(annotations_path)
    category = {c["name"]: c["id"] for c in coco["categories"]}
    area = {image["id"]: image["width"] * image["height"] for image in coco["images"]}
    boxes = defaultdict(list)
    for annotation in coco["annotations"]:
        if not annotation.get("iscrowd", 0):
            boxes[(annotation["image_id"], annotation["category_id"])].append(annotation["bbox"][2] * annotation["bbox"][3])
    rows = []
    for group in design.values():
        used = {i for per_n in group["series"].values() for ids in per_n.values() for i in ids}
        for strength in ("High", "Low"):
            for item in group[strength]:
                sizes = boxes[(item["image_id"], category[group["class"]])]
                rows.append({"class": group["class"], "strength": strength, "image_id": item["image_id"],
                             "probe_p": item["probability"], "used_in_series": item["image_id"] in used,
                             "largest_bbox_ratio": max(sizes) / area[item["image_id"]],
                             "total_bbox_ratio": min(sum(sizes) / area[item["image_id"]], 1.0), "instances": len(sizes)})
    return rows


def bbox_summary(rows):
    used = [r for r in rows if r["used_in_series"]]
    out = {}
    for strength in ("High", "Low"):
        sel = [r for r in used if r["strength"] == strength]
        out[strength] = {"images": len(sel),
                         **{f"median_{k}": statistics.median(r[k] for r in sel)
                            for k in ("largest_bbox_ratio", "total_bbox_ratio", "instances", "probe_p")}}
    by_class = defaultdict(dict)
    for r in used:
        by_class[r["class"]].setdefault(r["strength"], []).append(r["largest_bbox_ratio"])
    out["classes_high_larger"] = sum(statistics.median(v["High"]) > statistics.median(v["Low"]) for v in by_class.values())
    out["classes"] = len(by_class)
    return out


def run_exploration(config):
    acc = read_yaml(project_path(config["accumulation_config"]))
    statistics_ = config["statistics"]
    zi_dir = project_path(acc["zi_frozen_dir"])
    probes = {(p["class"], p["layer"]): p
              for p in torch.load(zi_dir / "zi_probes.pt", map_location="cpu", weights_only=True)["probes"]}
    hq = torch.load(zi_dir / "hq_readouts.pt", map_location="cpu", weights_only=True)
    main = load_run(project_path(acc["run_dir"]).resolve(), probes, acc["verification"])
    vision = load_run(project_path(acc["vision_fp32"]["run_dir"]).resolve(), probes, acc["verification"])
    cells, meta = index(main)
    clusters = Clusters(meta, statistics_["reps"], statistics_["seed"])
    subset = {cid: entry for cid, entry in main.items() if cid in vision}
    sub_cells, sub_meta = index(subset)
    vis_cells, vis_meta = index(vision)
    sub_clusters = Clusters(sub_meta, statistics_["reps"], statistics_["seed"])
    tables = {
        "z_n_by_a": (z_n_by_a(cells, meta, clusters, "bf16_main")
                     + z_n_by_a(sub_cells, sub_meta, sub_clusters, "bf16_main_on_fp32_subset")
                     + z_n_by_a(vis_cells, vis_meta, sub_clusters, "vision_fp32")),
        "p_control_by_a": p_control_by_a(cells, meta, clusters),
        "n_influence": n_influence(main, clusters, config["probe_layer"]),
        "yes_gap": yes_gap(cells, meta, clusters),
        "posthoc_placement": posthoc_placement(cells, meta, clusters),
        "precision_5_2b": precision_5_2b(cells, meta, clusters, config["precision_level"]),
        "step4_position_rates": step4_position_rates(project_path(config["step4_run_dir"])),
        "bbox": bbox(project_path(acc["design"]), project_path(config["coco_annotations"])),
    }
    tables["probe_and"], tables["position_accuracy"] = probe_and(main, clusters, config["probe_layer"])
    tables["readout_geometry"], geometry_info = readout_geometry(hq, project_path(config["model_dir"]))
    evaluation = project_path(config["evaluation_dir"])
    summary = {
        "notes": NOTES, "config": config, "geometry": geometry_info, "bbox": bbox_summary(tables["bbox"]),
        "control_layers": config["control_layers"],
        "vision_fp32_main_results": [row for row in json.loads((evaluation / "analysis.json").read_text())
                                     ["sensitivity"]["vision_fp32"]["rows"]],
        "sources": {"run_dir": acc["run_dir"], "vision_run_dir": acc["vision_fp32"]["run_dir"],
                    "evaluation": project_relative_path(evaluation / "analysis.json")},
    }
    return tables, summary
