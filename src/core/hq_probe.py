"""STEP 5 사전 진단: 마지막 입력 토큰 h_q에서 이미지별 P/N을 선형으로 읽을 수 있는지 (CPU).

z_i(이미지 토큰 평균)는 "그 이미지의 객체 정보"를, h_q는 "세 이미지와 질문이 반영된 상태"를 본다.
h_q Probe 성능이 낮아도 정보 부재나 통합 실패를 뜻하지 않고, 높아도 최종 결정에 쓰였다는 뜻은 아니다.
"""
import csv
from collections import defaultdict

import torch
from tqdm import tqdm

from src.common.data import read_json
from src.core.feature_analysis import extraction_signature
from src.core.feature_validation import inspect_run, load_validated_feature, require
from src.core.probe import fit_logistic, predict, score
from src.core.probe_cases import ImageBootstrap, group_comparison, localization
from src.core.probe_data import input_key


POSITION_DEFINITION = "Readout for image position p (1-3) predicts that image's P/N label from h_q of the whole input."
FEATURE_DEFINITION = "h_q = decoder hidden state of the last valid input token (chat template and generation prompt included)."
SHORTCUT_NOTE = (
    "Two-negative inputs are always PNN in this design, so knowing the number of N images reveals position-1 labels "
    "without position information. N-localization accuracy on PPN/PNP/NPP (chance 1/3) is the shortcut-free check."
)
SCOPE_NOTE = (
    "Diagnostic only. h_q sees all three images and the question. Training excludes every input that contains "
    "an image of the held-out fold; evaluated inputs may contain training images in the other positions (mixed). "
    "Low h_q readability does not prove missing information or failed integration; high readability does not "
    "prove the information was used for the final answer."
)


def load_unique_inputs(run_dir, atol, rtol):
    metadata, records, checks, _ = inspect_run(run_dir)
    failed = [name for name, passed in checks.items() if not passed]
    require(not failed, f"Invalid or incomplete extraction folder: {run_dir} ({', '.join(failed)})")
    seen, cases, features = {}, [], []
    for record in records:
        case = record["case"]
        key = input_key(case)
        if key in seen:
            seen[key]["n_records"] += 1
            require(seen[key]["prediction"] == record["prediction"], "Duplicate inputs with different VLM answers")
            continue
        payload, _ = load_validated_feature(run_dir, record, metadata, atol, rtol)
        features.append(payload["query_features"].clone())
        del payload
        entry = {
            "record": record["index"], "mode": case["mode"], "case_id": case["case_id"], "class": case["class"],
            "pair_id": case["pair_id"], "composition": "".join(case["image_labels"]), "gt": case["gt"],
            "prediction": record["prediction"], "logit_margin": record["logit_margin"],
            "image_ids": list(case["image_ids"]), "image_order": list(case["image_order"]),
            "labels": list(case["image_labels"]), "n_records": 1,
        }
        seen[key] = entry
        cases.append(entry)
    for index, entry in enumerate(cases):
        entry["vectors"] = [3 * index + position for position in range(3)]
    return metadata, cases, torch.stack(features)


def zi_predictions(cases, slots_csv, layers):
    """STEP 3 OOF z_i 확률을 같은 (입력, 위치) 인덱스로 가져온다."""
    lookup = {}
    with slots_csv.open(encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            lookup[(row["mode"], row["case_id"], int(row["position"]))] = [float(row[f"p_layer{layer}"]) for layer in layers]
    values = torch.empty(3 * len(cases), len(layers), dtype=torch.float64)
    for case in cases:
        for position in range(3):
            values[case["vectors"][position]] = torch.tensor(lookup[(case["mode"], case["case_id"], position + 1)])
    return values


def cross_fit_hq(cases, features, layers, image_fold, folds, settings, scope):
    """(레이어, 위치, Fold[, 클래스])별 h_q Probe와 OOF 예측."""
    predictions = torch.full((3 * len(cases), len(layers)), float("nan"), dtype=torch.float64)
    rows = []
    groups = sorted({case["class"] for case in cases}) if scope == "per_class" else ["all"]
    for group in groups:
        members = [i for i, case in enumerate(cases) if group == "all" or case["class"] == group]
        for fold in range(folds):
            train = [i for i in members if all(image_fold[image] != fold for image in cases[i]["image_ids"])]
            for position in range(3):
                val = [i for i in members if image_fold[cases[i]["image_ids"][position]] == fold]
                if not val:
                    continue
                train_labels = torch.tensor([cases[i]["labels"][position] == "P" for i in train])
                val_labels = torch.tensor([cases[i]["labels"][position] == "P" for i in val])
                both = bool(train_labels.any()) and not bool(train_labels.all())
                for column, layer in enumerate(layers):
                    slot = [cases[i]["vectors"][position] for i in val]
                    row = {"scope": scope, "group": group, "fold": fold, "position": position + 1, "layer": layer,
                           "n_train": len(train), "n_train_P": int(train_labels.sum()), "n_val": len(val)}
                    if not both:
                        predictions[slot, column] = 0.5
                        rows.append({**row, "status": "single_label_train_set_predicted_0.5"})
                        continue
                    model = fit_logistic(features[train, column], train_labels, **settings)
                    predictions[slot, column] = predict(model, features[val, column])
                    rows.append({**row, "status": "fitted", "converged": model["converged"],
                                 "iterations": model["iterations"]})
    require(not bool(torch.isnan(predictions).any()), "Some (input, position) slots received no OOF prediction")
    return predictions, rows


def readability_rows(cases, image_fold, predictions_by_feature, layers, threshold):
    """위치별 이미지 P/N 판별력. image_fold가 None이면(STEP 4) 모든 입력이 미관측이므로 'all'만 계산한다."""
    rows = []
    contexts = ("all",) if image_fold is None else ("all", "pure", "mixed")
    for column, layer in enumerate(layers):
        for position in range(3):
            for context in contexts:
                members = []
                for case in cases:
                    if image_fold is None:
                        members.append(case)
                        continue
                    target = image_fold[case["image_ids"][position]]
                    pure = all(image_fold[image] == target for image in case["image_ids"])
                    if context == "all" or (context == "pure") == pure:
                        members.append(case)
                if not members:
                    continue
                slots = [case["vectors"][position] for case in members]
                labels = [case["labels"][position] == "P" for case in members]
                row = {"layer": layer, "position": position + 1, "context": context, "n": len(members)}
                for feature, predictions in predictions_by_feature.items():
                    result = score(predictions[slots, column], labels, threshold)
                    row[f"{feature}_accuracy"], row[f"{feature}_auroc"] = result["accuracy"], result["auroc"]
                rows.append(row)
    return rows


def run_hq_probe(run_dir, step3_dir, config):
    verification = config["verification"]
    metadata, cases, features = load_unique_inputs(run_dir, verification["atol"], verification["rtol"])
    signature = extraction_signature(metadata)
    layers = signature["layers"]
    split = read_json(step3_dir / "split.json")
    require(split["folds"] == config["folds"] and split["seed"] == config["seed"], "Fold settings differ from STEP 3")
    image_fold = {int(key): value for key, value in split["image_fold"].items()}
    threshold = config["threshold"]
    predictions = {"z_i": zi_predictions(cases, step3_dir / "slots.csv", layers)}
    probe_rows = []
    for scope in config["scopes"]:
        values, rows = cross_fit_hq(cases, features, layers, image_fold, config["folds"], config["probe"], scope)
        predictions[f"h_q_{scope}"] = values
        probe_rows.extend(rows)
    readability = readability_rows(cases, image_fold, predictions, layers, threshold)
    bootstrap = ImageBootstrap([image for case in cases for image in case["image_ids"]],
                               config["bootstrap_reps"], config["seed"])
    groups, located = [], []
    pure_cases = [case for case in cases if len({image_fold[image] for image in case["image_ids"]}) == 1]
    for feature, values in predictions.items():
        for row in group_comparison({"unique_input": cases}, values, layers, threshold, bootstrap):
            groups.append({"feature": feature, **row})
        for subset, members in (("all", cases), ("pure", pure_cases)):
            for row in localization(members, values, layers, bootstrap):
                located.append({"feature": feature, "inputs": subset, **row})
    report = {
        "status": config["status"], "scope_note": SCOPE_NOTE, "run_dir": str(run_dir), "step3_dir": str(step3_dir),
        "extraction_settings": signature,
        "settings": {key: config[key] for key in ("folds", "seed", "threshold", "probe", "scopes", "bootstrap_reps")},
        "counts": {"unique_inputs": len(cases), "slots": 3 * len(cases),
                   "probes_fitted": sum(row["status"] == "fitted" for row in probe_rows),
                   "probes_single_label": sum(row["status"] != "fitted" for row in probe_rows),
                   "probes_not_converged": sum(row.get("converged") is False for row in probe_rows)},
        "shortcut_note": SHORTCUT_NOTE,
        "readability": readability, "group_comparison": groups, "localization": located,
    }
    return report, cases, predictions, probe_rows


def fit_final_hq(cases, features, layers, settings, threshold, scopes, progress=False):
    """STEP 4용 h_q readout: (범위, 클래스, 레이어, 위치)별로 모든 고유 입력을 써서 학습한다."""
    readouts, rows = [], []
    classes = sorted({case["class"] for case in cases})
    total = sum(len(classes) if scope == "per_class" else 1 for scope in scopes) * 3 * len(layers)
    bar = tqdm(total=total, desc="Fit h_q readouts", unit="readout", disable=not progress)
    for scope in scopes:
        groups = classes if scope == "per_class" else ["all"]
        for group in groups:
            members = [i for i, case in enumerate(cases) if group == "all" or case["class"] == group]
            for position in range(3):
                labels = torch.tensor([cases[i]["labels"][position] == "P" for i in members])
                for column, layer in enumerate(layers):
                    model = fit_logistic(features[members, column], labels, **settings)
                    train = score(predict(model, features[members, column]), labels, threshold)
                    readouts.append({"scope": scope, "class": group, "layer": layer, "position": position + 1,
                                     "mean": model["mean"], "scale": model["scale"], "weight": model["weight"],
                                     "bias": model["bias"]})
                    rows.append({"scope": scope, "class": group, "layer": layer, "position": position + 1,
                                 "n_train": len(members), "n_train_P": train["n_P"], "train_accuracy": train["accuracy"],
                                 "iterations": model["iterations"], "grad_max": model["grad_max"],
                                 "converged": model["converged"]})
                    bar.update()
    bar.close()
    return readouts, rows


def predict_hq(cases, queries, readouts, layers, scope):
    """고정 readout으로 (입력, 위치)별 확률을 만든다. queries[i]는 cases[i]의 h_q [레이어, 차원]."""
    lookup = {(r["scope"], r["class"], r["layer"], r["position"]): r for r in readouts}
    values = torch.full((3 * len(cases), len(layers)), float("nan"), dtype=torch.float64)
    groups = defaultdict(list)
    for index, case in enumerate(cases):
        groups[case["class"] if scope == "per_class" else "all"].append(index)
    for group, members in groups.items():
        for position in range(1, 4):
            for column, layer in enumerate(layers):
                readout = lookup.get((scope, group, layer, position))
                require(readout is not None, f"No frozen h_q readout: {scope}/{group}/L{layer}/position {position}")
                slots = [3 * index + position - 1 for index in members]
                values[slots, column] = predict(readout, queries[members, column])
    require(not bool(torch.isnan(values).any()), "Some (input, position) slots received no h_q prediction")
    return values
