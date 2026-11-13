"""STEP 5-3 실험 A·B 실행 (GPU).

python -m experiments.step5_causal.run --experiment A --stage pilot   # 연습 실행 (무결성·시간 측정 전용)
python -m experiments.step5_causal.run --experiment A --stage main    # 본 실행 (연습 실행 검사 통과 후)
--resume: 중단된 실행을 이어서 한다 (끝난 batch는 건너뜀).

실행 전에 고정 manifest의 파일 해시를 확인하고, 실행 환경(GPU·드라이버·라이브러리)을 run.json에 기록한다.
출력: phase1.jsonl (A: 입력별 무개입 결과), conditions.jsonl (batch별 조건 결과), reference.jsonl (연습 실행:
기준 추출과 같은 전체 prefill 경로의 같은 조건 결과).
"""
import argparse
from datetime import datetime, timezone
import json
import statistics

import torch
from tqdm import tqdm
import transformers

from src.common.config import read_yaml
from src.common.data import read_json, read_jsonl, write_json
from src.common.environment import collect_environment
from src.common.paths import project_path, project_relative_path
from src.core.causal_conditions import a_work, b_work
from src.core.causal_freeze import experiment_config, verify_causal_freeze
from src.core.causal_plan import load_plan
from src.core.causal_runtime import Runner, timed
from src.core.feature_validation import require
from src.core.inference import answer_variant_ids
from src.core.model import load_model
from src.core.probe_freeze import sha256_file


def existing(path, key):
    return {row[key] for row in read_jsonl(path)} if path.exists() else set()


def append(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


# ---------------------------------------------------------------- 실험 A

def run_a(runner, plan, work, output_dir, metadata):
    inputs, n_layers = plan["inputs"], runner.n_layers
    done_phase1 = existing(output_dir / "phase1.jsonl", "input_id")
    residuals = {}
    with (output_dir / "phase1.jsonl").open("a", encoding="utf-8") as handle:
        for input_id in tqdm(work["phase1"], desc="A phase 1 (무개입, donor residual)", unit="input"):
            entry = inputs[input_id]
            prepared, prepare_seconds = timed(runner.prepare, entry["case"], entry["baseline"])
            (rows, stack), seconds = timed(lambda: runner.run_suffix(
                prepared, runner.prefix(prepared, prepared["length"] - 1), 1, capture=True))
            seconds += prepare_seconds
            residuals[input_id] = stack[0]
            if input_id not in done_phase1:
                append(handle, {"input_id": input_id, "n_images": entry["case"]["n_images"], "seconds": seconds,
                                **rows[0]})
    metadata["phase1_inputs"] = len(work["phase1"])
    done = existing(output_dir / "conditions.jsonl", "batch")
    by_recipient = {}
    for batch in work["batches"]:
        by_recipient.setdefault(batch["recipient"], []).append(batch)
    references = set(work["reference_batches"])
    reference_handle = (output_dir / "reference.jsonl").open("a", encoding="utf-8") if references else None
    try:
        with (output_dir / "conditions.jsonl").open("a", encoding="utf-8") as handle:
            for recipient, batches in tqdm(sorted(by_recipient.items()), desc="A phase 2 (이식)", unit="recipient"):
                todo = [b for b in batches if b["batch"] not in done]
                if not todo:
                    continue
                entry = inputs[recipient]
                prepared, prepare_seconds = timed(runner.prepare, entry["case"], entry["baseline"])
                prefix, prefix_seconds = timed(runner.prefix, prepared, prepared["length"] - 1)
                prefix_seconds += prepare_seconds
                for batch in todo:
                    rows = batch["rows"]
                    layer = torch.tensor([-1 if r["layer"] is None else r["layer"] for r in rows])
                    donor = torch.stack([residuals[r["donor"]] if r["donor"] else
                                         torch.zeros_like(residuals[recipient]) for r in rows])
                    (results, _), seconds = timed(runner.run_suffix, prepared, prefix, len(rows),
                                                  patch={"layer": layer, "donor": donor})
                    common = {k: batch[k] for k in ("batch", "kind", "unit_id", "direction", "recipient", "n_images",
                                                    "half")}
                    append(handle, {**common, "seconds": seconds, "prefix_seconds": prefix_seconds,
                                    "rows": [{**spec, **result} for spec, result in zip(rows, results)]})
                    if batch["batch"] in references:
                        reference_rows = []
                        for spec in rows:
                            result = runner.reference(prepared, spec["layer"],
                                                      residuals[spec["donor"]] if spec["donor"] else None)
                            reference_rows.append({**spec, **result})
                        append(reference_handle, {"batch": batch["batch"], "rows": reference_rows})
                    prefix_seconds = 0.0
    finally:
        if reference_handle:
            reference_handle.close()


# ---------------------------------------------------------------- 실험 B

def b_attention(item, rows, prepared, n_layers, runner):
    start, length = item["suffix_start"], prepared["length"]
    suffix = length - start
    spans = prepared["spans"]
    batch = len(rows)
    layers = torch.zeros(batch, n_layers, dtype=torch.bool)
    select = torch.zeros(batch, suffix, length, dtype=torch.bool)
    value = torch.zeros(batch, dtype=torch.float32)
    specs = []
    for b, row in enumerate(rows):
        if row["kind"] == "none":
            specs.append(None)
            continue
        low, high = row["band"]
        layers[b, low - 1:high] = True
        index = item["target_index"] if row["target"] == "N" else item["control_index"]
        span = spans[index]
        queries = list(range(suffix)) if row["query"] == "post_image" else [suffix - 1]
        for q in queries:
            select[b, q, span["start"]:span["end"]] = True
        value[b] = {"block": float(runner.block_value), "boost": float(runner.boost_value), "zero": 0.0}[row["kind"]]
        specs.append({"layers": list(range(low - 1, high)), "queries": [start + q for q in queries],
                      "targets": [(span["start"], span["end"])], "value": value[b].item()})
    return {"layers": layers, "select": select, "value": value}, specs


def run_b(runner, plan, work, output_dir, metadata):
    inputs, n_layers = plan["inputs"], runner.n_layers
    done = existing(output_dir / "conditions.jsonl", "batch")
    references = set(work["reference_batches"])
    reference_handle = (output_dir / "reference.jsonl").open("a", encoding="utf-8") if references else None
    try:
        with (output_dir / "conditions.jsonl").open("a", encoding="utf-8") as handle:
            for batch in tqdm(work["batches"], desc="B attention", unit="item"):
                if batch["batch"] in done:
                    continue
                item = work["items"][batch["item_id"]]
                entry = inputs[batch["input_id"]]
                prepared, prepare_seconds = timed(runner.prepare, entry["case"], entry["baseline"])
                require(prepared["spans"][-1]["end"] + 1 == item["suffix_start"]
                        and prepared["length"] - 1 == item["last_index"], "Suffix boundary differs from the plan")
                prefix, prefix_seconds = timed(runner.prefix, prepared, item["suffix_start"])
                prefix_seconds += prepare_seconds
                attention, specs = b_attention(item, batch["rows"], prepared, n_layers, runner)
                (results, _), seconds = timed(runner.run_suffix, prepared, prefix, len(batch["rows"]),
                                              attention=attention)
                common = {k: batch[k] for k in ("batch", "item_id", "input_id", "n_images", "half")}
                append(handle, {**common, "seconds": seconds, "prefix_seconds": prefix_seconds,
                                "rows": [{**spec, **result} for spec, result in zip(batch["rows"], results)]})
                if batch["batch"] in references:
                    reference_rows = [{**row, **runner.reference(prepared, attention=spec)}
                                      for row, spec in zip(batch["rows"], specs)]
                    append(reference_handle, {"batch": batch["batch"], "rows": reference_rows})
        if work["reproduction"]:
            done_repro = existing(output_dir / "reproduction.jsonl", "input_id")
            with (output_dir / "reproduction.jsonl").open("a", encoding="utf-8") as handle:
                for input_id in tqdm(work["reproduction"], desc="B 재현 표본", unit="input"):
                    if input_id in done_repro:
                        continue
                    entry = inputs[input_id]
                    prepared = runner.prepare(entry["case"], entry["baseline"])
                    prefix = runner.prefix(prepared, prepared["spans"][-1]["end"] + 1)
                    attention = {"layers": torch.zeros(1, n_layers, dtype=torch.bool),
                                 "select": torch.zeros(1, prepared["length"] - prefix["length"], prepared["length"],
                                                       dtype=torch.bool), "value": torch.zeros(1)}
                    (results, _), seconds = timed(runner.run_suffix, prepared, prefix, 1, attention=attention)
                    append(handle, {"input_id": input_id, "n_images": entry["case"]["n_images"], "seconds": seconds,
                                    **results[0]})
    finally:
        if reference_handle:
            reference_handle.close()


# ---------------------------------------------------------------- 진입점

def model_files_sha256(model_name):
    directory = project_path("cache/models") / model_name
    return {path.name: sha256_file(path) for path in sorted(directory.glob("*.json"))}


def main():
    parser = argparse.ArgumentParser(description="STEP 5-3 실험 A·B 실행 (GPU)")
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--resume", action="store_true", help="중단된 실행을 이어서 (끝난 batch는 건너뜀)")
    args = parser.parse_args()
    _, config = experiment_config(args.experiment)
    manifest = verify_causal_freeze(args.experiment, config)
    split_config = read_yaml(project_path(config["split_config"]))
    plan = load_plan(split_config)
    output_dir = project_path(config["pilot_dir" if args.stage == "pilot" else "main_dir"])
    if (output_dir / "run.json").exists():
        previous = read_json(output_dir / "run.json")
        require(args.resume and previous["status"] != "completed",
                f"{output_dir} already has a run (status {previous['status']}); use --resume only for an unfinished run")
    if args.stage == "main":
        check = read_json(project_path(config["pilot_dir"]) / "pilot_check.json")
        require(check["status"] == "passed", "The pilot check did not pass; the main run is not allowed")
        estimate_path = project_path(config["stop_conditions"]["estimate_file"])
        require(estimate_path.exists() and read_json(estimate_path)["status"] == "passed",
                "The combined time estimate has not passed")
    output_dir.mkdir(parents=True, exist_ok=True)

    before, packages = collect_environment()
    metadata = {
        "status": "running", "experiment": args.experiment, "stage": args.stage, "resumed": args.resume,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen": {"dir": config["frozen_dir"], "version": manifest["version"],
                   "manifest_sha256": sha256_file(project_path(config["frozen_dir"]) / "manifest.json")},
        "environment": before, "model_name": config["model_name"],
        "model_files_sha256": model_files_sha256(config["model_name"]),
    }
    (output_dir / "environment_packages.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    write_json(metadata, output_dir / "run.json", allow_nan=False)
    model, processor, answer_ids = load_model(config["model_name"])
    variants = answer_variant_ids(processor.tokenizer)
    runner = Runner(model, processor, answer_ids, variants)
    metadata.update({
        "device": str(model.device), "dtype": str(runner.dtype), "vision_dtype": str(model.model.visual.dtype),
        "attention_implementation": model.get_decoder().config._attn_implementation, "n_layers": runner.n_layers,
        "repetition_penalty": runner.repetition_penalty, "boost_bias_in_mask_dtype": float(runner.boost_value),
        "answer_token_ids": list(answer_ids), "answer_variants": variants,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    })
    write_json(metadata, output_dir / "run.json", allow_nan=False)
    if args.experiment == "A":
        work = a_work(plan, args.stage, runner.n_layers)
        metadata.update(batches=len(work["batches"]), reproduction=len(work["reproduction"]),
                        reference_batches=len(work["reference_batches"]))
        write_json(metadata, output_dir / "run.json", allow_nan=False)
        run_a(runner, plan, work, output_dir, metadata)
    else:
        work = b_work(plan, args.stage)
        metadata.update(batches=len(work["batches"]), reproduction=len(work["reproduction"]),
                        reference_batches=len(work["reference_batches"]))
        write_json(metadata, output_dir / "run.json", allow_nan=False)
        run_b(runner, plan, work, output_dir, metadata)
    after, _ = collect_environment()
    seconds = [row["seconds"] for row in read_jsonl(output_dir / "conditions.jsonl")]
    metadata.update({
        "status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "same_gpu_at_end": before.get("gpus") == after.get("gpus") and before.get("nvidia_smi") == after.get("nvidia_smi"),
        "median_batch_seconds": statistics.median(seconds) if seconds else None,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    })
    write_json(metadata, output_dir / "run.json", allow_nan=False)
    print(f"완료: {project_relative_path(output_dir)} ({args.experiment} {args.stage})")


if __name__ == "__main__":
    main()
