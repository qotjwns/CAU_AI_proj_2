"""STEP 6-2 GPU 실행. 서버에서는 scripts/step6_contribution_gpu.sh가 순서대로 부른다.

python -m experiments.step6_contribution.run --stage basis   # STEP 3 풀 기여 → P 기저·평균 차이 (basis.pt)
python -m experiments.step6_contribution.run --stage pilot   # 연습 실행 (선택 절반, 무결성·시간 전용)
python -m experiments.step6_contribution.run --stage main    # 본 실행 (확증 절반)
--resume: 끝나지 않은 실행을 이어서 한다 (pilot·main은 끝난 batch를 건너뛰고, basis는 처음부터 다시 한다).

실행 전에 고정 manifest의 파일 해시를 확인하고 실행 환경(GPU·드라이버·라이브러리)을 run.json에 기록한다.
선행 조건: basis ← 실제 3B 검증 통과, pilot ← basis 검사 통과, main ← 연습 실행 검사·시간 추정 통과.
출력: extraction.jsonl (입력별 재구성 검사), conditions.jsonl (batch별 행 결과·벡터 검사·벡터 요약),
reference.jsonl (연습 실행: 같은 행의 전체 prefill 참조 경로 결과), basis 단계는 basis.pt·basis_moments.pt·
basis_summary.json.
"""
import argparse
from datetime import datetime, timezone
import json
import statistics

import torch
from tqdm import tqdm
import transformers

from src.common.data import read_json, read_jsonl, write_json
from src.common.environment import collect_environment
from src.common.paths import project_path, project_relative_path
from src.core.causal_plan import load_plan
from src.core.causal_runtime import Runner, timed
from src.core.feature_validation import require
from src.core.inference import answer_variant_ids
from src.core.model import load_model
from src.core.probe_freeze import sha256_file
from src.common.config import read_yaml
from experiments.step6_contribution.conditions import LAYERS, stage_work
from experiments.step6_contribution.contribution import attention_modules
from experiments.step6_contribution.freeze import load_config, verify_freeze
from experiments.step6_contribution.plan import load_step6_plan
from experiments.step6_contribution.runtime import (
    BasisAccumulator, Directions, SuffixAddition, basis_report, extract, fit_bases, reference_rows, run_batches,
    vector_summary)


def append(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def existing(path, key):
    return {row[key] for row in read_jsonl(path)} if path.exists() else set()


def passed(path):
    return path.exists() and read_json(path).get("status") == "passed"


def model_files_sha256(model_name):
    directory = project_path("cache/models") / model_name
    return {path.name: sha256_file(path) for path in sorted(directory.glob("*.json"))}


# ---------------------------------------------------------------- basis 단계

def run_basis(runner, plan6, output_dir, metadata, layers=LAYERS):
    path = output_dir / "extraction.jsonl"
    if path.exists():
        path.unlink()                                  # 처음부터 다시 (누적 합은 이어 붙일 수 없다)
        metadata["restarted_from_scratch"] = True
    hidden = runner.model.config.text_config.hidden_size
    accumulator = BasisAccumulator(hidden, runner.device, layers)
    with path.open("w", encoding="utf-8") as handle:
        for row in tqdm(plan6["pool"], desc="basis (STEP 3 풀 기여)", unit="input"):
            prepared, prepare_seconds = timed(runner.prepare, row["case"])
            (parts, extraction), seconds = timed(extract, runner.model, prepared["inputs"], prepared["spans"], layers)
            require(len(prepared["spans"]) == len(row["labels"]), "Image count differs from the case")
            if extraction["valid"]:
                accumulator.add(parts, row["labels"], row["mixed"])
            append(handle, {"input_id": row["input_id"], "class": row["class"], "composition": row["composition"],
                            "mixed": row["mixed"], "prepare_seconds": prepare_seconds, "seconds": seconds,
                            **extraction})
    payload = fit_bases(accumulator)
    torch.save(payload, output_dir / "basis.pt")
    torch.save({"layers": list(layers), "moment": {k: v.cpu() for k, v in accumulator.moment.items()},
                "sums": {key: {k: v.cpu() for k, v in sums.items()} for key, sums in accumulator.sums.items()},
                "vectors": dict(accumulator.vectors), "slots": dict(accumulator.slots)},
               output_dir / "basis_moments.pt")
    report = basis_report(payload)
    report["basis_sha256"] = sha256_file(output_dir / "basis.pt")
    report["basis_moments_sha256"] = sha256_file(output_dir / "basis_moments.pt")
    write_json(report, output_dir / "basis_summary.json", allow_nan=False)
    for layer, row in report["layers"].items():
        print(f"L{layer + 1}: k90 {row['k']['v90']}, k95 {row['k']['v95']}, r64 설명 {row['explained']['r64']:.3f}, "
              f"‖μ_N − μ_P‖ {row['mean_difference_norm']:.3f}")


# ---------------------------------------------------------------- pilot·main 단계

def run_items(runner, plan6, plan53, stage, output_dir, directions):
    layers = directions.layers
    work = stage_work(plan6, plan53, stage)
    done = existing(output_dir / "conditions.jsonl", "batch")
    by_item = {}
    for batch in work["batches"]:
        by_item.setdefault(batch["item_id"], []).append(batch)
    addition = SuffixAddition(attention_modules(runner.model, layers))
    reference = (output_dir / "reference.jsonl").open("a", encoding="utf-8") if stage == "pilot" else None
    try:
        with (output_dir / "conditions.jsonl").open("a", encoding="utf-8") as handle, \
                (output_dir / "extraction.jsonl").open("a", encoding="utf-8") as extraction_handle:
            for item_id, batches in tqdm(by_item.items(), desc=f"{stage} (기여 개입)", unit="item"):
                todo = [b for b in batches if b["batch"] not in done]
                if not todo:
                    continue
                item = work["items"][item_id]
                entry = plan53["inputs"][item["input_id"]]
                prepared, prepare_seconds = timed(runner.prepare, entry["case"], entry["baseline"])
                require(prepared["spans"][-1]["end"] + 1 == item["suffix_start"]
                        and prepared["length"] - 1 == item["last_index"], "Suffix boundary differs from the plan")
                (parts, extraction), extract_seconds = timed(extract, runner.model, prepared["inputs"],
                                                             prepared["spans"], layers)
                prefix, prefix_seconds = timed(runner.prefix, prepared, item["suffix_start"])
                append(extraction_handle, {"item_id": item_id, "input_id": item["input_id"],
                                           "seconds": extract_seconds, **extraction})
                outputs = run_batches(runner, addition, prepared, prefix, parts, item, todo, directions)
                for index, (batch, output) in enumerate(zip(todo, outputs)):
                    common = {k: batch[k] for k in ("batch", "item_id", "input_id", "kind", "slot", "primary_slot",
                                                    "n_images", "half")}
                    append(handle, {
                        **common, "seconds": output["seconds"],
                        "prefix_seconds": prepare_seconds + prefix_seconds if index == 0 else 0.0,
                        "extract_seconds": extract_seconds if index == 0 else 0.0,
                        "extraction_valid": extraction["valid"], "vector_checks": output["vector_checks"],
                        "vector_summary": vector_summary(parts, batch["slot"], directions, extraction),
                        "rows": [{**spec, **result} for spec, result in zip(batch["rows"], output["rows"])]})
                    if reference is not None:
                        append(reference, {"batch": batch["batch"],
                                           "rows": reference_rows(runner, prepared, batch, output["vectors"],
                                                                  output["scales"])})
                del parts, outputs, prefix
    finally:
        addition.remove()
        if reference is not None:
            reference.close()


# ---------------------------------------------------------------- 진입점

def preconditions(config, stage):
    dirs = {name: project_path(path) for name, path in config["dirs"].items()}
    if stage == "basis":
        path = project_path(config["validation_output"])
        require(path.exists() and read_json(path).get("passed") is True,
                "The real 3B validation has not passed; the basis stage is not allowed")
        return None
    check = dirs["basis"] / "basis_check.json"
    require(passed(check), "The basis check has not passed")
    digest = read_json(check)["basis_sha256"]
    require(sha256_file(dirs["basis"] / "basis.pt") == digest, "basis.pt differs from the checked basis")
    if stage == "main":
        require(passed(dirs["pilot"] / "pilot_check.json"), "The pilot check did not pass; the main run is not allowed")
        require(passed(project_path(config["stop_conditions"]["estimate_file"])), "The time estimate has not passed")
    return digest


def main():
    parser = argparse.ArgumentParser(description="STEP 6-2 GPU 실행")
    parser.add_argument("--stage", choices=("basis", "pilot", "main"), required=True)
    parser.add_argument("--resume", action="store_true", help="끝나지 않은 실행을 이어서")
    args = parser.parse_args()
    config = load_config()
    manifest = verify_freeze(config)
    basis_digest = preconditions(config, args.stage)
    plan6 = load_step6_plan(config["plan_dir"])
    plan53 = load_plan(read_yaml(project_path(config["split_config"])))
    output_dir = project_path(config["dirs"][args.stage])
    if (output_dir / "run.json").exists():
        previous = read_json(output_dir / "run.json")
        require(args.resume and previous["status"] != "completed",
                f"{output_dir} already has a run (status {previous['status']}); use --resume only for an unfinished run")
    output_dir.mkdir(parents=True, exist_ok=True)
    before, packages = collect_environment()
    metadata = {
        "status": "running", "stage": args.stage, "resumed": args.resume,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen": {"dir": config["frozen_dir"], "version": manifest["version"],
                   "manifest_sha256": sha256_file(project_path(config["frozen_dir"]) / "manifest.json")},
        "basis_sha256": basis_digest, "environment": before, "model_name": config["model_name"],
        "model_files_sha256": model_files_sha256(config["model_name"]),
    }
    (output_dir / "environment_packages.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    write_json(metadata, output_dir / "run.json", allow_nan=False)
    model, processor, answer_ids = load_model(config["model_name"])
    runner = Runner(model, processor, answer_ids, answer_variant_ids(processor.tokenizer))
    metadata.update({
        "device": str(model.device), "dtype": str(runner.dtype), "vision_dtype": str(model.model.visual.dtype),
        "attention_implementation": model.get_decoder().config._attn_implementation, "n_layers": runner.n_layers,
        "repetition_penalty": runner.repetition_penalty, "answer_token_ids": list(answer_ids),
        "layers": list(LAYERS), "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    })
    write_json(metadata, output_dir / "run.json", allow_nan=False)
    if args.stage == "basis":
        run_basis(runner, plan6, output_dir, metadata)
    else:
        directions = Directions(torch.load(project_path(config["dirs"]["basis"]) / "basis.pt"), runner.device)
        metadata["basis_k"] = directions.k
        work = stage_work(plan6, plan53, args.stage)
        metadata.update(items=len(work["items"]), batches=len(work["batches"]),
                        reference_batches=len(work["reference_batches"]))
        write_json(metadata, output_dir / "run.json", allow_nan=False)
        run_items(runner, plan6, plan53, args.stage, output_dir, directions)
    after, _ = collect_environment()
    seconds = [row["seconds"] for row in read_jsonl(output_dir / ("extraction.jsonl" if args.stage == "basis"
                                                                   else "conditions.jsonl"))]
    metadata.update({
        "status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "same_gpu_at_end": before.get("nvidia_smi") == after.get("nvidia_smi"),
        "median_seconds": statistics.median(seconds) if seconds else None,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    })
    write_json(metadata, output_dir / "run.json", allow_nan=False)
    print(f"완료: {project_relative_path(output_dir)} ({args.stage})")


if __name__ == "__main__":
    main()
