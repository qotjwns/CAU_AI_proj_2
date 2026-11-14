"""STEP 6-1 실제 Qwen2.5-VL-3B 소규모 검증: 입력 3개, batch 1, 무개입 forward와 개입 위치 검사 (성능 평가 아님).

python -m experiments.step6_contribution.validate_real                       # float32와 bfloat16 둘 다
python -m experiments.step6_contribution.validate_real --dtypes bfloat16
→ results/step6/qwen2_5_vl_3b/contribution_validation/real_3b.json

- 입력: validation.real_cases 규칙으로 고정한 3개 (실험 B 선택 절반: 3장·middle, 4장·first, 5장·last)
- 층: L21–24 (0부터 센 디코더 층 20–23), 질문 토큰: 마지막 vision_end 다음부터 마지막 입력 토큰까지 (STEP 5-3B와 같음)
- 재구성: 층별 c_image_i, c_nonimage를 더해 모델 attention 출력과 비교. 판정은 validation.RULE (고정본 v3: 항등식은
  원소별 allclose, attention·output은 층별 상대 오차 fp32 ≤ 1e-5 / bf16 ≤ 2⁻⁶). v2의 원소별 판정은 fp32 5장 입력
  L22에서 모델 fp32 커널 반올림만으로 1% 넘어 실패했다 (기록: real_3b_fail_v2.json)
  재계산은 float64 (STEP 6-2 본 실행 추출과 같음). float32 재계산은 항등식 검사가 합산 순서 반올림만으로 1e-5를
  넘을 수 있어 바꿨다 (작은 모델 모의 실행에서 발견, 실제 모델 실행 전, 허용 오차는 그대로)
- 개입: 별도 무개입 실행에서 뽑은 c_N을 λ = 3으로 더함. λ = 0 동일성, 지정 층·토큰에만 직접 수정이 있는지 확인.
  후속 층 hidden 변화는 자연스러운 결과로 따로 표시한다.
attention 구현은 기준 실행과 같은 기본값(sdpa)이다. 결과가 실패여도 JSON은 남기고 0이 아닌 종료 코드로 끝난다.
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.common.config import read_yaml
from src.common.paths import project_path
from src.core.causal_plan import load_plan
from experiments.step6_contribution.validation import RULE, intervention, prepare_real, real_cases, reconstruction


MODEL_DIR = "cache/models/Qwen/Qwen2.5-VL-3B-Instruct"
LAYERS = [20, 21, 22, 23]                    # 밴드 (21, 24), STEP 5-3B와 같은 번호 규칙 (low − 1 … high − 1)
SCALE = 3.0
OUTPUT = Path("results/step6/qwen2_5_vl_3b/contribution_validation/real_3b.json")


def load(dtype, device):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    directory = project_path(MODEL_DIR)
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(directory, dtype=dtype, local_files_only=True)
    model = model.to(device).eval()
    (yes,) = processor.tokenizer.encode("Yes", add_special_tokens=False)
    (no,) = processor.tokenizer.encode("No", add_special_tokens=False)
    return model, processor, (yes, no)


def main():
    parser = argparse.ArgumentParser(description="STEP 6-1 실제 3B 소규모 검증 (입력 3개)")
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"], choices=["float32", "bfloat16"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    import transformers

    plan = load_plan(read_yaml(project_path("configs/data/step5_causal_split.yaml")))
    items = real_cases(plan)
    config = read_yaml(project_path("configs/experiments/step6_contribution.yaml"))
    report = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": args.device,
              "frozen_dir": config["frozen_dir"], "reconstruction_rule": RULE,
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(),
              "torch": torch.__version__, "transformers": transformers.__version__, "layers": LAYERS, "scale": SCALE,
              "runs": {}}
    passed = True
    for name in args.dtypes:
        dtype = getattr(torch, name)
        model, processor, answer_ids = load(dtype, args.device)
        report["attention_implementation"] = model.get_decoder().config._attn_implementation
        for item in items:
            inputs, spans, info = prepare_real(model, processor, plan, item)
            key = f"{name}/{info['n_images']}img_{info['position']}"
            recon = reconstruction(model, inputs, spans, LAYERS, compute_dtype=torch.float64)
            inter = intervention(model, inputs, spans, LAYERS, info["target_index"], SCALE, answer_ids)
            report["runs"][key] = {"input": info, "reconstruction": recon, "intervention": inter}
            ok = (recon["passed"] and inter["passed"] and info["spans_match_stored_baseline"]
                  and info["suffix_start_matches_plan"])
            passed &= ok
            o = recon["overall"]["output"]
            print(f"{key} T={info['length']}: 재구성 max {o['max_abs']:.2e} 상대 {o['relative']:.2e} "
                  f"({'통과' if recon['passed'] else '실패'}) | 개입 위치 {'통과' if inter['passed'] else '실패'}")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    report["passed"] = passed
    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = project_path(OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"저장: {OUTPUT} | 전체 {'통과' if passed else '실패'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
