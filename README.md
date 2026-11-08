# 다중 이미지 객체 존재 판단 실험

## 1. 개요

Qwen2.5-VL에 이미지 3장과 질문을 주고, 특정 객체가 세 이미지 모두에 있는지 Yes/No로 답하게 합니다.
이미지를 한 장씩 바꾸거나 순서를 바꿨을 때 모델의 답변이 어떻게 달라지는지 확인합니다.

## 2. 실험내용

COCO val2017 이미지와 COCO-REM 주석을 사용합니다. P는 대상 객체가 있는 이미지, N은 없는 이미지입니다.

- **실험 A:** `PPP → PPN → PNN → NNN`으로 한 장씩 교체합니다. PPP의 정답은 Yes, 나머지는 No입니다.
- **실험 B:** 같은 이미지 3장을 `NPP`, `PNP`, `PPN` 순서로 배치합니다. 정답은 모두 No이며, 순서에 따라 답변이 바뀌는지 봅니다.

기본 설정은 20개 클래스이며, A는 1,600개, B는 1,200개 케이스입니다.
모델의 응답과 Yes/No logit을 저장하고, 정확도와 순서별 응답 일치율 등을 계산합니다.

## 3. 실행방법

가상환경을 활성화한 뒤 프로젝트 루트에서 실행합니다. 아래 명령은 기본 모델인 Qwen2.5-VL-3B 기준입니다.
설정과 결과 파일에 저장되는 경로는 프로젝트 루트 기준 상대경로를 사용합니다.

필요한 패키지와 데이터, 모델을 받습니다.

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/download_model.py --config configs/models/qwen2_5_vl_3b.yaml
```

이미지 후보를 고르고 A/B 케이스를 만듭니다.

```bash
python src/data_manipulate/make_pool.py --config configs/data/balanced3.yaml
python src/data_manipulate/make_inference_pool.py --config configs/data/balanced3.yaml
python src/data_manipulate/build_cases.py --groups-json data/balanced3/inference_pools/inference_groups_top20_seed42.json --mode A
python src/data_manipulate/build_cases.py --groups-json data/balanced3/inference_pools/inference_groups_top20_seed42.json --mode B
```

Baseline 추론을 실행합니다.

```bash
python src/inference/baseline.py --config configs/inference/baseline.yaml
```

결과를 집계하고 그래프로 저장합니다. 추론 결과가 이미 있으면 이 단계부터 실행하면 됩니다.

```bash
python src/evaluation/compute_metrics.py --results-jsonl reports/inference/baseline/qwen2_5_vl_3b/inference_groups_top20_seed42_A_results.jsonl reports/inference/baseline/qwen2_5_vl_3b/inference_groups_top20_seed42_B_results.jsonl --output-dir reports/metrics/baseline/qwen2_5_vl_3b
python src/visualize/plot_metrics.py --metrics-json reports/metrics/baseline/qwen2_5_vl_3b/metrics.json
```

추론 결과는 `reports/inference/`, 메트릭과 그래프는 `reports/metrics/` 아래에 저장됩니다.
