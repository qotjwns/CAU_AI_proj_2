# 1번 실험: 다중 이미지 객체 존재 판단 Baseline

## 1. 개요

Qwen2.5-VL에 이미지 3장과 질문을 주고, 특정 객체가 세 이미지 모두에 있는지 Yes/No로 답하게 합니다.
이미지를 한 장씩 바꾸거나 순서를 바꿨을 때 모델의 답변이 어떻게 달라지는지 확인합니다.

## 2. 실험내용

COCO val2017 이미지와 COCO-REM 주석을 사용합니다. P는 대상 객체가 있는 이미지, N은 없는 이미지입니다.

- **실험 A:** `PPP → PPN → PNN → NNN`으로 한 장씩 교체합니다. PPP의 정답은 Yes, 나머지는 No입니다.
- **실험 B:** 같은 이미지 3장을 `NPP`, `PNP`, `PPN` 순서로 배치합니다. 정답은 모두 No이며, 순서에 따라 답변이 바뀌는지 봅니다.

기본 설정은 20개 클래스이며, A는 1,600개, B는 1,200개 케이스입니다.
모델의 응답과 Yes/No logit을 저장하고, 정확도와 순서별 응답 일치율 등을 계산합니다.

## 3. 코드 구조

공통 구현은 `src/`, 데이터 준비 명령은 `scripts/`, 1번 실험의 실행 진입점은
`experiments/step1_baseline/`에 둡니다.

```text
configs/
├── data/balanced3.yaml
├── models/
│   ├── qwen2_5_vl_3b.yaml
│   └── qwen2_5_vl_7b.yaml
└── experiments/step1.yaml
src/
├── common/                 # 설정, JSON/JSONL 읽기·쓰기, 프로젝트 경로
├── data_manipulate/
│   ├── build_pools.py      # 클래스별 양성·음성 후보
│   ├── build_groups.py     # 이미지 샘플링과 비교 그룹
│   └── build_cases.py      # A/B 케이스 생성
├── core/
│   ├── model.py            # 모델과 processor 로딩
│   ├── prompts.py          # 입력 메시지 구성과 Yes/No 응답 해석
│   ├── inference.py        # 케이스 한 건의 추론과 logit 추출
│   ├── metrics.py          # 정확도·쌍별 비교 등 지표 계산
│   └── results.py          # 결과 레코드 구성과 JSONL/CSV 입출력
└── visualization/
    ├── plots.py            # 메트릭 그래프
    └── cases.py            # 클래스별 입력 이미지 시각화
experiments/step1_baseline/
├── run.py
├── analyze.py
├── visualize.py
└── visualize_cases.py
scripts/
├── download_data.py
├── download_model.py
├── build_pools.py
├── build_groups.py
└── build_cases.py
```

## 4. 실행방법

가상환경을 활성화한 뒤 프로젝트 루트에서 `python -m ...` 형태로 실행합니다.
아래 명령은 기본 모델인 Qwen2.5-VL-3B 기준입니다.
설정과 결과 파일에 저장되는 경로는 프로젝트 루트 기준 상대경로를 사용합니다.

필요한 패키지와 데이터, 모델을 받습니다.

```bash
python -m pip install -r requirements.txt
python -m scripts.download_data
python -m scripts.download_model --config configs/models/qwen2_5_vl_3b.yaml
```

이미지 후보를 고르고 A/B 케이스를 만듭니다.

```bash
python -m scripts.build_pools --config configs/data/balanced3.yaml
python -m scripts.build_groups --config configs/data/balanced3.yaml
python -m scripts.build_cases --groups-json data/balanced3/inference_pools/inference_groups_top20_seed42.json --mode A
python -m scripts.build_cases --groups-json data/balanced3/inference_pools/inference_groups_top20_seed42.json --mode B
```

Baseline 추론을 실행합니다.

```bash
python -m experiments.step1_baseline.run --config configs/experiments/step1.yaml
```

파일럿으로 총 10개 케이스만 추론하려면 다음 명령을 사용합니다.

```bash
python -m experiments.step1_baseline.run --limit 10
```

`--limit`은 설정 파일에 나열된 입력을 순서대로 읽어 전체에서 최대 해당 개수만 처리합니다.
기본 설정은 A 파일이 먼저이므로 위 명령은 A의 앞 10개 케이스를 처리합니다.
결과는 `results/step1/pilot_10/`에 A/B별 JSONL로 저장합니다.
케이스 단위로 자르므로 마지막 비교 쌍은 일부만 포함될 수 있습니다.

저장 위치를 직접 지정하거나 파일럿을 다시 실행할 때는 새 폴더를 지정합니다.
기존 결과 파일은 덮어쓰지 않습니다.

```bash
python -m experiments.step1_baseline.run --limit 10 --output-dir results/step1/pilot_10_run2
```

결과를 집계하고 그래프로 저장합니다. 추론 결과가 이미 있으면 이 단계부터 실행하면 됩니다.

```bash
python -m experiments.step1_baseline.analyze --results-jsonl results/step1/inference_groups_top20_seed42_A_results.jsonl results/step1/inference_groups_top20_seed42_B_results.jsonl
python -m experiments.step1_baseline.visualize --metrics-json results/step1/metrics/metrics.json
```

실험 결과 JSONL·JSON·CSV는 `results/`, PNG·JPG 등 그림은 `reports/`에 저장합니다.
STEP 1 추론 결과는 A/B 파일을 유지하며 `results/step1/`에 저장합니다.
메트릭은 기본적으로 입력 결과 폴더 아래 `metrics/`에 저장합니다.
그래프는 `results/` 아래 메트릭 경로에 대응하는 `reports/` 경로의 `plots/`에 저장합니다.
예를 들어 `results/step1/pilot_10/metrics/metrics.json`의 그래프는 `reports/step1/pilot_10/metrics/plots/`에 저장됩니다.
`results/` 밖의 메트릭 파일을 시각화할 때는 `--output-dir`로 그림 저장 폴더를 지정합니다.

```text
results/
└── step1/
    ├── inference_groups_top20_seed42_A_results.jsonl
    ├── inference_groups_top20_seed42_B_results.jsonl
    └── metrics/            # metrics.json, 지표별 CSV
reports/
└── step1/
    ├── metrics/plots/     # 메트릭 그래프
    └── cases/             # 입력 이미지 조합 그림
```

입력 이미지 조합을 클래스별로 확인하려면 다음 명령을 사용합니다.

```bash
python -m experiments.step1_baseline.visualize_cases --cases-jsonl data/balanced3/inference_cases/inference_groups_top20_seed42_A_cases.jsonl data/balanced3/inference_cases/inference_groups_top20_seed42_B_cases.jsonl --limit 2
```

이미지 조합 그림은 기본적으로 `reports/step1/cases/`에 저장됩니다.
기존 `--category`, `--num-classes`, `--seed` 등의 그룹 생성 옵션은 `scripts/build_groups.py`에서 사용합니다.
`python -m scripts.build_pools --mode A` 또는 `--mode B`로 후보·그룹·해당 케이스를 한 번에 생성할 수도 있습니다.
