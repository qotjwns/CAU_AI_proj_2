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

## 3. 코드 구조

공통 구현은 `src/`, 데이터 준비 명령은 `scripts/`, 실험별 실행 진입점은 `experiments/`에 둡니다.

```text
configs/
├── data/balanced3.yaml
├── models/
│   ├── qwen2_5_vl_3b.yaml
│   └── qwen2_5_vl_7b.yaml
└── experiments/
    ├── step1.yaml
    └── step2.yaml
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
│   ├── features.py         # 이미지 토큰 매핑과 Decoder 특징 추출
│   ├── metrics.py          # 정확도·쌍별 비교 등 지표 계산
│   └── results.py          # 결과 레코드 구성과 JSONL/CSV 입출력
└── visualization/
    ├── plots.py            # 메트릭 그래프
    ├── cases.py            # 클래스별 입력 이미지 시각화
    └── token_spans.py      # STEP 2 이미지와 토큰 구간 시각화
experiments/
├── step1_baseline/
│   ├── run.py
│   ├── analyze.py
│   ├── visualize.py
│   └── visualize_cases.py
└── step2_hidden_states/
    ├── run.py
    └── visualize.py
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

실험 결과 JSONL·JSON·CSV와 Hidden State 텐서는 `results/`, PNG·JPG 등 그림은 `reports/`에 저장합니다.
STEP 1 추론 결과는 A/B 파일을 유지하며 `results/step1/`에 저장합니다.
메트릭은 기본적으로 입력 결과 폴더 아래 `metrics/`에 저장합니다.
그래프는 `results/` 아래 메트릭 경로에 대응하는 `reports/` 경로의 `plots/`에 저장합니다.
예를 들어 `results/step1/pilot_10/metrics/metrics.json`의 그래프는 `reports/step1/pilot_10/metrics/plots/`에 저장됩니다.
`results/` 밖의 메트릭 파일을 시각화할 때는 `--output-dir`로 그림 저장 폴더를 지정합니다.

```text
results/
├── step1/
│   ├── inference_groups_top20_seed42_A_results.jsonl
│   ├── inference_groups_top20_seed42_B_results.jsonl
│   └── metrics/            # metrics.json, 지표별 CSV
└── step2/                 # Hidden State와 추출 기록
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

## 5. STEP 2: Hidden State 추출

STEP 1과 같은 전처리와 greedy generation을 사용합니다. 답변 생성 전 전체 입력을 처리하는
prefill에서 Decoder 레이어 9·18·27·36의 특징을 추출합니다. 모델은 별도 STEP 2 설정으로
Qwen2.5-VL-3B를 지정하며, hidden dimension은 2048입니다.

이미지별 `image_token_id` 구간을 찾고, 토큰 수가 `T × H × W / merge_size²`와 일치하는지 확인합니다.
vision 시작·종료 토큰과 padding은 이미지 pooling에 포함하지 않습니다.
`h_q`는 chat template과 generation prompt를 포함한 입력의 마지막 유효 토큰입니다.

레이어 번호는 [Hugging Face `hidden_states` 인덱스](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L805-L831) 기준입니다. 9·18·27은 해당 Decoder 블록 출력,
36은 마지막 블록 뒤 최종 normalization 출력입니다. 모델을 `eval()` 모드로 두며,
기본 mean pooling은 float32로 계산합니다. 기존 구현이 모델 dtype에서 평균을 계산했다면
`step2.yaml`의 `pooling_dtype`을 `model`로 지정할 수 있습니다. 저장되는 pooled 벡터는 float32입니다.

먼저 동일 입력 반복 추출과 표준 `hidden_states` 출력을 비교하는 명령입니다.

```bash
python -m experiments.step2_hidden_states.run --category person --limit 1 --verify-repeat --verify-reference
```

`--verify-repeat`은 같은 케이스를 다시 전처리·추론해 입력 토큰, 특징, 응답과 logit을 비교합니다.
`--verify-reference`는 같은 생성 과정의 `hidden_states` 첫 번째 스텝과 선택 레이어의 pooled 특징을 비교합니다.
검증 결과에는 완전 일치 여부, 허용 오차 내 일치 여부와 최대 절대 오차를 저장합니다.
표준 출력 검증은 모든 레이어의 Hidden State를 보관하므로 추가 메모리가 필요합니다.

10개 케이스 또는 설정에 지정된 A/B 전체 케이스를 추출할 수 있습니다.

```bash
python -m experiments.step2_hidden_states.run --limit 10
python -m experiments.step2_hidden_states.run
```

전체 추출은 기존 케이스 JSONL을 순서대로 처리합니다. 기본 설정의 예상 수는 A 1,600개와 B 1,200개입니다.
샘플 하나씩 추출·저장하며, 여러 사례가 같은 입력을 사용하더라도 각 사례의 기록을 보존합니다.
`--category person`은 현재 케이스에서 Person 전체를 선택하며, 과거의 진단용 35개를 자동 복원하지 않습니다.
기존 35개 사례를 재현하려면 해당 JSONL 파일을 `--cases-jsonl`로 지정합니다.

저장 위치는 기본적으로 `results/step2/qwen2_5_vl_3b/`입니다.
클래스나 개수 제한을 사용하면 그 아래에 `person/`, `pilot_10/` 등의 폴더를 사용합니다.
재실행할 때는 `--output-dir`로 새 폴더를 지정합니다.

```text
results/step2/qwen2_5_vl_3b/
├── run.json
├── manifest.jsonl
└── raw/
    ├── 000000.pt
    └── ...
```

- `run.json`: 적용 설정, 입력 파일 해시, 라이브러리 버전, 모델·processor 설정, 처리 수와 완료 상태
- `manifest.jsonl`: 원본 케이스, 특징 파일 위치, 이미지 토큰 구간, Yes/No 응답, logit과 검증 결과
- 각 `.pt`: 원본 케이스, 입력 토큰·mask·grid, `image_features[4, 3, 2048]`, `query_features[4, 2048]`

pooling 전 이미지 토큰 벡터도 필요하면 `--save-token-states`를 추가합니다.
선택 레이어마다 이미지별 `[토큰 수, 2048]` 텐서를 모델 dtype으로 저장하며, 디스크 사용량이 늘어납니다.

동일 입력 반복 검증과 과거 결과 재현은 별개입니다. 과거 결과와 비교할 때는 기존 케이스 목록,
모델·processor·라이브러리 버전, dtype, 레이어 정의와 pooling 방식을 맞춰야 합니다.
Cosine Distance 분석과 거리 그래프 생성은 아직 구현하지 않았습니다.

### 이미지와 토큰 구간 시각화

추출 결과의 `manifest.jsonl`과 원본 이미지만 읽어 토큰 구간 그림을 저장합니다.
모델 로딩이나 재추론, `.pt` 로딩은 하지 않으며 `--save-token-states`도 필요하지 않습니다.
같은 `mode`와 `pair_id`의 케이스를 한 그림에 배치합니다.
A는 PPP·PPN·PNN·NNN 구성 변화, B는 NPP·PNP·PPN 순서 변화를 비교합니다.
같은 이미지 ID는 그림 안에서 같은 색을 사용하며, 마지막 입력 토큰 `h_q`는 점선으로 표시합니다.

A/B를 각각 10개 추출한 폴더의 시각화 명령입니다. 두 추출 결과가 저장된 뒤 실행합니다.
하나만 시각화하려면 해당 manifest 경로만 지정합니다.

```bash
python -m experiments.step2_hidden_states.visualize \
  --manifest-jsonl \
    results/step2/qwen2_5_vl_3b/pilot_A_10/manifest.jsonl \
    results/step2/qwen2_5_vl_3b/pilot_B_10/manifest.jsonl
```

기본 저장 경로는 `reports/step2/qwen2_5_vl_3b/pilot_A_10/token_spans/A/person/<pair_id>.png`
형태이며, B 결과는 대응하는 `pilot_B_10/token_spans/B/` 아래에 저장합니다.
`--category person`으로 클래스를 제한하고, `--limit 1`로 첫 비교 묶음의 그림 하나만 저장할 수 있습니다.
`--limit`은 이 명령에서는 케이스 수가 아니라 전체 manifest에 걸친 그림 수입니다.
`results/` 밖의 manifest를 사용하려면 `--output-dir`로 `reports/` 아래 저장 위치를 지정합니다.

파일럿이 비교 묶음 중간에서 끝나면 저장된 케이스만 표시하고, 제목에 `2/4 saved cases`처럼
실제로 포함된 수를 표시합니다. 저장되지 않은 순서나 케이스를 새로 생성하지 않습니다.
이 그림은 저장된 이미지·토큰 구간의 대응을 보여주며, 단일 이미지 특징의 일치 여부를 검증하는 그림은 아닙니다.
