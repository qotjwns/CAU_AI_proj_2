# 다중 이미지 객체 존재 판단 실험

Qwen2.5-VL-3B에 이미지 3장과 "객체가 세 이미지 모두에 있는가?"를 묻고, 오답(False-Yes)과 내부 표현을 분석합니다.
모든 명령은 가상환경을 켠 뒤 프로젝트 루트에서 실행합니다. `[GPU]` 표시가 없는 단계는 CPU에서 돌아갑니다.

## 연구 개요

- **목적:** VLM이 N 이미지를 놓치고 Yes로 틀릴 때(False-Yes), 내부 Hidden State에서는 이미지별 객체 정보가 판별 가능한지 확인하고,
  최종적으로 그 정보가 출력에 반영되도록 개입할 수 있는지 검증합니다.
- **데이터:** COCO val2017 + COCO-REM 주석, 20개 클래스. P = 대상 객체 bbox가 이미지 면적의 5% 이상(crowd 제외), N = 대상 클래스 주석 없음.
  - STEP 1~3: 클래스당 P 8장·N 8장(원본 이미지 310장), 레코드 2,800 → 고유 입력 2,332 (분모 기준)
  - STEP 4: STEP 1~3 이미지를 전부 제외한 새 이미지 960장(클래스당 P 24·N 24), 레코드 2,080 → 고유 입력 1,920
- **분할:** STEP 3은 전역 image_id 4-Fold OOF(학습 벡터의 prefix에도 평가 이미지 제외). STEP 4는 STEP 3 전체로 학습한 Probe를 고정해 새 이미지에 적용.
- **주 지표 (STEP 4 사전 지정, `configs/experiments/step4.yaml`)**
  - H1-A: z_i L36 Strict Discrepancy = 세 이미지 모두 맞힌 False-Yes / VLM False-Yes
  - H1-B: h_q L27 통합 readout의 N 위치 정확도 = N 위치를 맞힌 입력 / PPN·PNP·NPP 입력 (우연 1/3)
  - 비교: False-Yes vs Correct-No(클래스·구성 맞춤, 같은 N 이미지), 같은 입력에서 z_i vs h_q
  - 부가: Probe AND 정확도, PPP 유지율 / 통계: 원본 image_id 클러스터 부트스트랩 95% CI (1,000회)
- **한계**
  - STEP 3은 클래스당 원본 이미지 16장 기반의 탐색적 결과이며, CI가 넓고 클래스 편차가 큼
  - STEP 4도 클래스당 24P+24N의 최소 파일럿 규모라 클래스별 검정력은 보장하지 않음
  - Probe·readout의 판별 가능성은 VLM이 그 정보를 실제로 사용했다는 증거가 아님 (인과는 STEP 5 개입으로 확인)
  - h_q readout은 STEP 4 관찰용이며, STEP 5의 개입용 결정 readout `s_i = f(h_q, z_i)`와 다름
  - N 2장 구성은 PNN뿐이라 h_q 위치 판별은 N 1장 구성의 N 위치 정확도로 판단
  - 라벨 검사는 생성과 같은 주석을 쓰는 일관성 검사이며 독립적인 정답 검증이 아님

## STEP 0. 준비

```bash
python -m pip install -r requirements.txt
python -m scripts.download_data                                              # COCO val2017 + COCO-REM
python -m scripts.download_model --config configs/models/qwen2_5_vl_3b.yaml

# 클래스별 P/N 후보 → 그룹 → A/B 케이스 (20클래스, A 1,600 / B 1,200)
python -m scripts.build_pools --config configs/data/balanced3.yaml
python -m scripts.build_groups --config configs/data/balanced3.yaml
python -m scripts.build_cases --groups-json data/balanced3/inference_pools/inference_groups_top20_seed42.json --mode A
python -m scripts.build_cases --groups-json data/balanced3/inference_pools/inference_groups_top20_seed42.json --mode B
```

## STEP 1. Baseline 추론

```bash
# [GPU] A/B 전체 추론 → results/step1/  (--limit N: 파일럿)
python -m experiments.step1_baseline.run --config configs/experiments/step1.yaml

# 지표 집계 → results/step1/metrics/, 그래프 → reports/step1/
python -m experiments.step1_baseline.analyze --results-jsonl \
  results/step1/inference_groups_top20_seed42_A_results.jsonl \
  results/step1/inference_groups_top20_seed42_B_results.jsonl
python -m experiments.step1_baseline.visualize --metrics-json results/step1/metrics/metrics.json

# 입력 이미지 조합 그림 → reports/step1/cases/
python -m experiments.step1_baseline.visualize_cases --cases-jsonl \
  data/balanced3/inference_cases/inference_groups_top20_seed42_A_cases.jsonl \
  data/balanced3/inference_cases/inference_groups_top20_seed42_B_cases.jsonl
```

## STEP 2. Hidden State 추출과 거리 분석

```bash
# [GPU] 사전 점검: 반복 추출·표준 hidden_states와 일치하는지
python -m experiments.step2_hidden_states.run --category person --limit 1 --verify-repeat --verify-reference

# [GPU] 전체 추출 (L9·18·27·36, 이미지별 z_i + 마지막 토큰 h_q) → results/step2/qwen2_5_vl_3b/
python -m experiments.step2_hidden_states.run

# 저장 특징 검증 + STEP 1 응답·logit 비교
python -m experiments.step2_hidden_states.validate --run-dir results/step2/qwen2_5_vl_3b --step1-results \
  results/step1/inference_groups_top20_seed42_A_results.jsonl \
  results/step1/inference_groups_top20_seed42_B_results.jsonl

# 이미지·토큰 구간 그림
python -m experiments.step2_hidden_states.visualize --manifest-jsonl results/step2/qwen2_5_vl_3b/manifest.jsonl

# 동일 N1 이미지의 문맥별 cosine distance + 그래프
python -m experiments.step2_hidden_states.analyze --run-dir results/step2/qwen2_5_vl_3b
python -m experiments.step2_hidden_states.visualize_distances \
  --analysis-json results/step2/qwen2_5_vl_3b/analysis/analysis.json
```

## STEP 3. Linear Probe (탐색적)

```bash
# 20클래스 × 4레이어 × 4-Fold(전역 image_id) Probe, OOF 평가 → results/step3/qwen2_5_vl_3b/
# Strict Discrepancy, VLM 응답 그룹 비교, 이미지 부트스트랩 CI, 라벨 순열 100회(약 7분)
python -m experiments.step3_linear_probe.run              # 빠른 확인: --permutation-runs 0
python -m experiments.step3_linear_probe.visualize --analysis-json results/step3/qwen2_5_vl_3b/analysis.json
```

- 결과 상태 `exploratory`, 기존 Person 800/200 결과는 `configs/experiments/step3_reference.yaml`에 `reference_only`

## STEP 4. 독립 이미지 검증

순서: **데이터·라벨 검사 → 고정 → GPU 파일럿 → 무결성 검사 → GPU 본 실행 → 평가.**
고정 이후에는 케이스·Probe·임계값·레이어를 바꾸지 않습니다. 파일럿에서 수정이 필요하면 기존 고정을 고치지 말고
`step4.yaml`의 `frozen_dir`을 `v2`로 바꿔 `--reason`과 함께 새로 고정합니다.

```bash
# 4-1. 독립 데이터 생성 (STEP 1~3 이미지 310장 제외, 클래스당 P24+N24, A/B/C 모드)
#      → data/step4_independent/step4_seed2026_cases.jsonl, _groups.json (덮어쓰기 금지)
python -m scripts.build_step4_cases
#      Annotation Consistency Check: ID·클래스·P/N 조건·이미지 순서·생성 규칙 일치 → _consistency_check.json
python -m scripts.check_step4_cases

# 4-2. 고정: z_i Probe + h_q readout 학습, 평가 케이스·학습 데이터 전체·설정·사전 지정 해시 기록
#      → results/step4/qwen2_5_vl_3b/frozen/v1/ (덮어쓰기 금지, 새 버전은 --reason 필수)
python -m experiments.step4_independent.freeze

# 4-3. [GPU] 파일럿: 첫 13개 = bed pair 1개, A·B·C 모드와 모든 구성 포함 (평가 지표는 보지 않음)
python -m experiments.step2_hidden_states.run --config configs/experiments/step4_extract.yaml \
  --limit 13 --verify-repeat --verify-reference --output-dir results/step4/qwen2_5_vl_3b/pilot

# 4-4. 파일럿 무결성 검사: 케이스·이미지 순서·토큰 구간·h_q 위치·응답/z_i/h_q 연결·추출 설정·고정 파일
python -m experiments.step4_independent.check_pilot --run-dir results/step4/qwen2_5_vl_3b/pilot

# 4-5. [GPU] 본 추출 + 저장 특징 검증
python -m experiments.step2_hidden_states.run --config configs/experiments/step4_extract.yaml
python -m experiments.step2_hidden_states.validate --run-dir results/step4/qwen2_5_vl_3b/extraction

# 4-6. 고정 Probe 적용 (고정 해시·케이스 해시·전체 실행·추출 설정·이미지 중복·사전 지정 검사 후 실행)
#      평가 후 고정 파일 해시 재확인, 바뀌었으면 INVALID.json 기록 후 실패
python -m experiments.step4_independent.run
python -m experiments.step4_independent.visualize --analysis-json results/step4/qwen2_5_vl_3b/analysis.json
```

- H1-A → `strict_discrepancy.csv`, `group_comparison.csv` / H1-B → `localization.csv` / 같은 입력 비교 → `zi_vs_hq.csv`
- 고정 이력: `frozen/v*/manifest.json` (버전, 이유, 이전 버전 해시), 적용 전후 확인: `frozen_verification.json`

## STEP 5. 사전 진단과 개입

```bash
# h_q Probe 진단 (STEP 3 데이터, STEP 3 이후 언제든) → results/step5_diagnostics/qwen2_5_vl_3b/hq_probe/
# 위치별 P/N 판별(z_i vs h_q), N 위치 정확도, pure(완전 미관측) 입력 분리
python -m experiments.step5_diagnostics.hq_probe
```

- 개입 실험(readout–logit 관계, Hidden State 개입)은 STEP 4 이후 진행

## 결과 위치

| 단계 | 표·텐서 | 그림 |
| --- | --- | --- |
| STEP 1 | `results/step1/` | `reports/step1/` |
| STEP 2 | `results/step2/qwen2_5_vl_3b/` | `reports/step2/qwen2_5_vl_3b/` |
| STEP 3 | `results/step3/qwen2_5_vl_3b/` | `reports/step3/qwen2_5_vl_3b/` |
| STEP 4 | `results/step4/qwen2_5_vl_3b/` (`frozen/`, `pilot/`, `extraction/` 포함) | `reports/step4/qwen2_5_vl_3b/` |
| STEP 5 진단 | `results/step5_diagnostics/qwen2_5_vl_3b/` | `reports/step5_diagnostics/qwen2_5_vl_3b/` |

설정 파일은 `configs/` (데이터 `data/`, 모델 `models/`, 단계별 `experiments/`)에 있습니다.
