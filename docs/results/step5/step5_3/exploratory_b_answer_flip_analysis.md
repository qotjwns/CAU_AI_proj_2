# STEP 5-3 실험 B 추가 분석: N 위치별·장수별 실제 답 변화율 (EXPLORATORY)

> **Exploratory.** 사전 등록 판정과 무관하다. 확증 판정(B-H1 inconclusive, B-H2 robust support)과 고정본 v3는 바뀌지 않는다.
> 새 GPU 실행·모델 추론·개입·기존 평가 재실행 없이, 저장된 입력별 결과(`attention/main/conditions.jsonl`)와
> `attention/evaluation/analysis.json`만 읽었다.

## 조건

- 대상: 확증 절반, 선택 구간 L21–24 (강화·차단 모두 선택 절반에서 고른 구간), 개입 위치 = 이미지 뒤 모든 토큰
- 강화: 기준 False-Yes 입력에서 Yes → No 비율 (실제 교정률). 차단: 기준 Correct-No 입력에서 No → Yes 비율 (실제 오답 전환율)
- N 대상과 P 대조는 같은 입력의 두 개입 행이다. P 대조는 계획 단계에서 정한 P 한 장이다:
  이미지 토큰 수가 N과 가장 가까운 P, 동률이면 위치가 가까운 것, 그래도 같으면 앞쪽.
- 분모: 기존 제외 규칙(이 경로의 무개입 답이 기준 답과 다른 입력 제외)을 적용한 뒤 남은 입력.
  개입 뒤 답이 Yes/No가 아닌 행은 바뀌지 않은 것으로 센다 (기존 평가와 같음). 그런 행의 수는 CSV의 `n_answer_not_yes_no_*`에 있다.
- Δm = 개입 margin − 같은 입력 무개입 margin (fp32 변형 margin, logit). C = Δm(N) − Δm(P).
- Middle = N이 첫 자리도 마지막 자리도 아닌 모든 위치.

## 신뢰구간

- 기존 평가와 같은 함수(`causal_eval.summarize`)를 썼다. N 이미지 군집 percentile 부트스트랩이고, 클래스 군집도 함께 낸다.
  반복 1000회, seed 42, 수준 95%이며, 모두 기존 부가 분석과 같다.
- 부트스트랩 표본은 따로 저장돼 있지 않다. 다만 (군집 집합, 반복 수, seed)가 같으면 같은 표본이 다시 만들어진다.
  기존 표에 이미 있던 값(전체 답 변화율, 위치·장수별 C·Δm(N)·Δm(P))이 소수점까지 똑같이 나오는 것으로 확인했다 (아래 검증).
- 한 하위 집단 안의 모든 지표(N, P, N − P, Δm, C)는 같은 재표본을 쓴다. N − P 차이는 입력별 차이(짝 유지)의 평균이다.
- 하위 집단마다 그 집단의 N 이미지·클래스로 재표본을 새로 만든다. 이것도 기존 부가 분석과 같다.
  따라서 하위 집단 사이에서는 재표본이 공유되지 않는다.
- 군집이 2개 미만이면 CI를 N/A로 둔다. 이번 결과에는 해당 사례가 없다.

## N 위치별 (EXPLORATORY)

### 강화 (Boost): 기준 False-Yes → No 교정률

| 집단 | 입력 / N 이미지 / 클래스 | 제외 | N 대상 변화율 [95% CI] | P 대조 변화율 [95% CI] | N − P [95% CI] | N − P 클래스 CI |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2453 / 210 / 20 | 48 | 0.375 [0.347, 0.406] (921) | 0.084 [0.072, 0.097] (205) | +0.292 [+0.263, +0.320] | [+0.244, +0.338] |
| first | 595 / 197 / 20 | 14 | 0.020 [0.007, 0.039] (12) | 0.222 [0.185, 0.266] (132) | -0.202 [-0.246, -0.164] | [-0.277, -0.145] |
| middle | 1404 / 206 / 20 | 22 | 0.419 [0.381, 0.456] (588) | 0.033 [0.023, 0.044] (46) | +0.386 [+0.349, +0.423] | [+0.321, +0.462] |
| last | 454 / 160 / 20 | 12 | 0.707 [0.642, 0.773] (321) | 0.059 [0.037, 0.084] (27) | +0.648 [+0.586, +0.712] | [+0.514, +0.779] |

| 집단 | Δm(N) [95% CI] | Δm(P) [95% CI] | C [95% CI] | C 클래스 CI |
| --- | --- | --- | --- | --- |
| all | -0.662 [-0.704, -0.621] | +0.018 [-0.007, +0.043] | -0.680 [-0.732, -0.627] | [-0.806, -0.558] |
| first | -0.025 [-0.047, -0.001] | -0.258 [-0.308, -0.208] | +0.233 [+0.177, +0.293] | [+0.149, +0.323] |
| middle | -0.751 [-0.791, -0.706] | +0.079 [+0.057, +0.102] | -0.830 [-0.883, -0.777] | [-0.979, -0.700] |
| last | -1.225 [-1.315, -1.147] | +0.188 [+0.151, +0.229] | -1.413 [-1.523, -1.321] | [-1.621, -1.188] |

### 차단 (Block): 기준 Correct-No → Yes 오답 전환율

| 집단 | 입력 / N 이미지 / 클래스 | 제외 | N 대상 변화율 [95% CI] | P 대조 변화율 [95% CI] | N − P [95% CI] | N − P 클래스 CI |
| --- | --- | --- | --- | --- | --- | --- |
| all | 3224 / 237 / 20 | 35 | 0.171 [0.151, 0.193] (551) | 0.037 [0.029, 0.044] (118) | +0.134 [+0.115, +0.155] | [+0.095, +0.185] |
| first | 824 / 223 / 20 | 7 | 0.011 [0.004, 0.019] (9) | 0.101 [0.079, 0.125] (83) | -0.090 [-0.115, -0.066] | [-0.123, -0.058] |
| middle | 1436 / 216 / 20 | 18 | 0.164 [0.141, 0.190] (236) | 0.019 [0.012, 0.029] (28) | +0.145 [+0.120, +0.172] | [+0.100, +0.199] |
| last | 964 / 231 / 20 | 10 | 0.317 [0.283, 0.353] (306) | 0.007 [0.003, 0.013] (7) | +0.310 [+0.274, +0.347] | [+0.229, +0.408] |

| 집단 | Δm(N) [95% CI] | Δm(P) [95% CI] | C [95% CI] | C 클래스 CI |
| --- | --- | --- | --- | --- |
| all | +0.316 [+0.289, +0.347] | -0.163 [-0.189, -0.137] | +0.479 [+0.435, +0.527] | [+0.387, +0.599] |
| first | -0.072 [-0.084, -0.058] | +0.055 [+0.018, +0.094] | -0.127 [-0.170, -0.086] | [-0.178, -0.065] |
| middle | +0.311 [+0.280, +0.339] | -0.181 [-0.211, -0.152] | +0.491 [+0.444, +0.538] | [+0.401, +0.597] |
| last | +0.655 [+0.604, +0.713] | -0.324 [-0.361, -0.286] | +0.979 [+0.906, +1.061] | [+0.816, +1.184] |

- 괄호 안 정수는 답이 바뀐 입력 수다. 변화율의 클래스 CI는 CSV에 있다.

## 이미지 장수별 (EXPLORATORY)

### 강화 (Boost): 기준 False-Yes → No 교정률

| 집단 | 입력 / N 이미지 / 클래스 | 제외 | N 대상 변화율 [95% CI] | P 대조 변화율 [95% CI] | N − P [95% CI] | N − P 클래스 CI |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2453 / 210 / 20 | 48 | 0.375 [0.347, 0.406] (921) | 0.084 [0.072, 0.097] (205) | +0.292 [+0.263, +0.320] | [+0.244, +0.338] |
| 3 | 437 / 157 / 19 | 7 | 0.460 [0.420, 0.504] (201) | 0.158 [0.127, 0.190] (69) | +0.302 [+0.251, +0.353] | [+0.252, +0.358] |
| 4 | 755 / 179 / 20 | 19 | 0.454 [0.412, 0.491] (343) | 0.086 [0.066, 0.108] (65) | +0.368 [+0.324, +0.406] | [+0.316, +0.422] |
| 5 | 1261 / 196 / 20 | 22 | 0.299 [0.268, 0.332] (377) | 0.056 [0.042, 0.071] (71) | +0.243 [+0.214, +0.274] | [+0.184, +0.304] |

| 집단 | Δm(N) [95% CI] | Δm(P) [95% CI] | C [95% CI] | C 클래스 CI |
| --- | --- | --- | --- | --- |
| all | -0.662 [-0.704, -0.621] | +0.018 [-0.007, +0.043] | -0.680 [-0.732, -0.627] | [-0.806, -0.558] |
| 3 | -0.770 [-0.836, -0.715] | -0.039 [-0.097, +0.012] | -0.731 [-0.810, -0.652] | [-0.877, -0.581] |
| 4 | -0.745 [-0.795, -0.693] | +0.032 [-0.003, +0.064] | -0.777 [-0.844, -0.707] | [-0.914, -0.631] |
| 5 | -0.576 [-0.617, -0.536] | +0.029 [+0.005, +0.053] | -0.604 [-0.656, -0.553] | [-0.733, -0.481] |

### 차단 (Block): 기준 Correct-No → Yes 오답 전환율

| 집단 | 입력 / N 이미지 / 클래스 | 제외 | N 대상 변화율 [95% CI] | P 대조 변화율 [95% CI] | N − P [95% CI] | N − P 클래스 CI |
| --- | --- | --- | --- | --- | --- | --- |
| all | 3224 / 237 / 20 | 35 | 0.171 [0.151, 0.193] (551) | 0.037 [0.029, 0.044] (118) | +0.134 [+0.115, +0.155] | [+0.095, +0.185] |
| 3 | 991 / 233 / 20 | 5 | 0.249 [0.219, 0.283] (247) | 0.049 [0.036, 0.064] (49) | +0.200 [+0.170, +0.233] | [+0.137, +0.271] |
| 4 | 1132 / 214 / 20 | 14 | 0.158 [0.131, 0.184] (179) | 0.035 [0.024, 0.048] (40) | +0.123 [+0.097, +0.149] | [+0.083, +0.177] |
| 5 | 1101 / 182 / 17 | 16 | 0.114 [0.091, 0.137] (125) | 0.026 [0.018, 0.037] (29) | +0.087 [+0.066, +0.109] | [+0.057, +0.128] |

| 집단 | Δm(N) [95% CI] | Δm(P) [95% CI] | C [95% CI] | C 클래스 CI |
| --- | --- | --- | --- | --- |
| all | +0.316 [+0.289, +0.347] | -0.163 [-0.189, -0.137] | +0.479 [+0.435, +0.527] | [+0.387, +0.599] |
| 3 | +0.475 [+0.432, +0.520] | -0.219 [-0.262, -0.174] | +0.694 [+0.622, +0.765] | [+0.551, +0.854] |
| 4 | +0.305 [+0.274, +0.337] | -0.155 [-0.193, -0.119] | +0.460 [+0.405, +0.517] | [+0.363, +0.574] |
| 5 | +0.184 [+0.163, +0.208] | -0.121 [-0.152, -0.092] | +0.305 [+0.265, +0.346] | [+0.250, +0.377] |

- 괄호 안 정수는 답이 바뀐 입력 수다. 변화율의 클래스 CI는 CSV에 있다.

## 기존 전체 결과와의 일치 검증

| 검사 | 결과 |
| --- | --- |
| position labels: first = slot 0, last = final slot, middle = every other slot | 일치 |
| boost: N and P rows present for every analysed input | 일치 |
| boost position: subgroups add up to the overall counts | 일치 |
| boost n_images: subgroups add up to the overall counts | 일치 |
| boost overall flip_N = stored secondary answer change rate (N) | 일치 |
| boost overall flip_P = stored secondary answer change rate (P) | 일치 |
| boost N position first C = stored secondary | 일치 |
| boost N position first dm_N = stored secondary | 일치 |
| boost N position first dm_P = stored secondary | 일치 |
| boost N position middle C = stored secondary | 일치 |
| boost N position middle dm_N = stored secondary | 일치 |
| boost N position middle dm_P = stored secondary | 일치 |
| boost N position last C = stored secondary | 일치 |
| boost N position last dm_N = stored secondary | 일치 |
| boost N position last dm_P = stored secondary | 일치 |
| boost 3 images C = stored secondary | 일치 |
| boost 3 images dm_N = stored secondary | 일치 |
| boost 3 images dm_P = stored secondary | 일치 |
| boost 4 images C = stored secondary | 일치 |
| boost 4 images dm_N = stored secondary | 일치 |
| boost 4 images dm_P = stored secondary | 일치 |
| boost 5 images C = stored secondary | 일치 |
| boost 5 images dm_N = stored secondary | 일치 |
| boost 5 images dm_P = stored secondary | 일치 |
| boost overall C estimate and n = stored primary (97.5% CI not compared) | 일치 |
| boost overall dm_N estimate and n = stored primary (97.5% CI not compared) | 일치 |
| boost overall dm_P estimate and n = stored primary (97.5% CI not compared) | 일치 |
| boost excluded inputs = stored primary | 일치 |
| block: N and P rows present for every analysed input | 일치 |
| block position: subgroups add up to the overall counts | 일치 |
| block n_images: subgroups add up to the overall counts | 일치 |
| block overall flip_N = stored secondary answer change rate (N) | 일치 |
| block overall flip_P = stored secondary answer change rate (P) | 일치 |
| block N position first C = stored secondary | 일치 |
| block N position first dm_N = stored secondary | 일치 |
| block N position first dm_P = stored secondary | 일치 |
| block N position middle C = stored secondary | 일치 |
| block N position middle dm_N = stored secondary | 일치 |
| block N position middle dm_P = stored secondary | 일치 |
| block N position last C = stored secondary | 일치 |
| block N position last dm_N = stored secondary | 일치 |
| block N position last dm_P = stored secondary | 일치 |
| block 3 images C = stored secondary | 일치 |
| block 3 images dm_N = stored secondary | 일치 |
| block 3 images dm_P = stored secondary | 일치 |
| block 4 images C = stored secondary | 일치 |
| block 4 images dm_N = stored secondary | 일치 |
| block 4 images dm_P = stored secondary | 일치 |
| block 5 images C = stored secondary | 일치 |
| block 5 images dm_N = stored secondary | 일치 |
| block 5 images dm_P = stored secondary | 일치 |
| block overall C estimate and n = stored primary (97.5% CI not compared) | 일치 |
| block overall dm_N estimate and n = stored primary (97.5% CI not compared) | 일치 |
| block overall dm_P estimate and n = stored primary (97.5% CI not compared) | 일치 |
| block excluded inputs = stored primary | 일치 |
| frozen v3 files unchanged before and after this analysis | 일치 |

## N-first 역방향 대조 효과에 대한 해석상 주의 (EXPLORATORY)

- 관찰: 강화에서 N-first의 C는 +0.233 [+0.177, +0.293]로, 사전 등록 방향(C < 0)과 반대다.
  - 이 값은 주로 P 쪽에서 나온다: Δm(N) -0.025 [-0.047, -0.001], Δm(P) -0.258 [-0.308, -0.208].
  - 실제 답도 P 대조에서 더 많이 바뀌었다: Yes → No 비율이 N 0.020, P 0.222. 차단도 마찬가지로 No → Yes 비율이 N 0.011, P 0.101이다.
  - 차단에서는 N-first의 C가 -0.127 [-0.170, -0.086], Δm(N)이 -0.072 [-0.084, -0.058]로, 역시 예측과 반대 부호다.
- 이 결과를 'N-first에서는 N 이미지 attention이 답에 반대로 작용한다'로 읽으면 안 되는 이유:
  1. **사전 등록 판정 대상이 아니다.** 위치·장수 하위 집단은 부가 분석이다. 2개 개입 × 6개 집단을 보면서 다중 비교를 보정하지 않았다.
  2. **구간을 위치를 모두 합쳐 골랐다.** L21–24는 선택 절반의 전체 입력에서 고른 구간이고, middle·last 입력이 대부분이다.
     N-first에서 효과가 가장 큰 구간은 다를 수 있다. 위치별 구간 곡선은 이번에 계산하지 않았다.
  3. **P 대조의 위치가 N과 체계적으로 다르다.** N-first에서는 P 대조가 항상 N보다 뒤에 있다
     (강화 595/595, 차단 824/824).
     N-last에서는 항상 앞에 있다 (강화 454/454, 차단 964/964). middle에서는 섞여 있다 (뒤: 강화 517/1404, 차단 601/1436).
     그래서 C에는 'N이냐 P냐'의 차이와 '어느 자리의 이미지를 건드렸느냐'의 차이가 섞여 있다.
     뒤쪽 이미지에 대한 개입이 margin을 더 크게 움직이는 위치 효과만으로도 N-first의 C 부호가 바뀔 수 있다.
  4. **N-first에서는 N 개입 자체의 효과가 작다** (강화 Δm(N) -0.025, middle -0.751, last -1.225). 그래서 강화 N-first의 C 부호는 사실상 P 대조 쪽 값으로 정해진다.
     P 대조는 'N과 무관한 같은 크기의 개입'이라는 기준선이다. 위치가 다르면 이 기준선 역할이 약해진다.
  5. **개입은 attention 확률만 바꾼다.** 한 이미지를 강화하면 다른 토큰들의 attention 몫이 줄고, 차단하면 늘어난다.
     따라서 P 강화의 효과에는 N을 포함한 나머지 이미지의 몫이 줄어든 영향이 함께 들어 있다.
     이 분석으로는 둘을 나눌 수 없다.
- 이 관찰을 확인하려면 개입 실험을 새로 설계해야 한다. 예: P 대조를 N과 같은 쪽 인접 위치로 맞춘 대조, 위치별 구간 선택.
  이번 작업 범위(새 실행 금지) 밖이다.

## 재현

- 코드: `experiments/step5_causal/explore_answer_flip.py` (sha256 f0d7da742e9fe889…)
- 입력: 고정본 `results/step5/qwen2_5_vl_3b/attention/frozen/v3` (manifest sha256 9850639c64114327…), 본 실행 `results/step5/qwen2_5_vl_3b/attention/main` (실행 고정본 v2), 평가 2026-11-13T16:46:51+00:00
- 실행 시각 2026-11-14T03:39:47+00:00
