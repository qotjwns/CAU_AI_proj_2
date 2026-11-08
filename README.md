# 다중 이미지 객체 존재 판단 실험

COCO 이미지로 Qwen2.5-VL이 세 이미지 모두에 특정 객체가 있는지 판단하는 능력을 비교합니다.

- **A — 한 장 교체:** PPP → PPN → PNN → NNN
- **B — 순서 변경:** 같은 이미지로 NPP · PNP · PPN 비교

데이터 생성, 클래스별 시각화, baseline 추론, 결과 메트릭 계산과 시각화를 지원합니다.

설정과 생성된 JSON·JSONL에 기록하는 경로는 프로젝트 루트 기준 상대경로이며,
경로 구분자는 `/`로 통일합니다. 프로젝트 폴더를 옮겨도 같은 디렉터리 구조로 사용할 수 있습니다.

## 메트릭 계산

추론 결과 JSONL을 입력합니다. 모델을 다시 로드하지 않으며, A/B 파일 중 하나만 넣어도 됩니다.
입력은 같은 모델·추론 방법의 결과여야 합니다. 경로는 프로젝트 루트 기준입니다.

현재 baseline 설정의 A/B 전체 명령은 다음과 같습니다.

```powershell
.\.venv\Scripts\python.exe src/evaluation/compute_metrics.py --results-jsonl reports/inference/baseline/qwen2_5_vl_3b/inference_groups_top20_seed42_A_results.jsonl reports/inference/baseline/qwen2_5_vl_3b/inference_groups_top20_seed42_B_results.jsonl --output-dir reports/metrics/baseline/qwen2_5_vl_3b
```

출력 디렉터리에 `metrics.json`과 다음 CSV를 저장합니다. 해당 항목이 없는 CSV는 생성하지 않습니다.

| 파일 | 내용 |
| --- | --- |
| `overall.csv` | 입력된 모든 케이스의 정확도·응답 형식 준수율·평균 토큰 수 |
| `by_mode.csv` | A/B 모드별 메트릭 |
| `by_class.csv` | 모드·클래스별 메트릭 |
| `by_composition.csv` | 모드·PPP/PPN 등 구성별 메트릭 |
| `by_negative_position.csv` | B 모드의 음성 이미지 위치별 메트릭, 위치는 1부터 시작 |
| `pair_summary.csv` | 완전/불완전 pair 수, 전체 구성 정답률, B 모드 응답 일치율 |
| `pair_transitions.csv` | 구성 변경별 예측 변화율과 평균 logit margin 변화 |
| `pair_details.csv` | 개별 pair의 구성 변경 전후 예측과 logit margin |

정확도는 `prediction`과 `gt`로 다시 계산합니다. Yes/No 이외의 응답은 오답이며,
혼동행렬에서는 별도의 `Invalid` 열로 집계합니다. 비율은 0~1이며 분모가 0인 값은
JSON에서 `null`, CSV에서는 빈칸입니다. B 모드는 정답이 모두 No이므로 Yes recall과
balanced accuracy가 `null`입니다. `false_positive_rate`는 정답 No 중 Yes 응답 비율이며,
`error_rate`는 형식 오류까지 포함합니다.

A/B를 함께 넣은 전체 정확도는 입력된 행을 합친 값입니다. 두 모드에 공유되는 케이스도
모드별로 각각 집계하며, 동일 모드 안의 중복 `case_id`는 오류로 처리합니다.

Pair 비교는 A의 PPP/PPN/PNN/NNN 또는 B의 NPP/PNP/PPN이 모두 있는 pair만 사용합니다.
누락된 pair 수를 별도로 기록하며, 전체 케이스 메트릭에는 그 케이스들도 포함합니다.
A는 인접 구성 3쌍, B는 모든 구성 조합 3쌍을 비교합니다.
예측 변화율은 비교하는 두 응답 모두 Yes/No인 pair를 분모로 사용합니다.
B의 `order_agreement_rate`는 세 응답이 모두 유효한 완전 pair에서 응답이 모두 같은 비율입니다.
응답이 일치해도 모두 오답일 수 있으므로 `all_correct_rate`와 함께 봅니다.
A에서는 PPP와 나머지 구성의 정답이 다르므로 예측 변화 자체를 오류로 해석하지 않습니다.

## 메트릭 시각화

계산된 `metrics.json`을 읽어 차트를 저장합니다.

```powershell
.\.venv\Scripts\python.exe src/visualize/plot_metrics.py --metrics-json reports/metrics/baseline/qwen2_5_vl_3b/metrics.json
```

기본 출력은 메트릭 디렉터리 아래 `plots/`이며, `--output-dir`로 바꿀 수 있습니다.
전체 정확도와 형식 오류율, 클래스별·구성별 정확도, 혼동행렬, B의 음성 위치별 오답률,
정답/오답의 logit margin 분포, pair의 예측·margin 변화 PNG를 생성합니다.
해당 모드나 완전 pair가 없으면 관련 차트를 생략합니다. 표본이 없는 막대는 `N/A`로 표시합니다.

Logit margin은 첫 출력 토큰 위치의 `Yes logit - No logit`입니다.
Pair의 margin 변화는 뒤 구성에서 앞 구성 값을 뺀 값이며, 양수이면 Yes 쪽 점수가
상대적으로 증가했다는 뜻입니다. 예측은 실제 생성 응답에서 얻으므로 margin 부호와
반드시 일치하지는 않습니다. 메트릭과 차트는 같은 출력 경로로 다시 실행하면 덮어씁니다.
현재 입력에 해당 항목이 없으면 이 스크립트가 관리하는 이전 CSV·선택 차트도 제거합니다.
