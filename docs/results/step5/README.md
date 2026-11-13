# STEP 5-2 · 5-2b 결과 보존본

`results/`·`reports/`는 git에 올라가지 않으므로, 확증 평가의 요약 표·검사 기록·고정 manifest·그림을 여기에 복사해 둔다.
원본은 그대로 두었고, 복사본의 SHA-256은 `SHA256SUMS`에 있다. 대용량 산출물(특징 `.pt`, 입력별 표, 추출 묶음)은 git에 넣지 않는다.

## 기록 시점

| 항목 | 값 |
| --- | --- |
| STEP 5-2 평가 실행 | 2026-11-13T05:58:51Z (사전 등록 `accumulation/frozen/v2`) |
| STEP 5-2b 평가 실행 | 2026-11-13T09:14:45Z (사전 등록 `position/frozen/v1`) |
| 5-2b 평가 코드 해시 | 실행 직전 `step5_2b/eval_code_at_run.sha256`에 기록, 커밋 직전 대조 결과 5개 파일 모두 일치 |
| 커밋 | 두 평가가 끝난 **뒤**에 만든 커밋이다 (시각은 git 커밋 기록 그대로). 5-2 평가 코드는 평가 후 `890f0e6`로 커밋 |
| 고정본 확인 (커밋 직전) | STEP 4 v1, STEP 5-2 v2, STEP 5-2b v1 해시 변경 없음 |

## STEP 5-2 (P 누적, confirmatory)

- 사전 등록 H1(두 배치 모두 False-Yes P4 − P1 지지)은 충족되지 않았다. **N-last에서만 지지**(robust support),
  N-first는 결론 유보(최소 효과 0.05 미달).
  - N-last: P4 − P1 = +0.259 [+0.222, +0.298], 클래스 CI [+0.146, +0.379]; margin +0.870
  - N-first: P4 − P1 = +0.045 [+0.010, +0.077], 클래스 CI [−0.020, +0.110]; margin +0.222
- All-P 대조군은 이미지가 늘수록 Yes 비율이 낮아짐(−0.09). 차이의 차이(margin)는 두 배치 모두 양수.
- H2(같은 N 이미지의 High − Low)는 크고 P 장수와 함께 커짐. 단, **High/Low 차이는 물체 크기·현저성과 구분되지 않는다**
  (High-P 이미지의 대상 bbox가 더 큼). "증거 강도"라는 표현에는 이 단서를 붙인다.
- 탐색(판정과 무관): `step5_2/exploratory/` — a에 따른 z_N 변화와 P 대조군, 장수별 Yes 격차, 배치 상호작용,
  h_q readout 기하. All-P의 Probe AND 수치는 Probe 양성 P만 고른 설계에 따른 값이다.

## STEP 5-2b (N 위치 격자, confirmatory)

| 가설 | 추정 [97.5% N 이미지 CI] | 97.5% 클래스 CI | 판정 |
| --- | --- | --- | --- |
| H1b: N-last advantage (가운데 평균 − 끝) | +0.142 [+0.125, +0.160] | [+0.105, +0.187] | **robust support** |
| H2b: Middle-position effect (가운데 평균 − 첫 위치) | +0.040 [+0.025, +0.057] | [+0.000, +0.074] | **inconclusive** (추정값이 최소 효과 0.05 미만) |

- 요약 문구: **N이 마지막 자리에 있을 때만 False-Yes가 뚜렷이 낮고, 나머지 자리는 비슷하다.**
  ("가운데가 가장 나쁘다"로 쓰지 않는다. H2b는 결론 유보다.)
- 부가: 부호 불일치 입력을 빼면 H2b는 +0.031 [+0.018, +0.047] (민감도, 주 판정은 그대로). 3장 PNP > NPP > PPN 순서는
  STEP 4와 같음(같은 이미지 풀에서의 후속 확인이며 독립 재현이 아님).
- 실행 간: 5-2와 같은 입력 5,760건의 답·margin·z_N이 5-2와 비트 단위로 같다(같은 GPU·드라이버·라이브러리).
- 탐색(판정과 무관): 같은 a에서는 z_N이 동일한데(prefix 성질) b가 늘면 False-Yes가 오른다. 이는 exploratory 관찰이며,
  어느 처리 단계(attention, 집계, 최종 결정)가 원인인지는 이 실험으로 정하지 않는다. 장수도 함께 바뀌어 분리되지 않는다.

## STEP 5-3 인과 실험 A·B (사전 등록 고정, GPU 실행 전)

`step5_3/`: 고정 manifest 사본 4개(A `patching/frozen/v1`·`v2`, B `attention/frozen/v1`·`v2`), 계획 검사(`plan_check.json`),
freeze 전 표본 수·예상 CI 반폭(`prefreeze_report.json`), 분할(`split.json`, seed 53).
- 고정 시각: v1은 A 13:13:03Z, B 13:13:04Z. v2는 A 13:24:31Z, B 13:24:32Z (모두 2026-11-13).
  모두 5-3 모델 출력이 생기기 전이다. **평가는 v2를 쓴다.**
- v2 수정 (사용자 승인, v1 보존): 답 일치 기준을 bf16 동점 근처(|m| ≤ 0.125)가 아닌 쌍에 적용하고, 전체 일치율은
  참고값으로 낸다. margin 기준은 전체 쌍에 그대로 적용한다. 새 계산 경로에서 무개입 답이 바뀐 입력은 분석에서 빼고
  개수를 보고한다. 이유와 변경 항목은 v2 manifest의 `revision`에 있다.
- 계획 파일 원본(`data/step5_causal/`, 34MB)은 git 밖에 있다. 해시는 manifest와 `plan_check.json`에 기록돼 있다.
- 계획서: `docs/plans/step5_3_causal_ab.md`

## 대용량 산출물 (git 밖)

| 내용 | 위치 |
| --- | --- |
| STEP 1~5-2 서버 백업 (5-2 본 추출·fp32 추출 포함) | `cache/CAUproj2_backup.tar` (+ `.sha256`) |
| STEP 5-2b 파일럿·본 추출 묶음 | `cache/step5_position/pilot.tar`, `cache/step5_position/extraction.tar` |
| 로컬에서 만든 나머지 산출물 (5-2b 케이스·고정·평가, 5-2 평가·탐색) | `cache/backups/2026-11-13_step5_local_outputs.tar` (+ `.sha256`) |
