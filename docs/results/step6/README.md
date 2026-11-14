# STEP 6 결과 보존본

- `frozen_v1_manifest.json`: STEP 6-2 사전 등록 고정본 v1 사본
  - 원본: `results/step6/qwen2_5_vl_3b/contribution/frozen/v1/manifest.json`
  - 고정 시각: 2026-11-14T06:33:55Z. STEP 6-2 GPU 출력(실제 3B 검증, basis, 연습 실행, 본 실행) 전에 고정했다.
- `frozen_v2_manifest.json`: 고정본 v2 사본
  - 원본: `results/step6/qwen2_5_vl_3b/contribution/frozen/v2/manifest.json`
  - 고정 시각: 2026-11-14, STEP 6-2 GPU 출력 전. v1과 다른 파일은 설정과 `selftest_run.py`뿐이다.
  - 사유: 서버 CPU 자체 시험의 검사 하나(batch 크기가 다른 실행 비교)가 반올림 차이로 실패했다.
    같은 크기 batch끼리 비트 단위로 비교하도록 바꿨다. 사전 등록·설계·기준은 그대로다.
  - 기록: 설계 문서 §13.8, v3 설정의 `revision.history`
- `frozen_v3_manifest.json`: 고정본 v3 사본 (실행·평가에 쓰는 버전)
  - 원본: `results/step6/qwen2_5_vl_3b/contribution/frozen/v3/manifest.json`
  - 고정 시점: 2026-11-14, 실제 3B 검증이 실패한 뒤이고 basis·연습·본 실행 출력 전이다.
  - 변경: 재구성 판정. 항등식은 원소별 그대로이고, attention·output은 층별 상대 오차(fp32 ≤ 1e-5, bf16 ≤ 2⁻⁶)로 판정한다.
  - 기록: 설정의 `revision`, 설계 문서 §13.9
- `real_3b_fail_v2.json`, `step6_gpu_fail_v2.log`: 고정본 v2로 한 실제 3B 검증의 결과와 서버 로그 사본
  - 결과: fp32 5장 입력 L22 output 검사 1건이 원소별 한계를 1.2e-7 넘어 실패했다.
  - 원본: `results/step6/qwen2_5_vl_3b/contribution_validation/real_3b_fail_v2.json`, `results/step6/qwen2_5_vl_3b/step6_gpu_fail_v2.log`
- `plan_check.json`: STEP 6-2 계획 검사 사본 (`data/step6_contribution/plan_check.json`)
  - STEP 3 풀과 평가 이미지의 겹침 0 (COCO id, 파일 SHA-256)
  - 평균 차이 방향을 만들 N·P 기여의 충분성
  - 항목 수
- 설계·결정: `docs/plans/step6_attention_contribution.md` §13
- 사전 등록: `configs/experiments/step6_contribution.yaml`
- 검증: `shasum -a 256 -c SHA256SUMS` (이 폴더에서)
