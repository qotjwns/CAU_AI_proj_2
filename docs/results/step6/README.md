# STEP 6 결과 보존본

- `frozen_v1_manifest.json`: STEP 6-2 사전 등록 고정본 v1 사본
  - 원본: `results/step6/qwen2_5_vl_3b/contribution/frozen/v1/manifest.json`
  - 고정 시각: 2026-11-14T06:33:55Z. STEP 6-2 GPU 출력(실제 3B 검증, basis, 연습 실행, 본 실행) 전에 고정했다.
- `plan_check.json`: STEP 6-2 계획 검사 사본 (`data/step6_contribution/plan_check.json`)
  - STEP 3 풀과 평가 이미지의 겹침 0 (COCO id, 파일 SHA-256)
  - 평균 차이 방향을 만들 N·P 기여의 충분성
  - 항목 수
- 설계·결정: `docs/plans/step6_attention_contribution.md` §13
- 사전 등록: `configs/experiments/step6_contribution.yaml`
- 검증: `shasum -a 256 -c SHA256SUMS` (이 폴더에서)
