#!/usr/bin/env bash
# STEP 6-2 GPU 서버 업로드 묶음 만들기 (로컬, freeze·커밋 뒤에 프로젝트 루트에서: bash scripts/step6_contribution_bundle.sh)
# 1) 코드: 현재 커밋의 git archive
# 2) 데이터: STEP 5-3 계획(data/step5_causal), STEP 6 계획(data/step6_contribution), STEP 3 풀 케이스
#    (data/balanced3/inference_cases), 고정 manifest(frozen/v1·v2), 쓰는 COCO 이미지 (5-3 입력 762장 + STEP 3 풀 310장)
set -euo pipefail
test -z "$(git status --porcelain -- src experiments scripts configs)" || { echo "커밋하지 않은 코드 변경이 있습니다"; exit 1; }
OUT=cache/step6/upload
CODE="$OUT/step6_code.tar"
DATA="$OUT/step6_upload.tar"
FROZEN=results/step6/qwen2_5_vl_3b/contribution/frozen      # 모든 버전 (v1 기록 보존, 실행은 설정의 frozen_dir)
mkdir -p "$OUT"
test -f "$FROZEN/v2/manifest.json"
COMMIT=$(git rev-parse HEAD)
git archive --format=tar -o "$CODE" "$COMMIT"
.venv/bin/python - "$FROZEN" > "$OUT/files.txt" <<'PY'
import json, sys
names = set()
for path in ("data/step5_causal/inputs.jsonl", "data/step6_contribution/pool_inputs.jsonl"):
    for line in open(path, encoding="utf-8"):
        case = json.loads(line)["case"]
        names.update(f"{case['image_dir']}/{name}" for name in case["image_order"])
print("data/step5_causal")
print("data/step6_contribution")
print("data/balanced3/inference_cases")
print(sys.argv[1])
print("\n".join(sorted(names)))
PY
tar -cf "$DATA" -T "$OUT/files.txt"
(cd "$OUT" && shasum -a 256 "$(basename "$CODE")" "$(basename "$DATA")" > SHA256SUMS)
echo "$COMMIT" > "$OUT/COMMIT"
echo "커밋 $COMMIT"
echo "이미지 $(grep -c '\.jpg$' "$OUT/files.txt")장"
ls -lh "$CODE" "$DATA"
cat "$OUT/SHA256SUMS"
