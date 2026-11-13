#!/usr/bin/env bash
# STEP 5-3 GPU 서버 업로드 묶음 만들기 (로컬, freeze·커밋 뒤에 프로젝트 루트에서: bash scripts/step5_causal_bundle.sh)
# 1) 코드: 현재 커밋의 git archive (push 없이 서버에서 같은 코드를 쓰기 위해)
# 2) 데이터: 계획 파일(data/step5_causal), 고정 manifest(A·B frozen/v1·v2), 쓰는 COCO 이미지 762장
set -euo pipefail
test -z "$(git status --porcelain -- src experiments scripts configs)" || { echo "커밋하지 않은 코드 변경이 있습니다"; exit 1; }
OUT=cache/step5_causal/upload
CODE="$OUT/step5_causal_code.tar"
DATA="$OUT/step5_causal_upload.tar"
mkdir -p "$OUT"
for v in v1 v2; do
  test -f "results/step5/qwen2_5_vl_3b/patching/frozen/$v/manifest.json"
  test -f "results/step5/qwen2_5_vl_3b/attention/frozen/$v/manifest.json"
done
COMMIT=$(git rev-parse HEAD)
git archive --format=tar -o "$CODE" "$COMMIT"
python - > "$OUT/files.txt" <<'PY'
import json
names = set()
for line in open("data/step5_causal/inputs.jsonl", encoding="utf-8"):
    case = json.loads(line)["case"]
    names.update(f"{case['image_dir']}/{name}" for name in case["image_order"])
print("data/step5_causal")
for name in ("patching", "attention"):
    for version in ("v1", "v2"):          # v1은 기록 보존용, 실행·평가는 v2
        print(f"results/step5/qwen2_5_vl_3b/{name}/frozen/{version}")
print("\n".join(sorted(names)))
PY
tar -cf "$DATA" -T "$OUT/files.txt"
(cd "$OUT" && shasum -a 256 "$(basename "$CODE")" "$(basename "$DATA")" > SHA256SUMS)
echo "$COMMIT" > "$OUT/COMMIT"
echo "커밋 $COMMIT"
ls -lh "$CODE" "$DATA"
cat "$OUT/SHA256SUMS"
