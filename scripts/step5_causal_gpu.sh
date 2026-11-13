#!/usr/bin/env bash
# STEP 5-3 실험 A·B GPU 실행 (서버의 프로젝트 루트에서: bash scripts/step5_causal_gpu.sh)
#
# 순서: 환경·자체 시험 → 연습 실행 A → 검사 → 연습 실행 B → 검사 → 시간 추정(8시간 한도) → 본 실행 A → 검사
#       → 본 실행 B → 검사 → 결과 백업 tar.
# 중단 조건(사용자 지정): 빠른 경로 동등성 실패, 무결성 기준 미달, 실측 기준 예상 시간 8시간 초과.
# 각 검사는 실패하면 종료 코드 1을 돌려주고, set -e 때문에 스크립트가 그 자리에서 멈춘다.
# 다시 실행하면 검사를 통과한 단계는 건너뛰고, 끝나지 않은 실행은 --resume으로 이어 간다.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTHONUNBUFFERED=1
ROOT=results/step5/qwen2_5_vl_3b
LOG="$ROOT/causal_gpu.log"
mkdir -p "$ROOT" cache/step5_causal
stage() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

passed() {   # 검사 파일이 있고 status가 passed이면 0
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
sys.exit(0 if path.exists() and json.loads(path.read_text())["status"] == "passed" else 1)
PY
}

run_stage() {   # $1 실험(A|B) $2 단계(pilot|main) $3 출력 폴더
  local experiment=$1 step=$2 dir=$3 extra=""
  if passed "$dir/${step}_check.json"; then
    stage "건너뜀: $experiment $step (검사 통과 기록 있음)"
    return
  fi
  if [ -f "$dir/run.json" ]; then extra="--resume"; fi
  stage "실행: $experiment $step $extra"
  python -m experiments.step5_causal.run --experiment "$experiment" --stage "$step" $extra 2>&1 | tee -a "$LOG"
  stage "검사: $experiment $step"
  python -m experiments.step5_causal.check --experiment "$experiment" --stage "$step" 2>&1 | tee -a "$LOG"
}

stage "0. 환경 확인과 자체 시험 (작은 무작위 모델, 개입 경로가 이 라이브러리에서 맞는지)"
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch, transformers
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | transformers {transformers.__version__} | "
      f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
assert torch.cuda.is_available(), "GPU가 없습니다"
PY
python -m experiments.step5_causal.selftest --model 2>&1 | tee -a "$LOG"

stage "1. 연습 실행 (무결성·동등성·시간 측정 전용, 분석에 쓰지 않음)"
run_stage A pilot "$ROOT/patching/pilot"
run_stage B pilot "$ROOT/attention/pilot"

stage "2. 본 실행 예상 시간 (연습 실행 실측, 한도 8시간)"
python -m experiments.step5_causal.check --estimate 2>&1 | tee -a "$LOG"

stage "3. 본 실행"
run_stage A main "$ROOT/patching/main"
run_stage B main "$ROOT/attention/main"

stage "4. 결과 백업"
BACKUP=cache/step5_causal/step5_3_results.tar
tar -cf "$BACKUP" "$ROOT/patching" "$ROOT/attention" "$ROOT/causal_time_estimate.json" "$LOG"
sha256sum "$BACKUP" > "$BACKUP.sha256"
ls -lh "$BACKUP"
cat "$BACKUP.sha256"
stage "완료: $BACKUP 과 $BACKUP.sha256 을 로컬로 내려받으세요 (평가는 로컬 CPU에서 한 번만)"
