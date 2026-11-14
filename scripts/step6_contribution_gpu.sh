#!/usr/bin/env bash
# STEP 6-2 GPU 실행 (서버의 프로젝트 루트에서: bash scripts/step6_contribution_gpu.sh)
#
# 순서: 고정 확인·환경·자체 시험 → 0. 실제 3B 검증(재구성·개입 위치, fp32·bf16) → 1. basis(STEP 3 풀 → P 기저·평균 차이)
#       → 검사 → 2. 연습 실행(선택 절반) → 검사 → 3. 시간 추정(8시간 한도) → 4. 본 실행(확증 절반) → 검사 → 5. 결과 백업 tar.
# 중단 조건(사용자 지정): 실제 3B 검증 실패, 무결성 기준 미달, 실측 기준 예상 시간 8시간 초과.
# 각 검사는 실패하면 종료 코드 1을 돌려주고, set -e 때문에 스크립트가 그 자리에서 멈춘다.
# 다시 실행하면 통과 기록이 있는 단계는 건너뛰고, 끝나지 않은 실행은 --resume으로 이어 간다 (basis는 처음부터).
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTHONUNBUFFERED=1
ROOT=results/step6/qwen2_5_vl_3b
DIR="$ROOT/contribution"
LOG="$ROOT/step6_gpu.log"
mkdir -p "$DIR" cache/step6
stage() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

passed() {   # $1 JSON 파일, $2 키(status|passed): status == "passed" 또는 passed == true이면 0
  python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
path, key = Path(sys.argv[1]), sys.argv[2]
if not path.exists():
    sys.exit(1)
value = json.loads(path.read_text()).get(key)
sys.exit(0 if value in ("passed", True) else 1)
PY
}

run_stage() {   # $1 단계(basis|pilot|main)
  local step=$1 dir="$DIR/$1" extra=""
  if passed "$dir/${step}_check.json" status; then
    stage "건너뜀: $step (검사 통과 기록 있음)"
    return
  fi
  if [ -f "$dir/run.json" ]; then extra="--resume"; fi
  stage "실행: $step $extra"
  python -m experiments.step6_contribution.run --stage "$step" $extra 2>&1 | tee -a "$LOG"
  stage "검사: $step"
  python -m experiments.step6_contribution.check --stage "$step" 2>&1 | tee -a "$LOG"
}

stage "고정 확인 (서버 파일 해시), 환경, 자체 시험 (작은 무작위 모델)"
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch, transformers
from experiments.step6_contribution.freeze import load_config, verify_freeze
manifest = verify_freeze(load_config())
print(f"고정 {manifest['version']} 확인 | torch {torch.__version__} (CUDA {torch.version.cuda}) | "
      f"transformers {transformers.__version__} | GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
assert torch.cuda.is_available(), "GPU가 없습니다"
PY
python -m experiments.step6_contribution.selftest 2>&1 | tee -a "$LOG"
python -m experiments.step6_contribution.selftest_run --model 2>&1 | tee -a "$LOG"

stage "0. 실제 3B 검증 (재구성·개입 위치, fp32·bf16, 입력 3개). 실패하면 여기서 멈춘다"
if passed "$ROOT/contribution_validation/real_3b.json" passed; then
  stage "건너뜀: 실제 3B 검증 (통과 기록 있음)"
else
  python -m experiments.step6_contribution.validate_real 2>&1 | tee -a "$LOG"
fi

stage "1. basis: STEP 3 풀 기여 → P 기저(v90·v95·r64)·평균 차이"
run_stage basis

stage "2. 연습 실행 (선택 절반, 무결성·동등성·시간 측정 전용, 분석에 쓰지 않음)"
run_stage pilot

stage "3. 본 실행 예상 시간 (연습 실행 실측, 한도 8시간)"
python -m experiments.step6_contribution.check --estimate 2>&1 | tee -a "$LOG"

stage "4. 본 실행 (확증 절반: False-Yes one-N 2,501 + All-P 1,985)"
run_stage main

stage "5. 결과 백업"
BACKUP=cache/step6/step6_2_results.tar
tar -cf "$BACKUP" "$ROOT/contribution_validation" "$DIR/basis" "$DIR/pilot" "$DIR/main" "$DIR/time_estimate.json" "$LOG"
sha256sum "$BACKUP" > "$BACKUP.sha256"
ls -lh "$BACKUP"
cat "$BACKUP.sha256"
stage "완료: $BACKUP 과 $BACKUP.sha256 을 로컬로 내려받으세요 (평가는 로컬 CPU에서 한 번만)"
