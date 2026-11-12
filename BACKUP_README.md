# CAUproj2 백업 표지

이 묶음은 git에 올라가지 않는 데이터와 실험 결과(`data/`, `results/`, `reports/`)를 서버에서 그대로 떠 온 것이다.
코드는 GitHub에 있으므로, **코드를 먼저 받고 그 프로젝트 폴더 안에 이 묶음을 풀면** 서버를 끄기 직전 상태로 돌아간다.

## 1. 어디서 왔나

| 항목 | 값 |
|---|---|
| 원래 위치 | Vast.ai GPU 서버 `/workspace/CAUproj2` (볼륨 아님, 서버 삭제 시 사라짐) |
| 묶은 시각 | {{CREATED_AT}} |
| 코드 버전 | `https://github.com/qotjwns/CAUproj2.git` 커밋 `ed35a3656a4fffe94ac4f72016260ecdb0b17da0` (`main`) |
| 하드웨어 | NVIDIA GeForce RTX 4090 (48GB), 드라이버 615.71.09 |
| 환경 | Python 3.12.15, torch 2.14.1+cu132 (CUDA 13.2), transformers 5.22.2 — 전체 목록 `results/environment_pip_freeze.txt` |
| 모델 | Qwen/Qwen2.5-VL-3B-Instruct, bf16 (STEP 5-2 민감도 분석만 비전 인코더 fp32) |
| 묶음 크기 / 파일 수 | {{SIZE}} / {{FILE_COUNT}}개 |

## 2. 로컬에 받은 뒤 정리하는 법

서버에서 받을 파일은 두 개다: `CAUproj2_backup.tar`(이 묶음)와 `CAUproj2_backup.tar.sha256`(묶음 전체의 체크섬, tar 밖에 있음).

**보관 위치 (권장)** — 풀지 말고 tar 그대로, 날짜·서버별 폴더에 둔다.

```
<내 문서 또는 홈>/CAUproj2_backups/
└── 2026-11-12_vast_rtx4090_step5-2/
    ├── CAUproj2_backup.tar
    ├── CAUproj2_backup.tar.sha256
    └── BACKUP_README.md          ← tar에서 이 파일만 꺼내 둔 사본 (선택)
```

- 받은 직후 체크섬 확인: 같은 폴더에서
  - macOS/Linux: `shasum -a 256 -c CAUproj2_backup.tar.sha256` (Linux는 `sha256sum -c`)
  - Windows PowerShell: `Get-FileHash CAUproj2_backup.tar -Algorithm SHA256` 결과를 `.sha256` 파일의 값과 비교
- 표지만 꺼내기: `tar -xf CAUproj2_backup.tar BACKUP_README.md`
- 이 백업 폴더는 git 저장소 **밖**에 둔다 (4GB, 사전 고정 기록이 섞이지 않게).

**다시 쓸 때** — 쓸 곳(로컬 또는 새 서버)의 프로젝트 루트에 푼다.

- 로컬에서 CPU 분석(STEP 5-2 평가·시각화 등)을 이어갈 때: 로컬에 `git clone`한 `CAUproj2/` 루트에서
  `tar -xf <백업 폴더>/CAUproj2_backup.tar`. `data/`, `results/`, `reports/`는 `.gitignore` 대상이라 git에 섞이지 않는다.
- 새 GPU 서버로 옮길 때: tar를 서버로 올린 뒤(VS Code 업로드 또는 `scp -P <포트> CAUproj2_backup.tar root@<IP>:/workspace/`)
  서버의 프로젝트 루트에서 같은 방법으로 푼다. 5절 복원 순서를 따른다.

## 3. 묶음 안 파일이 들어갈 자리

묶음 안의 경로는 **프로젝트 루트 기준 상대 경로**다. 프로젝트 루트(= `git clone`한 `CAUproj2/` 폴더)에서 풀면 제자리에 들어간다.

| 묶음 안 경로 | 복원 위치 (프로젝트 루트 기준) | 내용 |
|---|---|---|
| `BACKUP_README.md` | `./BACKUP_README.md` | 이 표지 (git에는 untracked로 보임, 지워도 됨) |
| `data/balanced3/` | `./data/balanced3/` | STEP 1~3 클래스 풀·케이스 (seed 42) |
| `data/step4_independent/` | `./data/step4_independent/` | STEP 4 독립 검증 케이스 (seed 2026) |
| `data/step5_accumulation/` | `./data/step5_accumulation/` | STEP 5-2 케이스(seed 52), 설계, fp32 부분집합, STEP 1~5 사용 이미지 목록 |
| `results/step2/` | `./results/step2/` | STEP 2 Hidden State 추출 (고정 Probe 학습 데이터) |
| `results/step3/` | `./results/step3/` | STEP 3 Linear Probe OOF 결과 |
| `results/step4/` | `./results/step4/` | STEP 4 **고정 Probe (`frozen/v1`)**, 파일럿, 본 추출, 평가 |
| `results/step5/` | `./results/step5/` | STEP 5-2 **사전 등록 `frozen/v1`·`v2`**, 파일럿, 본 추출, fp32 비전 민감도 추출 |
| `results/step5_diagnostics/` | `./results/step5_diagnostics/` | STEP 5-1 분석(evidence_margin, evidence_followup), h_q 진단, 문맥 잡음 |
| `results/environment_pip_freeze.txt` | `./results/environment_pip_freeze.txt` | 패키지 버전 전체 |
| `reports/` | `./reports/` | 그림 (STEP 4, STEP 5 진단) |

이 서버에는 `results/step1/`과 STEP 1~3의 `reports/`가 없었으므로 묶음에도 없다.

## 4. 묶음에 넣지 않은 것 (다시 받기)

| 경로 | 크기 | 다시 받는 법 |
|---|---|---|
| `cache/models/Qwen/Qwen2.5-VL-3B-Instruct/` | 7.1GB | `python -m scripts.download_model` |
| `cache/downloads/`, `data/raw/` (COCO val2017 이미지, COCO-REM 주석) | 1.6GB | `python -m scripts.download_data` |

## 5. 복원 순서

```bash
git clone https://github.com/qotjwns/CAUproj2.git
cd CAUproj2
git checkout ed35a3656a4fffe94ac4f72016260ecdb0b17da0     # 또는 그 이후 main
tar -xf /경로/CAUproj2_backup.tar                         # 프로젝트 루트에서 풀기
pip install -r results/environment_pip_freeze.txt         # 안 되면 최소한 torch 2.14.1, transformers 5.22.2는 맞춤
python -m scripts.download_model
python -m scripts.download_data
sha256sum -c <<'EOF'                                      # 고정 기록 확인
fd3f19205218420d5acd0b2325a986e2d3333b42a221f5040020503d1599c6be  results/step4/qwen2_5_vl_3b/frozen/v1/manifest.json
da8273aa9cf6b158da963221e4c319d810f130a1e6052e8ee1453131b32e01a1  results/step5/qwen2_5_vl_3b/accumulation/frozen/v1/manifest.json
8b76d02ca609672bbcba0fe337c020dc6f716c5ae993dd288111ac39affae54b  results/step5/qwen2_5_vl_3b/accumulation/frozen/v2/manifest.json
EOF
```

컨테이너 CPU 할당량이 작은 서버에서는 CPU 스크립트가 스레드를 자동으로 줄이지만, GPU 추출은
`OMP_NUM_THREADS=16`처럼 직접 지정하는 것이 빠르다.

## 6. 손대면 안 되는 것

- `results/step4/qwen2_5_vl_3b/frozen/v1/`, `results/step5/qwen2_5_vl_3b/accumulation/frozen/v1/`, `v2/`:
  사전 고정 기록. 수정하지 말고, 바꿔야 하면 새 버전(v3)을 이유와 함께 만든다.
- `results/step5/qwen2_5_vl_3b/accumulation/extraction/`: 사전 등록한 본 추출. 다시 돌리면 라이브러리·GPU가 달라
  bf16 수치가 바뀔 수 있으므로, 평가는 이 결과로 한다.
- 다른 버전의 torch/transformers에서 추출을 다시 하면 STEP 1~5 결과와 수치가 달라질 수 있다.

## 7. 진행 상태 (묶은 시점)

| 단계 | 상태 |
|---|---|
| STEP 1~4 | 완료 |
| STEP 5-1 (CPU, exploratory) | 주 분석과 보완 분석 A1~A7 완료 (`results/step5_diagnostics/.../evidence_margin`, `evidence_followup`) |
| STEP 5-2 사전 등록 | `frozen/v2` (v1은 보존. v2는 본 추출 전 N-first 무결성 기준만 코사인 거리 ≤ 0.001로 변경) |
| STEP 5-2 파일럿 (64건) | v2 기준 통과. 무결성 확인용이며 분석에 쓰지 않음 (STEP 4와 같은 방식) |
| STEP 5-2 본 추출 (15,360건) | {{MAIN_STATUS}} |
| STEP 5-2 fp32 비전 민감도 추출 (1,920건) | {{FP32_STATUS}} |
| STEP 5-2 평가·시각화 | **아직 안 함** — 다음 할 일 |

## 8. 다음 할 일

1. STEP 5-2 평가 분석 (CPU, `configs/experiments/step5_accumulation.yaml`의 사전 등록 v2 그대로):
   P4 − P1(False-Yes, fp32 변형 margin) 배치별 판정과 견고성 표기, H2(High − Low), All-P 대조군, 차이의 차이,
   민감도(bf16 margin, 부호 불일치 제외, 무결성 기준 초과 제외, fp32 비전), 탐색(클래스별, N-first z_N 변화).
2. 시각화 (각주에 status와 표본 정의).
3. 결과 보고 후 STEP 5-3(집계 교정) 설계.
