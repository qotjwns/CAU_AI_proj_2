# STEP 6 이미지별 Attention 기여 추출·개입 설계 (초안, STEP 6-1)

- 상태: **6-1 초안** (2026-11-14). 범위는 구현 가능성 점검, 수치 정확성 검증, 설계까지다.
  - 대규모 GPU 실행, 성능 평가, 새 인과 실험, 사전 등록, freeze는 하지 않았다.
  - 다음 실험의 사전 등록, freeze, GPU 본 실행은 별도 승인을 받은 뒤에 한다.
- 코드: `experiments/step6_contribution/`
  - `contribution.py`: 추출·재구성·개입
  - `directions.py`: 다섯 조건과 P 기저 (정의만)
  - `validation.py`: 검증 함수와 허용 오차
  - `selftest.py`: CPU 자체 시험
  - `validate_real.py`: 실제 3B 소규모 검증
- 기존 STEP 1~5-3 고정본, 평가 결과, 사전 등록 파일은 바꾸지 않았다 (§10).

## 1. 기존 코드 점검 결과

| 항목 | 확인 내용 | 근거 |
| --- | --- | --- |
| 층 구조 | `input_layernorm → self_attn → residual 덧셈 → post_attention_layernorm → mlp → residual 덧셈` | `Qwen2_5_VLDecoderLayer.forward` |
| L21–24 | 5-3B 구간 (21, 24)는 0부터 센 디코더 층 20–23이다 (`layers[b, low−1:high]`) | `experiments/step5_causal/run.py` `b_attention` |
| attention 구현 | q/k/v 투영(bias 있음) → mrope(`apply_rotary_pos_emb`) → attention 함수(기본 sdpa) → head 결합 → `o_proj` | `Qwen2_5_VLAttention.forward` |
| head 결합 | `[1, H, T, d] → transpose → [T, H·d]`, head 순서대로 이어 붙인다. GQA는 `repeat_kv`라서 query head h가 KV head ⌊h/8⌋을 쓴다 (3B: 16 head, KV 2, d = 128) | 같은 파일 `repeat_kv` |
| `o_proj` bias | 없음(`bias=False`). attention 모듈은 `o_proj` 출력을 그대로 돌려주고, residual 덧셈은 디코더 층에서 한다 | 같은 파일 |
| attention 가중치 | sdpa는 가중치를 돌려주지 않는다. 그래서 층 입력(hidden, mask, cos/sin)을 기록해 A를 다시 계산한다. eager 구현은 가중치를 돌려줘서 교차 확인에 썼다 | §4 |
| value | `v_proj(hidden)`에서 직접 얻는다 (KV cache 없이 전체 prefill) | `contribution.recompute_attention` |
| 이미지 토큰 경계 | `map_image_tokens`: 이미지마다 image-pad 토큰 `[start, end)`. `vision_start`는 start − 1, `vision_end`는 end 위치다 | `src/core/features.py` |
| 질문 구간 (5-3B) | `suffix_start = spans[-1].end + 1`(마지막 vision_end 다음)부터 마지막 입력 토큰까지. 26~28개이고 5-3B의 `post_image` query와 같다 | `run.py`, 계획 `b_items.suffix_start` |
| mask | 전체 prefill에서 sdpa는 mask 없이 인과로 돈다. 5-3B 빠른 경로는 명시적 4D 가산 mask를 쓴다. 재계산은 없음·bool·가산 mask를 모두 처리한다 | `contribution.recompute_attention` |

기존 코드 재사용 여부
- **그대로 쓴다:**
  - 입력 준비(`prepare_inputs`), 이미지 경계(`map_image_tokens`)
  - 계획 파일: `inputs.jsonl`(기준 경계·답), `b_items.jsonl`(`target_index`, `control_index`, `suffix_start`, 절반)
  - 층 번호 규칙, 분할(seed 53)
- **그대로는 못 쓴다:**
  - 5-3B의 `Hooks`/`Runner`는 attention mask를 바꾸는 softmax 개입이고, 빠른 경로(prefix cache + suffix batch)를 쓴다.
  - 기여 추출은 각 key 묶음의 A·v가 필요한데, 6-1은 KV cache 없는 전체 prefill(batch 1)만 지원한다.
  - 빠른 경로로 넓히려면 prefix cache의 k/v(rope 적용 후)를 읽어 같은 재계산을 해야 한다 (후속 작업, §9).

## 2. 기여 정의

층 ℓ, 질문 토큰 t, key 묶음 g:

$$z_g[t]=\operatorname{concat}_h\sum_{j\in g}A_{h,t,j}\,v_{h,j},\qquad c_g[t]=W_O\,z_g[t],\qquad o[t]=\sum_g c_g[t]+b_O$$

- **묶음:** 이미지 i마다 image-pad 토큰 `[start, end)` 하나씩(c_i), 나머지 key 전부를 비이미지 하나로(c_nonimage) 둔다.
  - 비이미지에는 시스템·질문 텍스트, vision_start/end 표시 토큰, 질문 토큰 자신이 들어간다.
  - 묶음들은 key 위치 [0, T)를 겹침 없이 모두 나눈다 (`key_groups`에서 검사).
- **공간:** o_proj 이후 공간(hidden 2048)이다. 3B는 b_O가 없다. bias가 있는 모델이면 전체 재구성에서 한 번만 더한다.
  - 묶음마다 더하면 (묶음 수 − 1)·b_O만큼 틀린다. 가짜 attention 시험에서 정확히 그만큼 틀리는 것을 확인했다.
- **A 재계산:** 모델과 같은 입력을 쓴다 — 층의 attention 입력, 모델 dtype으로 한 q/k/v 투영, 같은 mrope cos/sin, 같은 mask, 같은 scaling.
  - softmax와 가중합은 float32로 한다.
  - c_N은 N 이미지 묶음의 c_i이고, c_P는 계획의 P 대조 이미지(`control_index`) 묶음의 c_i다.

## 3. 구현 위치

| 기능 | 위치 |
| --- | --- |
| 층 입력·`o_proj` 입력·출력 기록 (batch 1, `use_cache=False`) | `contribution.AttentionRecorder` (`self_attn` forward pre-hook + `o_proj` forward hook) |
| A·v 재계산, 묶음별 c_g, 분해하지 않은 참조 | `contribution.recompute_attention`, `contributions` |
| 재구성 o = Σ c_g + b_O | `contribution.reconstruct` |
| 개입 o' = o + λv (o_proj 뒤, residual 덧셈 앞) | `contribution.OutputAddition` (`self_attn` forward hook, 출력 tuple의 첫 원소를 바꾼다) |
| 개입 전후 비교 | `contribution.OutputProbe` (개입 hook 앞에 prepend, 뒤에 append) |
| 다섯 조건, P 기저 | `directions.py` |

## 4. 재구성 검증

허용 오차는 실행 전에 `validation.TOLERANCES`에 고정했다. 기준은 `torch.allclose`: |a − b| ≤ atol + rtol·|b|.
- **항등식:** Σ c_g + b_O 대 분해하지 않은 참조. 같은 A·v·W_O로 float32 계산하고, rtol = atol = 1e-5다.
- **attention:** 다시 계산한 o_proj 입력 대 모델 커널이 o_proj에 넣은 값이다.
  - float32 모델: rtol 1e-4, atol 1e-5
  - bfloat16 모델: rtol = atol = 2⁻⁶
- **output:** Σ c_g + b_O 대 모델 attention 출력이다. 허용 오차는 attention과 같다.
- **o_proj:** W_O·(모델 z) 대 모델 o. o_proj 자체의 반올림 크기를 따로 보는 참고 항목이다.

환경: CPU, eval 모드(dropout 0), batch 1, 무개입 forward, `use_cache=False`.
torch 2.15.0, transformers 5.22.2, attention 구현은 sdpa이고 교차 확인에는 eager를 썼다.

### 4.1 가짜 attention (o_proj bias 있음, GQA 6/2, mrope 함수, mask 세 종류, 계산은 독립 커널 sdpa)

| dtype | mask | 출력 max abs | 상대 오차 | allclose | bias를 묶음마다 더한 경우 |
| --- | --- | --- | --- | --- | --- |
| float64 | 없음(인과) / bool / 가산(+log 4 포함) | 1.8e-14 / 2.1e-14 / 1.8e-14 | 4e-16 | 통과 | max 4.000 = (4 − 1)·max\|b\| (예상과 일치) |
| float32 | 같음 | 1.3e-5 / 1.1e-5 / 7.6e-6 | 1.6e-7 ~ 1.9e-7 | 통과 | 같음 |

### 4.2 작은 무작위 Qwen2.5-VL (같은 구조·코드 경로, 4층, hidden 64, head 4 / KV 2, 패치·병합 크기는 실제와 같음)

전체 4층, 질문 토큰 기준이다. "max / mean / RMSE / 상대"는 Σ c + b 대 모델 attention 출력이다.

| 입력 | dtype | 출력 max / mean / RMSE | 상대 | attention 재계산 max | 항등식 max | o_proj 반올림 max | allclose |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 합성 이미지 3장 | float32 sdpa | 2.2e-6 / 2.7e-7 / 3.7e-7 | 2.5e-7 | 1.7e-6 | 1.9e-6 | 0 | 통과 |
| 합성 이미지 3장 | float32 eager | 2.2e-6 | 2.2e-7 | 0 (A가 eager 가중치와 비트 동일) | 2.2e-6 | 0 | 통과 |
| 합성 이미지 3장 | bfloat16 sdpa | 2.2e-2 / 2.7e-3 / 3.6e-3 | 2.4e-3 | 1.5e-2 | 2.4e-6 | 1.6e-2 | 통과 |
| 실제 입력 3장·middle (T 1127) | float32 | 4.1e-6 / 6.7e-7 / 8.7e-7 | 6.0e-7 | 5.7e-6 | 4.2e-6 | 0 | 통과 |
| 실제 입력 4장·first (T 1520) | float32 | 5.4e-6 / 6.0e-7 / 8.1e-7 | 5.6e-7 | 8.6e-6 | 6.0e-6 | 0 | 통과 |
| 실제 입력 5장·last (T 2144) | float32 | 1.6e-5 / 1.2e-6 / 1.6e-6 | 8.8e-7 | 1.3e-5 | 1.1e-5 | 0 | 통과 |
| 실제 입력 3장·middle | bfloat16 | 2.0e-2 / 2.6e-3 / 3.4e-3 | 2.4e-3 | 9.0e-3 | 3.8e-6 | 1.6e-2 | 통과 |
| 실제 입력 4장·first | bfloat16 | 1.8e-2 / 2.6e-3 / 3.4e-3 | 2.3e-3 | 8.6e-3 | 6.2e-6 | 1.5e-2 | 통과 |
| 실제 입력 5장·last | bfloat16 | 2.7e-2 / 3.2e-3 / 4.2e-3 | 2.3e-3 | 1.5e-2 | 1.1e-5 | 1.6e-2 | 통과 |

- **실제 입력:** 실제 processor와 COCO 이미지로 만든 3·4·5장 입력이다 (`validation.real_cases` 규칙 고정, 5-3B 선택 절반).
  - 이미지 토큰 경계와 질문 구간 시작이 저장된 기준 실행·계획과 모두 일치했다.
  - 가중치는 무작위이므로 attention 패턴은 실제 모델과 다르다. 이 시험은 구현 정확성만 확인한다.
- **층별·토큰별 최대 오차:**
  - float32: 층별 최대 2.9e-6 ~ 1.6e-5, 질문 토큰별 최대 1.6e-5 이하
  - bfloat16: 층별 최대 1.3e-2 ~ 2.7e-2
  - 전체 값은 `results/step6/contribution_validation/selftest.json`의 `per_layer`, `per_token_output_max_abs`에 있다.
- **확률 행 합:** |1 − Σ_j A| ≤ 1.8e-6이다.
- **bfloat16 오차의 출처:**
  - 분해 자체(항등식)는 float32 수준이다 (상대 ~6e-7).
  - 모델 출력과의 차이(상대 ~2.3e-3)는 모델의 bf16 반올림이다. 근거: o_proj만 다시 계산해도 같은 크기(상대 ~1.6e-3)가 나오고, attention 커널 출력 반올림(상대 ~1.7e-3)이 더해진다.
  - 허용 오차를 늘리지 않았다.

### 4.3 실제 Qwen2.5-VL-3B: **미검증**

- 이유: 이 컴퓨터에서는 GPU 단계와 모델 실행을 하지 않는다는 기존 규칙을 따랐다.
  - RAM이 18 GB라 float32 3B(가중치만 약 15 GB)는 메모리가 빠듯하다.
- 준비된 것: `python -m experiments.step6_contribution.validate_real` (float32·bfloat16, 위 입력 3개, L21–24, λ = 3)
  → `results/step6/qwen2_5_vl_3b/contribution_validation/real_3b.json`
  - GPU에서는 모델 로드를 빼고 수 분이면 된다.
  - 서버에 필요한 것: 코드, `data/step5_causal/` 계획 파일, 해당 COCO 이미지, 모델.
- 실행 전까지 실제 3B 재구성 오차와 개입 위치는 **보고하지 않는다**.

## 5. 개입 지점과 검증

- **지점:** 층 ℓ attention 모듈 출력(o_proj 뒤, residual 덧셈 앞)에서 질문 토큰 t에 o'[t] = o[t] + λ v[ℓ, t].
  - 덧셈은 float32로 하고 모델 dtype으로 되돌린다.
  - 길이가 전체 입력 길이인 forward(prefill)에서만 작동한다. 생성 단계(길이 1)는 직접 건드리지 않는다.
  - 다만 생성 토큰은 개입으로 바뀐 앞 토큰들의 KV를 본다.
- **v는 별도 무개입 실행에서 뽑는다.** 같은 입력의 개입 실행에서 그 저장값을 더한다.
  - 그래서 L22–24에서 더하는 c_N은 '그 실행에서 다시 계산한 c_N'이 아니라 무개입 실행의 값이다. 앞 층 개입 때문에 둘은 다르다.
  - 실행 중 다시 계산하는 방식은 별도 선택지로 남긴다.
- **검증 (작은 모델, 실제 입력 3개 × float32·bfloat16, 개입 층 1–2 / 전체 4층, λ = 0과 3): 6개 모두 통과**

| 검사 | 결과 |
| --- | --- |
| λ = 0이면 logits가 기준과 비트 단위로 같다 | 6/6 |
| λ = 0이면 모든 층의 attention 출력이 개입 전후·기준 실행과 비트 단위로 같다 | 6/6 |
| 지정 층 × 질문 토큰: 개입 뒤 출력 = (개입 전 + λv)를 모델 dtype으로 반올림한 값 (비트 동일) | 6/6. 더해진 양과 λv의 차이는 float32에서 ≤ 9.5e-7, bfloat16에서 ≤ 3.1e-2 (bf16 반올림) |
| 지정 층의 질문 밖 토큰은 직접 수정되지 않는다 | 6/6 |
| 지정하지 않은 층에서는 직접 수정이 없다 (개입 전후 비트 동일) | 6/6 |
| 첫 지정 층 이전 층들과 첫 지정 층의 개입 전 출력은 기준 실행과 같다 | 6/6 |
| 질문 구간 앞 토큰은 모든 층에서 기준 실행과 같다 (인과) | 6/6 |
| 후속 층 변화 (자연스러운 결과, 직접 개입 아님) | 층 2·3의 attention 출력이 기준과 달라졌다. 직접 수정 검사와 따로 표시한다 |

## 6. 강도 정의

$$o' = o + \lambda\,c_N$$

- λ = 3이면 N 이미지의 기존 기여가 모두 4배가 된다 (c_N + 3c_N).
- **attention 가중치를 다시 정규화하지 않는 기여량 증폭**이다. 다른 묶음의 기여는 그대로라서 확률 질량이 합쳐 1이 되지 않는다.
- STEP 5-3B 강화는 softmax 전 logit에 +log 4를 더했다. 그러면 N 토큰의 승산비가 4배가 되고, 다른 토큰 몫은 줄어든다(재정규화).
  - 두 연산은 같지 않다. 결과를 비교할 때 같은 개입으로 해석하지 않는다.
- 이번 단계에서는 정의만 확정했다. 실제 강도 탐색은 하지 않는다.

## 7. 후속 실험의 다섯 비교 조건 (정의·인터페이스만, `directions.condition_vector`)

| 조건 | 개입 벡터 v | 비고 |
| --- | --- | --- |
| Full N | c_N | |
| Parallel | U_P U_Pᵀ c_N | U_P: [2048, k] 정규직교 |
| Orthogonal | (I − U_P U_Pᵀ) c_N | Parallel + Orthogonal = Full |
| Random | c_N과 노름이 같은 등방 무작위 방향 | 벡터마다 (입력, 층, 토큰) 고정 seed 생성기. seed 규칙은 사전 등록 때 고정 |
| Mean Difference | μ_N − μ_P | μ_N, μ_P는 학습 분할에서 층별 c_N, c_P 평균. 크기를 그대로 둘지(`raw`) c_N 노름에 맞출지(`match`)는 사전 등록 때 정한다 |

- 합성 데이터로 성질을 확인했다 (모두 통과).
  - Parallel + Orthogonal = Full, Orthogonal ⟂ U_P, 기저 span 안 벡터는 Parallel에서 그대로다.
  - 심은 차원(rank 5)의 기저는 심은 부분공간을 복원한다.
  - Random은 노름이 같고 같은 seed에서 재현된다. Mean Difference의 `raw`/`match`가 정의대로 나온다.
- 각 조건의 개입 위치·층·λ는 Full N과 같게 둔다. 실제 비교는 하지 않았다.

## 8. P 부분공간

- **학습 벡터:** P 대조 이미지(또는 입력의 모든 P 이미지)의 c_P. L21–24 층별로, 질문 토큰을 모두 모아서(기본) 쓴다.
  - 질문 토큰 역할별(예: 마지막 토큰만) 기저는 대안이다.
- **분할:** 5-3의 N 이미지 분할(seed 53)을 그대로 쓴다. 선택 절반에서 기저·μ를 학습하고 확증 절반에서 평가한다.
  - **제약:** P 이미지 282장은 두 절반에 **모두** 나온다 (겹침 282/282). N 이미지는 겹치지 않는다 (240/240).
  - 그래서 P 기저는 평가 입력의 P 이미지와 독립이 아니다.
  - 선택지:
    - (a) 그대로 쓰고 한계로 적기
    - (b) P 이미지를 id로 둘로 나눠, 학습 P 집합의 이미지에서 나온 c_P만 기저 학습에 쓰기
    - (c) 새 P 이미지
  - **사전 등록 전에 정해야 한다.**
- **중심화**
  - 기본안: 중심화하지 않는다 (원점 기준 2차 모멘트 분해). 선형 사영 U Uᵀ c와 바로 맞는다.
  - 대안: 평균을 빼고 공분산 분해. 이때 평균 방향이 기저에 자동으로 들어가지 않으므로 μ_P/‖μ_P‖를 기저에 더해 다시 정규직교화하는 방식을 함께 정의해야 한다.
- **기저 구성:** 학습 행렬의 SVD 오른쪽 특이벡터 상위 k개로 U_P [2048, k]를 만든다 (정규직교 확인).
  - 학습 벡터는 한 절반에 P 슬롯 18,240개 × 질문 토큰 26~28개 ≈ 49만 개(층당)다.
  - 행렬 전체를 두는 대신 2차 모멘트 XᵀX(2048 × 2048, float64)를 누적해 고유분해하는 방식이 필요하다. 결과는 SVD와 같다.
- **차원 후보:** 설명 분산 90%, 95%, 고정 rank.
  - 고정 rank 값과 최종 규칙은 파일럿 평가 전에 고정한다. 결과를 보고 최적 차원을 고르지 않는다.
  - `directions.fit_basis(rule=("variance", 0.90|0.95) | ("rank", k), centered=False|True)`.

## 9. All-P 부작용 측정 규칙 (초안)

- All-P 입력에는 N이 없다. P 한 장을 가짜 N 후보로 정해 같은 개입(같은 층, 질문 토큰, λ, 조건)을 넣는다.
- **기본 규칙:** 입력의 첫 번째 P 이미지(슬롯 0)를 가짜 N으로 쓴다.
- **민감도:** 모든 슬롯을 차례로 가짜 N으로 둔다.
- 결과 변수는 Yes → No 비율과 Δm이다. 5-3B처럼 무개입 답이 Yes인 입력만 쓴다.
- 후보 규칙은 평가 전에 고정한다.
- 의미: 앞으로 N 탐지기가 P를 N으로 잘못 고른 경우의 부작용을 모의로 측정한다.
- 참고: 5-3B의 All-P 항목 3,994개는 배치에 따라 슬롯 0(2,003개) 또는 마지막 슬롯(1,991개)을 대상으로 했다.
  - 6의 기본 규칙(슬롯 0)과 다르다. 비교할 때 주의한다.

## 10. 발견한 구현상 제약

1. **batch 1, 전체 prefill만 지원한다.** 5-3B의 빠른 경로(prefix cache + suffix batch)에서 기여를 뽑으려면 cache의 k/v를 읽는 재계산이 필요하다. 본 실험 비용에 영향이 있다.
2. **sdpa·flash 커널은 attention 가중치를 내주지 않는다.** A를 다시 계산하므로 커널마다 수치 차이가 생긴다. float32는 1e-6 수준, bf16은 반올림 수준이다. eager 가중치와는 비트 단위로 같았다.
3. **bf16 모델에서는 o + λv를 bf16으로 반올림한다.** 실제로 더해지는 양은 λv와 bf16 반올림만큼 다르다 (작은 모델 최대 0.03).
4. **v는 무개입 실행의 저장값이다.** 여러 층에 넣으면 뒤 층의 '현재' c_N과 다르다 (정의상 의도).
5. **검증 비용:** 개입 위치 검사는 모든 층 출력을 보관한다 (T 2,144, float32, 36층에서 약 0.6 GB). 검증 전용이고 본 실험에서는 필요 없다.
6. **입력의 한계:** 검증 입력 3개는 규칙상 모두 같은 클래스(bed)이고 차단 모집단(Correct-No)이다. 수치 검증 용도로는 충분하지만 대표성은 없다.
7. **작은 모델 시험의 한계:** 가중치가 무작위라 실제 3B의 크기 분포·attention 패턴은 확인하지 못했다 (§4.3).
8. **P 이미지가 두 절반에 모두 나온다** (§8).

## 11. 다음 단계 (모두 별도 승인 필요)

1. 실제 3B 소규모 검증 실행: GPU 세션, 또는 로컬 bfloat16 CPU.
2. §8의 P 분할 방식, 중심화, 차원 규칙, Mean Difference 크기, Random seed 규칙을 결정한다.
3. 사전 등록 초안 작성 → 확인 → freeze.
4. 파일럿 → GPU 본 실행.

## 12. 고정본 확인 (6-1 작업 후)

다음 해시가 바뀌지 않았다.
- STEP 4 v1 (모델 파일 4, 설정 2, 평가 케이스)
- STEP 5-2 v2, 5-2b v1
- STEP 5-3 v1·v2·v3: A `dba09660`·`70d3a21b`·`7cb1fdaa`, B `6a779b39`·`0a127b18`·`9850639c`. v3는 파일·기준 해시를 다시 검사했다.
- `docs/results/step5/SHA256SUMS`는 전부 통과했다.
