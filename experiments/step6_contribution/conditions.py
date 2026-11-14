"""STEP 6-2 개입 조건과 작업 목록 (사전 등록 v1의 실행 쪽 정의).

층: L21–24 = 0부터 센 디코더 층 20–23 (STEP 5-3B 구간 (21, 24)와 같은 번호 규칙).
위치: 질문 구간 = 마지막 vision_end 다음 토큰부터 마지막 입력 토큰까지 (STEP 5-3B post_image query와 같음).
지점: attention 모듈 출력(o_proj 뒤, residual 덧셈 앞). o' = o + λ v, 주 강도 λ = 3 (c_N이면 N 기여가 4배).
v는 같은 입력의 별도 무개입 실행(전체 prefill, batch 1)에서 뽑은 대상 이미지 기여 c_N에서 만든다.

batch 하나 = (항목, 대상 자리). one-N 항목은 N 자리 하나, All-P 항목은 자리마다 하나(가짜 N 순환).
batch 행 순서는 고정이다 (CONDITION_ROWS). 0번 행은 같은 batch의 무개입 행이다.
"""
import hashlib

from src.core.feature_validation import require


LAYERS = (20, 21, 22, 23)
BAND = (21, 24)                      # 1부터 센 층 번호 (STEP 5-3B 표기)
LAMBDA = 3.0
RANDOM_DRAWS = 5
FIXED_RANK = 64
BASES = {"v90": ("variance", 0.90), "v95": ("variance", 0.95), "r64": ("rank", FIXED_RANK)}
PRIMARY_BASIS = "v90"
ALLP_PRIMARY_SLOT = 0                # All-P: 첫 번째 P 이미지를 가짜 N으로 (주 규칙). 나머지 자리는 민감도 분석


def spec(condition, basis=None, size="raw", draw=None, lam=LAMBDA):
    return {"condition": condition, "basis": basis, "size": size, "draw": draw, "lam": lam}


def condition_rows(pilot=False):
    rows = [spec("none", lam=0.0),
            spec("full"),
            spec("parallel", PRIMARY_BASIS), spec("orthogonal", PRIMARY_BASIS)]
    rows += [spec("random", draw=draw) for draw in range(RANDOM_DRAWS)]
    rows += [spec("mean_difference", size="raw"), spec("mean_difference", size="match"),
             spec("parallel", PRIMARY_BASIS, "match"), spec("orthogonal", PRIMARY_BASIS, "match"),
             spec("parallel", "v95"), spec("orthogonal", "v95"),
             spec("parallel", "r64"), spec("orthogonal", "r64")]
    if pilot:
        rows.append(spec("full", lam=0.0))          # 크기 0 개입 (무결성 전용)
    return rows


ROW_KEYS = ("condition", "basis", "size", "draw", "lam")


def row_key(row):
    return tuple(row[k] for k in ROW_KEYS)


def random_seed(item_id, slot, draw):
    """Random 조건 seed: 항목 id·대상 자리·추첨 번호의 SHA-256 앞 60비트 (입력 id 기반, 결과와 무관하게 고정)."""
    digest = hashlib.sha256(f"step6|{item_id}|slot{slot}|draw{draw}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def item_slots(item):
    if item["kind"] == "one_negative":
        return [item["target_index"]]
    require(item["kind"] == "all_positive", f"Unknown item kind {item['kind']}")
    return list(range(item["n_images"]))


def batches_for(item, pilot=False):
    rows = condition_rows(pilot)
    primary = item["target_index"] if item["kind"] == "one_negative" else ALLP_PRIMARY_SLOT
    return [{"batch": f"{item['item_id']}#s{slot}", "item_id": item["item_id"], "input_id": item["input_id"],
             "kind": item["kind"], "slot": slot, "primary_slot": slot == primary, "n_images": item["n_images"],
             "half": item["half"], "rows": rows} for slot in item_slots(item)]


def design():
    """설정 파일 design 항목과 비교할 실행 쪽 상수 (freeze 때 같아야 한다)."""
    return {"layers_one_based": list(BAND), "layers_zero_based": list(LAYERS), "lambda": LAMBDA,
            "random_draws": RANDOM_DRAWS, "bases": {name: list(rule) for name, rule in BASES.items()},
            "primary_basis": PRIMARY_BASIS, "allp_primary_slot": ALLP_PRIMARY_SLOT,
            "rows": [list(row_key(row)) for row in condition_rows(pilot=False)],
            "pilot_extra_rows": [list(row_key(row)) for row in condition_rows(pilot=True)[len(condition_rows()):]]}


def stage_items(plan6, plan53, stage):
    require(stage in ("pilot", "main"), f"Unknown stage {stage}")
    items = {i["item_id"]: i for i in plan53["b_items"]}
    ids = plan6["items"][stage]["one_negative"] + plan6["items"][stage]["all_positive"]
    return [items[i] for i in ids]


def stage_work(plan6, plan53, stage):
    """단계의 항목·batch 목록. 연습 실행은 모든 행을 참조 경로로 다시 계산한다."""
    items = stage_items(plan6, plan53, stage)
    batches = [b for item in items for b in batches_for(item, pilot=stage == "pilot")]
    return {"items": {i["item_id"]: i for i in items}, "batches": batches,
            "inputs": sorted({i["input_id"] for i in items}),
            "reference_batches": [b["batch"] for b in batches] if stage == "pilot" else []}
