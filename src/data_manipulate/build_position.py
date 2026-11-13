"""STEP 5-2b 위치 격자 케이스 생성.

- 5-2 계열(N 이미지 × High/Low)의 이미지만 쓴다. 새 이미지는 없다.
- 계열·P 장수 k(2~4, 전체 3~5장)마다 a = 0..k: P1..Pk 순서를 유지하고 N을 a+1번째에 둔다.
  a = 0(첫 위치)과 a = k(끝 위치)는 5-2와 같은 입력을 5-2b 실행에서 다시 추출한다 (같은 실행 안 비교).
  a = 1..k-1(가운데)이 새 조건이다.
- 이미지 토큰은 질문 텍스트보다 앞에 있으므로, N의 표현 z_N은 N 앞의 이미지(P1..Pa)에만 의존한다.
"""
from collections import Counter
import hashlib
import json

from src.common.config import read_yaml
from src.common.data import read_jsonl
from src.common.paths import project_path
from src.core.feature_validation import require


def source_series(cases_path, p_counts):
    """5-2 one-N 케이스 → {(계열, k): {"N-first": case, "N-last": case}} (5-2 파일 순서 유지)."""
    series = {}
    for case in read_jsonl(cases_path):
        if case["condition"] != "one_negative" or case["series_step"] not in p_counts:
            continue
        key = (case["pair_id"], case["series_step"])
        require(case["placement"] not in series.setdefault(key, {}), f"Duplicate 5-2 case for {key}")
        series[key][case["placement"]] = case
    require(all(set(value) == {"N-first", "N-last"} for value in series.values()), "5-2 series lack N-first/N-last")
    return series


def position_case(first, last, a):
    """first/last: 같은 계열·k의 5-2 N-first·N-last 케이스. a = N 앞의 P 수."""
    k = first["series_step"]
    negative, positives = first["image_order"][0], first["image_order"][1:]
    require(last["image_order"] == [*positives, negative], f"{first['pair_id']} k={k}: N-first/N-last differ")
    negative_id, positive_ids = first["image_ids"][0], first["image_ids"][1:]
    require(positive_ids == first["series_image_ids"][:k], f"{first['pair_id']} k={k}: P order differs from series")
    labels = ["P"] * a + ["N"] + ["P"] * (k - a)
    role = "first" if a == 0 else "last" if a == k else "middle"
    placement = {"first": "N-first", "last": "N-last", "middle": "middle"}[role]
    record = {key: value for key, value in first.items() if key not in ("case_id", "mode")}
    record.update(
        experiment_type="position_grid", placement=placement, role=role, n_before=a, n_after=k - a,
        image_order=[*positives[:a], negative, *positives[a:]],
        image_ids=[*positive_ids[:a], negative_id, *positive_ids[a:]],
        image_labels=labels, composition="+".join(labels), negative_positions=[a],
        source_cases=({"N-first": first["case_id"], "N-last": last["case_id"]} if role == "middle"
                      else {placement: (first if role == "first" else last)["case_id"]}),
    )
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return {"case_id": f"posgrid__{first['group_id']}__{digest}", "mode": "5-2b", **record}


def build_position_cases(config):
    """계열(5-2 파일 순서) → k → a 순서. 한 계열이 12개(3 + 4 + 5) 케이스를 만든다."""
    series = source_series(project_path(config["accumulation_cases"]), set(config["p_counts"]))
    order = list(dict.fromkeys(name for name, _ in series))
    cases = []
    for name in order:
        for k in config["p_counts"]:
            pair = series[(name, k)]
            cases += [position_case(pair["N-first"], pair["N-last"], a) for a in range(k + 1)]
    return cases


def summarize(cases):
    inputs = Counter((case["prompt"], tuple(case["image_ids"])) for case in cases)
    return {
        "records": len(cases), "unique_inputs": len(inputs),
        "by_role": dict(Counter(case["role"] for case in cases)),
        "by_n_images_and_position": {f"{n}_images_a{a}": count for (n, a), count in sorted(
            Counter((case["n_images"], case["n_before"]) for case in cases).items())},
        "series": len({case["pair_id"] for case in cases}),
        "n_images_anchor": len({case["anchor_negative_id"] for case in cases}),
        "classes": len({case["class"] for case in cases}),
        "images": len({i for case in cases for i in case["image_ids"]}),
    }


def paths(config):
    output_dir = project_path(config["output_dir"])
    return {"cases": output_dir / f"{config['stem']}_cases.jsonl",
            "check": output_dir / f"{config['stem']}_consistency_check.json"}


def save_position_cases(config_path):
    config = read_yaml(project_path(config_path))
    cases = build_position_cases(config)
    target = paths(config)
    target["cases"].parent.mkdir(parents=True, exist_ok=True)
    with target["cases"].open("x", encoding="utf-8") as destination:
        for case in cases:
            destination.write(json.dumps(case, ensure_ascii=False) + "\n")
    return target, summarize(cases)
