"""STEP 6-2 평가 자체 시험 (CPU, 가짜 결과). 실제 계획 파일의 항목 구조로 가짜 본 실행 결과를 만들어
analysis.analyze가 사전 등록대로 판정하는지 본다. 실행 결과 파일(results/)은 열지 않는다 (열면 실패).

시나리오 (사전 등록 최소 효과 −0.25, 97.5% N 이미지 군집, 클래스 군집 robust):
1. 효과 0 (잡음만): H1·H2 모두 "no meaningful difference". 제외 항목(재구성 실패·무개입 답 변화·기록 없음)에
   큰 효과를 심어도 결과에 들어가지 않고 개수가 맞아야 한다.
2. 효과를 심은 경우 (Full −1.0, Random −0.05, Parallel −0.2, Orthogonal −0.9): 둘 다 "robust support"
3. 대비만 충족 (Full 0, Random +0.6 / Parallel +0.7, Orthogonal +0.1): 둘 다 "inconclusive", 미충족 = n_arm
4. 일부 클래스에만 효과 (항목이 가장 많은 3개 클래스 −4, 나머지 +0.1): "support (cross-class generalization uncertain)"
5. All-P 부작용: 자리 0에서 Full만 10% 답 바뀜 → Full 경고 하나, 다른 조건은 경고 없음
6. 5-3B 나란히 비교: 공통 항목 수와 Δm 추정이 심은 값과 맞음

python -m experiments.step6_contribution.selftest_eval
"""
import builtins
import io
import random

from src.common.config import read_yaml
from src.common.paths import project_path
from src.core.causal_plan import load_plan
from src.core.feature_validation import require
from experiments.step6_contribution.analysis import analyze
from experiments.step6_contribution.conditions import BASES, LAYERS, batches_for
from experiments.step6_contribution.freeze import load_config
from experiments.step6_contribution.plan import load_step6_plan


LABEL = {"robust": "robust support", "support": "support (cross-class generalization uncertain)",
         "none": "no meaningful difference", "other": "inconclusive"}
FIELDS = ["c_norm_mean", "c_over_o_norm", "target_attention_share", "cos_mean_difference"]
FIELDS += [f"{kind}_{name}" for name in BASES for kind in ("orthogonal_ratio", "parallel_energy")]


def arm_name(spec):
    name = spec["condition"]
    if name in ("parallel", "orthogonal"):
        return f"{name}_{spec['basis']}" + ("_match" if spec["size"] == "match" else "")
    if name == "mean_difference":
        return f"mean_difference_{spec['size']}"
    return name


def fake_run(plan53, plan6, effect, seed=0, noise=0.3, allp_effect=None, invalid=(), changed=(), missing=()):
    rng = random.Random(seed)
    items53 = {i["item_id"]: i for i in plan53["b_items"]}
    batches, extraction = {}, {}
    for kind in ("one_negative", "all_positive"):
        for item_id in plan6["items"]["main"][kind]:
            if item_id in missing:
                continue
            item = items53[item_id]
            extraction[item_id] = {"valid": item_id not in invalid}
            base = plan53["inputs"][item["input_id"]]["baseline"]
            for batch in batches_for(item):
                rows, none_margin = [], base["fp32_variant_logit_margin"]
                none_answer = base["prediction"]
                if item_id in changed:
                    none_answer = "No" if none_answer == "Yes" else "Yes"
                for spec in batch["rows"]:
                    if spec["condition"] == "none":
                        margin, answer = none_margin, none_answer
                    else:
                        if kind == "one_negative":
                            delta = effect(item, arm_name(spec)) + rng.gauss(0, noise)
                        else:
                            delta = allp_effect(item, batch["slot"], arm_name(spec)) if allp_effect else 0.0
                        margin = none_margin + delta
                        answer = "No" if margin < 0 else "Yes"
                    rows.append({**spec, "answer": answer, "raw": answer, "margin": margin, "margin_fp32": margin,
                                 "margin_bf16": margin, "margin_bf16_variant": margin, "first_token": 0})
                summary = {str(layer): {field: rng.random() for field in FIELDS} for layer in LAYERS}
                batches[batch["batch"]] = {**{k: batch[k] for k in ("batch", "item_id", "input_id", "kind", "slot",
                                                                    "primary_slot", "n_images", "half")},
                                           "rows": rows, "vector_summary": summary}
    return {"meta": {"status": "completed"}, "batches": batches, "extraction": extraction}


class Guard:
    """실행 결과 파일을 열면 실패시킨다 (가짜 데이터만 쓰는지 확인)."""

    def __init__(self):
        self.original = builtins.open

    def __enter__(self):
        original = self.original

        def guarded(file, *args, **kwargs):
            if "results/" in str(file):
                raise AssertionError(f"selftest_eval opened a result file: {file}")
            return original(file, *args, **kwargs)
        builtins.open = io.open = guarded
        return self

    def __exit__(self, *exc):
        builtins.open = io.open = self.original


def labels(result):
    return {row["hypothesis"]: (row["label"], row["unmet_conditions"]) for row in result["primary"]}


def main():
    config = load_config()
    preregistration = config["preregistration"]
    plan6 = load_step6_plan(config["plan_dir"])
    plan53 = load_plan(read_yaml(project_path(config["split_config"])))
    items53 = {i["item_id"]: i for i in plan53["b_items"]}
    one_n = plan6["items"]["main"]["one_negative"]
    report = {}
    with Guard():
        # 1. 효과 0 + 제외 항목에 큰 효과
        rng = random.Random(5)
        excluded = rng.sample(one_n, 60)
        invalid, changed, missing = set(excluded[:20]), set(excluded[20:50]), set(excluded[50:])
        bad = invalid | changed | missing

        def null(item, arm):
            return -50.0 if item["item_id"] in bad and arm in ("full", "orthogonal_v90") else 0.0
        result = analyze(plan53, plan6, fake_run(plan53, plan6, null, invalid=invalid, changed=changed, missing=missing),
                         preregistration)
        got = labels(result)
        require(got["H1"][0] == LABEL["none"] and got["H2"][0] == LABEL["none"], f"Null scenario: {got}")
        counts = result["counts"]
        require(counts == {"planned": len(one_n), "analyzed": len(one_n) - 60, "excluded_missing": 10,
                           "excluded_invalid_extraction": 20, "excluded_changed_none_answer": 30}, f"Counts: {counts}")
        require(abs(result["primary"][0]["contrast"]["estimate"]) < 0.05, "Excluded items leaked into the estimate")
        sensitivity = [r for r in result["secondary"] if r["analysis"].startswith("sensitivity: D1 including")][0]
        require(sensitivity["n"] == len(one_n) - 30, f"Sensitivity population: {sensitivity['n']}")
        report["null"] = got

        # 2. 효과를 심은 경우
        planted = {"full": -1.0, "random": -0.05, "parallel_v90": -0.2, "orthogonal_v90": -0.9}
        result = analyze(plan53, plan6, fake_run(plan53, plan6, lambda item, arm: planted.get(arm, 0.0)),
                         preregistration)
        got = labels(result)
        require(got["H1"][0] == LABEL["robust"] and got["H2"][0] == LABEL["robust"], f"Planted scenario: {got}")
        h1 = result["primary"][0]
        require(abs(h1["contrast"]["estimate"] + 0.95) < 0.05 and abs(h1["reported_arm"]["estimate"] + 0.05) < 0.05,
                f"Planted estimates: {h1['contrast']['estimate']}, {h1['reported_arm']['estimate']}")
        report["planted"] = got

        # 3. 대비만 충족
        contrast_only = {"full": 0.0, "random": 0.6, "parallel_v90": 0.7, "orthogonal_v90": 0.1}
        result = analyze(plan53, plan6, fake_run(plan53, plan6, lambda item, arm: contrast_only.get(arm, 0.0)),
                         preregistration)
        got = labels(result)
        require(got["H1"] == (LABEL["other"], ["n_arm"]) and got["H2"] == (LABEL["other"], ["n_arm"]),
                f"Contrast-only scenario: {got}")
        report["contrast_only"] = got

        # 4. 일부 클래스에만 효과
        by_class = {}
        for i in one_n:
            by_class[items53[i]["class"]] = by_class.get(items53[i]["class"], 0) + 1
        classes = sorted(by_class, key=lambda c: (-by_class[c], c))[:3]      # 항목이 가장 많은 3개 클래스

        def few(item, arm):          # 다른 클래스는 +0.1: 영향 클래스가 안 뽑힌 재표본은 0보다 커야 클래스 CI가 0을 넘는다
            if arm not in ("full", "orthogonal_v90"):
                return 0.0
            return -4.0 if item["class"] in classes else 0.1
        result = analyze(plan53, plan6, fake_run(plan53, plan6, few), preregistration)
        got = labels(result)
        require(got["H1"][0] == LABEL["support"] and got["H2"][0] == LABEL["support"], f"Few-class scenario: {got}")
        report["few_classes"] = got

        # 5. All-P 부작용
        allp_ids = plan6["items"]["main"]["all_positive"]
        flipped = set(random.Random(9).sample(allp_ids, len(allp_ids) // 10))

        def allp(item, slot, arm):
            return -100.0 if arm == "full" and slot == 0 and item["item_id"] in flipped else 0.0
        run = fake_run(plan53, plan6, lambda item, arm: 0.0, allp_effect=allp)
        result = analyze(plan53, plan6, run, preregistration)
        require(len(result["warnings"]) == 1 and "condition full" in result["warnings"][0],
                f"Side-effect warnings: {result['warnings']}")
        row = [r for r in result["all_positive_side_effects"]["rows"]
               if r["rule"] == "slot 0 (primary)" and r["arm"] == "full" and r["outcome"] == "yes_to_no_rate"][0]
        require(abs(row["estimate"] - len(flipped) / len(allp_ids)) < 1e-9, f"All-P full rate {row['estimate']}")
        report["side_effect_warnings"] = result["warnings"]

        # 6. 5-3B 나란히 비교
        table53 = {}
        for item_id in one_n:
            base = plan53["inputs"][items53[item_id]["input_id"]]["baseline"]
            m0 = base["fp32_variant_logit_margin"]
            table53[item_id] = {"none": {"answer": base["prediction"], "margin": m0},
                                ("post_image", "boost", "N", (21, 24)): {"answer": "No" if m0 - 0.66 < 0 else "Yes",
                                                                       "margin": m0 - 0.66}}
        table53[one_n[0]]["none"]["answer"] = "No"                    # 5-3B에서 제외되는 항목 하나
        result = analyze(plan53, plan6, run, preregistration, {"table": table53, "band": [21, 24]})
        comparison = result["comparison_5_3b"]
        require(comparison["n_common"] == len(one_n) - 1, f"Common items: {comparison['n_common']}")
        first = comparison["rows"][0]
        require(abs(first["estimate"] + 0.66) < 1e-9, f"5-3B delta estimate {first['estimate']}")
        report["comparison_n_common"] = comparison["n_common"]
    print(report)
    print("통과")


if __name__ == "__main__":
    main()
