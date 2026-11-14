"""STEP 5-3 실험 B 추가 탐색 (EXPLORATORY): N 위치별·장수별 실제 답 변화율. 저장된 입력별 결과만 쓴다.

새 GPU 실행·모델 추론·개입·기존 평가 재실행 없음. 고정본과 확증 판정은 바꾸지 않는다 (실행 전후 고정 해시 확인).

python -m experiments.step5_causal.explore_answer_flip
→ docs/results/step5/step5_3/exploratory_b_answer_flip_by_position.csv
  docs/results/step5/step5_3/exploratory_b_answer_flip_by_nimages.csv
  docs/results/step5/step5_3/exploratory_b_answer_flip_analysis.md

- 대상: 확증 절반, 선택 구간(선택 절반에서 고른 구간, analysis.json), 이미지 뒤 모든 토큰(post_image).
  기존 제외 규칙(무개입 행 답이 기준 답과 다른 입력 제외, causal_eval.b_valid)을 그대로 쓴다.
- 강화: 기준 False-Yes 입력의 Yes → No 비율. 차단: 기준 Correct-No 입력의 No → Yes 비율.
  N 대상과 P 대조는 같은 입력의 두 개입 행이다. 답이 Yes/No가 아니면 바뀌지 않은 것으로 센다 (기존 평가와 같음, 개수 보고).
- CI: 기존 평가와 같은 함수 causal_eval.summarize (N 이미지 군집·클래스 군집 percentile 부트스트랩,
  사전 등록 statistics의 반복 수·seed), 수준 95%. 부트스트랩 표본은 저장돼 있지 않지만 (군집 집합, 반복 수, seed)의
  결정적 함수라서 같은 집합이면 같은 표본이 다시 만들어진다 (기존 표 값과 정확히 같은지 검사).
  한 하위 집단의 모든 지표는 같은 군집 집합·seed를 쓰므로 같은 재표본을 공유한다. N − P 차이는 입력별 차이의 평균이다.
"""
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import read_json, write_csv
from src.common.paths import project_path, project_relative_path
from src.core.causal_check import load_run
from src.core.causal_eval import SECONDARY_LEVEL, b_index, b_population, b_valid, summarize
from src.core.causal_freeze import changed_files, experiment_config, verify_causal_freeze
from src.core.causal_plan import load_plan
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file


OUTPUT = Path("docs/results/step5/step5_3")
QUERY = "post_image"
FLIPPED = {"boost": "No", "block": "Yes"}
INPUTS = {"boost": "False-Yes (baseline Yes)", "block": "Correct-No (baseline No)"}
CHANGE = {"boost": "Yes -> No (correction)", "block": "No -> Yes (new error)"}
GROUPS = {"position": ("first", "middle", "last"), "n_images": (3, 4, 5)}
METRICS = ("flip_N", "flip_P", "flip_diff", "dm_N", "dm_P", "C")
STAT_KEYS = ("estimate", "ci_low", "ci_high", "class_ci_low", "class_ci_high")


def subgroup(table, chosen, planned, population, band, meta, reps, seed):
    keys = [i["item_id"] for i in chosen]
    rows = [table[k] for k in keys]
    n_rows = [r[(QUERY, population, "N", band)] for r in rows]
    p_rows = [r[(QUERY, population, "P", band)] for r in rows]
    target = FLIPPED[population]
    values = {
        "flip_N": [int(r["answer"] == target) for r in n_rows],
        "flip_P": [int(r["answer"] == target) for r in p_rows],
        "dm_N": [n["margin"] - r["none"]["margin"] for n, r in zip(n_rows, rows)],
        "dm_P": [p["margin"] - r["none"]["margin"] for p, r in zip(p_rows, rows)],
    }
    values["flip_diff"] = [a - b for a, b in zip(values["flip_N"], values["flip_P"])]
    values["C"] = [a - b for a, b in zip(values["dm_N"], values["dm_P"])]
    out = {"n_inputs": len({i["input_id"] for i in chosen}), "n_items": len(chosen),
           "n_n_images": len({i["anchor"] for i in chosen}), "n_classes": len({i["class"] for i in chosen}),
           "excluded_changed_no_intervention_answer": planned - len(chosen),
           "n_flip_N": sum(values["flip_N"]), "n_flip_P": sum(values["flip_P"]),
           "n_answer_not_yes_no_N": sum(r["answer"] not in ("Yes", "No") for r in n_rows),
           "n_answer_not_yes_no_P": sum(r["answer"] not in ("Yes", "No") for r in p_rows),
           "n_control_after_N": sum(i["control_index"] > i["target_index"] for i in chosen),
           "n_control_adjacent": sum(abs(i["control_index"] - i["target_index"]) == 1 for i in chosen)}
    stats = {}
    for metric in METRICS:
        row = summarize(meta, keys, values[metric], SECONDARY_LEVEL, reps, seed)
        na = []
        if out["n_n_images"] < 2:
            row["ci_low"] = row["ci_high"] = None
            na.append("N-image CI: fewer than 2 clusters")
        if out["n_classes"] < 2:
            row["class_ci_low"] = row["class_ci_high"] = None
            na.append("class CI: fewer than 2 classes")
        stats[metric] = row
        for key in STAT_KEYS:
            out[f"{metric}_{key}" if key != "estimate" else metric] = row.get(key)
        if na:
            out[f"{metric}_na_reason"] = "; ".join(na)
    return out, stats


def compare(name, mine, stored, keys=("n",) + STAT_KEYS):
    """기존 값과 비트 단위로 같은지 (같은 함수·같은 입력 순서·같은 seed)."""
    differing = {k: (mine.get(k), stored.get(k)) for k in keys if mine.get(k) != stored.get(k)}
    return {"check": name, "passed": not differing, "detail": differing or "exact match"}


def fmt(value, digits=3, sign=True):
    if value is None:
        return "N/A"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def ci(row, metric, digits=3, sign=True):
    return (f"{fmt(row[metric], digits, sign)} [{fmt(row[metric + '_ci_low'], digits, sign)}, "
            f"{fmt(row[metric + '_ci_high'], digits, sign)}]")


def class_ci(row, metric, digits=3, sign=True):
    return f"[{fmt(row[metric + '_class_ci_low'], digits, sign)}, {fmt(row[metric + '_class_ci_high'], digits, sign)}]"


def main():
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _, config = experiment_config("B")
    manifest = verify_causal_freeze("B", config, with_baseline=True)
    frozen_manifest = project_path(config["frozen_dir"]) / "manifest.json"
    analysis = read_json(project_path(config["evaluation_dir"]) / "analysis.json")
    require(analysis["preregistration"]["version"] == manifest["version"], "analysis.json is not from the frozen version")
    statistics = manifest["preregistration"]["statistics"]
    reps, seed = statistics["reps"], statistics["seed"]
    plan = load_plan(read_yaml(project_path(config["split_config"])))
    run = load_run(project_path(config["main_dir"]))
    table, items = b_index(plan, run["batches"])
    valid = b_valid(plan, table, items)
    meta = {i["item_id"]: (i["anchor"], i["class"]) for i in items.values()}

    checks, rows, stats = [], {name: [] for name in GROUPS}, {}
    for item in items.values():
        expected = "first" if item["target_index"] == 0 else (
            "last" if item["target_index"] == item["n_images"] - 1 else "middle")
        require(item["position"] == expected, f"Position label mismatch: {item['item_id']}")
    checks.append({"check": "position labels: first = slot 0, last = final slot, middle = every other slot",
                   "passed": True, "detail": f"{len(items)} items"})

    for population in ("boost", "block"):
        band = tuple(analysis["selection"][population]["band"])
        planned_all = b_population(items, "confirmation", population, None)
        chosen_all = b_population(items, "confirmation", population, valid)
        missing = [i["item_id"] for i in chosen_all for t in ("N", "P") if (QUERY, population, t, band) not in table[i["item_id"]]]
        checks.append({"check": f"{population}: N and P rows present for every analysed input", "passed": not missing,
                       "detail": f"missing {len(missing)}"})
        base = {"label": "exploratory", "intervention": population, "inputs": INPUTS[population],
                "answer_change": CHANGE[population], "half": "confirmation", "band": f"{band[0]}-{band[1]}",
                "query": "all post-image tokens", "ci_level": SECONDARY_LEVEL,
                "ci_method": f"N-image and class cluster percentile bootstrap, {reps} reps, seed {seed} (same as evaluation)"}
        overall, overall_stats = subgroup(table, chosen_all, len(planned_all), population, band, meta, reps, seed)
        stats[(population, "all", "all")] = overall_stats
        for name, levels in GROUPS.items():
            rows[name].append({**base, "group": name, "value": "all", **overall})
            for level in levels:
                keep = (lambda i, lv=level: i["position"] == lv) if name == "position" else (
                    lambda i, lv=level: i["n_images"] == lv)
                planned = b_population(items, "confirmation", population, None, keep)
                chosen = b_population(items, "confirmation", population, valid, keep)
                row, row_stats = subgroup(table, chosen, len(planned), population, band, meta, reps, seed)
                stats[(population, name, level)] = row_stats
                rows[name].append({**base, "group": name, "value": level, **row})
            parts = [r for r in rows[name] if r["intervention"] == population and r["value"] != "all"]
            sums = {k: sum(r[k] for r in parts) for k in ("n_items", "n_flip_N", "n_flip_P",
                                                         "excluded_changed_no_intervention_answer")}
            checks.append({"check": f"{population} {name}: subgroups add up to the overall counts",
                           "passed": all(sums[k] == overall[k] for k in sums),
                           "detail": {k: (sums[k], overall[k]) for k in sums}})

        # 기존 평가 값과 대조
        secondary = analysis["secondary"]
        for target, metric in (("N", "flip_N"), ("P", "flip_P")):
            stored = next(r for r in secondary if r["outcome"] == "answer_change_rate" and r["population"] == population
                          and r["arm"] == target)
            checks.append(compare(f"{population} overall {metric} = stored secondary answer change rate ({target})",
                                  overall_stats[metric], stored))
        for name, levels in GROUPS.items():
            for level in levels:
                label = f"N position {level}" if name == "position" else f"{level} images"
                for arm, metric in (("C", "C"), ("delta_N", "dm_N"), ("delta_P", "dm_P")):
                    stored = next(r for r in secondary if r["analysis"] == label and r["population"] == population
                                  and r["arm"] == arm and r["outcome"] == "margin")
                    checks.append(compare(f"{population} {label} {metric} = stored secondary", stats[(population, name, level)][metric],
                                          stored))
        primary = next(r for r in analysis["primary"] if r["population"] == population)
        for part, metric in (("contrast", "C"), ("n_arm", "dm_N"), ("p_arm_reported", "dm_P")):
            checks.append(compare(f"{population} overall {metric} estimate and n = stored primary (97.5% CI not compared)",
                                  overall_stats[metric], primary[part], keys=("n", "estimate")))
        checks.append({"check": f"{population} excluded inputs = stored primary",
                       "passed": overall["excluded_changed_no_intervention_answer"]
                       == primary["excluded_changed_no_intervention_answer"],
                       "detail": (overall["excluded_changed_no_intervention_answer"],
                                  primary["excluded_changed_no_intervention_answer"])})

    changed = changed_files(manifest["files_sha256"]) + changed_files(manifest["baseline_sha256"])
    checks.append({"check": "frozen v3 files unchanged before and after this analysis", "passed": not changed,
                   "detail": changed or f"manifest sha256 {sha256_file(frozen_manifest)[:16]}..."})
    output = project_path(OUTPUT)
    columns = list(dict.fromkeys(k for r in rows["position"] + rows["n_images"] for k in r))
    write_csv(rows["position"], output / "exploratory_b_answer_flip_by_position.csv", columns)
    write_csv(rows["n_images"], output / "exploratory_b_answer_flip_by_nimages.csv", columns)
    write_report(output / "exploratory_b_answer_flip_analysis.md", rows, checks, analysis, config, frozen_manifest,
                 run, reps, seed, started)
    failed = [c["check"] for c in checks if not c["passed"]]
    for c in checks:
        print(f"  {'OK  ' if c['passed'] else 'FAIL'} {c['check']}")
    print(f"저장: {project_relative_path(output)}/exploratory_b_answer_flip_*.csv, *_analysis.md")
    require(not failed, f"Consistency checks failed: {failed}")


def write_report(path, rows, checks, analysis, config, frozen_manifest, run, reps, seed, started):
    code = project_path("experiments/step5_causal/explore_answer_flip.py")
    lines = [
        "# STEP 5-3 실험 B 추가 분석: N 위치별·장수별 실제 답 변화율 (EXPLORATORY)",
        "",
        "> **Exploratory.** 사전 등록 판정과 무관하다. 확증 판정(B-H1 inconclusive, B-H2 robust support)과 고정본 v3는 바뀌지 않는다.",
        "> 새 GPU 실행·모델 추론·개입·기존 평가 재실행 없이, 저장된 입력별 결과(`attention/main/conditions.jsonl`)와",
        "> `attention/evaluation/analysis.json`만 읽었다.",
        "",
        "## 조건",
        "",
        f"- 대상: 확증 절반, 선택 구간 L{analysis['selection']['boost']['band'][0]}–{analysis['selection']['boost']['band'][1]}"
        f" (강화·차단 모두 선택 절반에서 고른 구간), 개입 위치 = 이미지 뒤 모든 토큰",
        "- 강화: 기준 False-Yes 입력에서 Yes → No 비율 (실제 교정률). 차단: 기준 Correct-No 입력에서 No → Yes 비율 (실제 오답 전환율)",
        "- N 대상과 P 대조는 같은 입력의 두 개입 행이다. P 대조는 계획 단계에서 정한 P 한 장이다:",
        "  이미지 토큰 수가 N과 가장 가까운 P, 동률이면 위치가 가까운 것, 그래도 같으면 앞쪽.",
        "- 분모: 기존 제외 규칙(이 경로의 무개입 답이 기준 답과 다른 입력 제외)을 적용한 뒤 남은 입력.",
        "  개입 뒤 답이 Yes/No가 아닌 행은 바뀌지 않은 것으로 센다 (기존 평가와 같음). 그런 행의 수는 CSV의 `n_answer_not_yes_no_*`에 있다.",
        "- Δm = 개입 margin − 같은 입력 무개입 margin (fp32 변형 margin, logit). C = Δm(N) − Δm(P).",
        "- Middle = N이 첫 자리도 마지막 자리도 아닌 모든 위치.",
        "",
        "## 신뢰구간",
        "",
        f"- 기존 평가와 같은 함수(`causal_eval.summarize`)를 썼다. N 이미지 군집 percentile 부트스트랩이고, 클래스 군집도 함께 낸다.",
        f"  반복 {reps}회, seed {seed}, 수준 95%이며, 모두 기존 부가 분석과 같다.",
        "- 부트스트랩 표본은 따로 저장돼 있지 않다. 다만 (군집 집합, 반복 수, seed)가 같으면 같은 표본이 다시 만들어진다.",
        "  기존 표에 이미 있던 값(전체 답 변화율, 위치·장수별 C·Δm(N)·Δm(P))이 소수점까지 똑같이 나오는 것으로 확인했다 (아래 검증).",
        "- 한 하위 집단 안의 모든 지표(N, P, N − P, Δm, C)는 같은 재표본을 쓴다. N − P 차이는 입력별 차이(짝 유지)의 평균이다.",
        "- 하위 집단마다 그 집단의 N 이미지·클래스로 재표본을 새로 만든다. 이것도 기존 부가 분석과 같다.",
        "  따라서 하위 집단 사이에서는 재표본이 공유되지 않는다.",
        "- 군집이 2개 미만이면 CI를 N/A로 둔다. 이번 결과에는 해당 사례가 없다." if not any(
            k.endswith("_na_reason") for r in rows["position"] + rows["n_images"] for k in r) else
        "- 군집이 2개 미만인 CI는 N/A로 두었다 (CSV의 `*_na_reason`).",
        "",
    ]
    for name, title in (("position", "N 위치별"), ("n_images", "이미지 장수별")):
        lines += [f"## {title} (EXPLORATORY)", ""]
        for population in ("boost", "block"):
            part = [r for r in rows[name] if r["intervention"] == population]
            head = "강화 (Boost): 기준 False-Yes → No 교정률" if population == "boost" else "차단 (Block): 기준 Correct-No → Yes 오답 전환율"
            lines += [f"### {head}", "",
                      "| 집단 | 입력 / N 이미지 / 클래스 | 제외 | N 대상 변화율 [95% CI] | P 대조 변화율 [95% CI] | N − P [95% CI] | N − P 클래스 CI |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for r in part:
                lines.append(f"| {r['value']} | {r['n_inputs']} / {r['n_n_images']} / {r['n_classes']} | "
                             f"{r['excluded_changed_no_intervention_answer']} | "
                             f"{ci(r, 'flip_N', sign=False)} ({r['n_flip_N']}) | {ci(r, 'flip_P', sign=False)} ({r['n_flip_P']}) | "
                             f"{ci(r, 'flip_diff')} | {class_ci(r, 'flip_diff')} |")
            lines += ["", "| 집단 | Δm(N) [95% CI] | Δm(P) [95% CI] | C [95% CI] | C 클래스 CI |",
                      "| --- | --- | --- | --- | --- |"]
            for r in part:
                lines.append(f"| {r['value']} | {ci(r, 'dm_N')} | {ci(r, 'dm_P')} | {ci(r, 'C')} | {class_ci(r, 'C')} |")
            lines.append("")
        lines += ["- 괄호 안 정수는 답이 바뀐 입력 수다. 변화율의 클래스 CI는 CSV에 있다.", ""]
    first = {p: next(r for r in rows["position"] if r["intervention"] == p and r["value"] == "first") for p in ("boost", "block")}
    later = {p: {v: next(r for r in rows["position"] if r["intervention"] == p and r["value"] == v) for v in ("middle", "last")}
             for p in ("boost", "block")}
    lines += [
        "## 기존 전체 결과와의 일치 검증", "",
        "| 검사 | 결과 |", "| --- | --- |",
        *[f"| {c['check']} | {'일치' if c['passed'] else '불일치: ' + str(c['detail'])} |" for c in checks],
        "",
        "## N-first 역방향 대조 효과에 대한 해석상 주의 (EXPLORATORY)", "",
        f"- 관찰: 강화에서 N-first의 C는 {ci(first['boost'], 'C')}로, 사전 등록 방향(C < 0)과 반대다.",
        f"  - 이 값은 주로 P 쪽에서 나온다: Δm(N) {ci(first['boost'], 'dm_N')}, Δm(P) {ci(first['boost'], 'dm_P')}.",
        f"  - 실제 답도 P 대조에서 더 많이 바뀌었다: Yes → No 비율이 N {fmt(first['boost']['flip_N'], sign=False)},"
        f" P {fmt(first['boost']['flip_P'], sign=False)}. 차단도 마찬가지로 No → Yes 비율이 N {fmt(first['block']['flip_N'], sign=False)},"
        f" P {fmt(first['block']['flip_P'], sign=False)}이다.",
        f"  - 차단에서는 N-first의 C가 {ci(first['block'], 'C')}, Δm(N)이 {ci(first['block'], 'dm_N')}로, 역시 예측과 반대 부호다.",
        "- 이 결과를 'N-first에서는 N 이미지 attention이 답에 반대로 작용한다'로 읽으면 안 되는 이유:",
        "  1. **사전 등록 판정 대상이 아니다.** 위치·장수 하위 집단은 부가 분석이다. 2개 개입 × 6개 집단을 보면서 다중 비교를 보정하지 않았다.",
        "  2. **구간을 위치를 모두 합쳐 골랐다.** L21–24는 선택 절반의 전체 입력에서 고른 구간이고, middle·last 입력이 대부분이다.",
        "     N-first에서 효과가 가장 큰 구간은 다를 수 있다. 위치별 구간 곡선은 이번에 계산하지 않았다.",
        f"  3. **P 대조의 위치가 N과 체계적으로 다르다.** N-first에서는 P 대조가 항상 N보다 뒤에 있다",
        f"     (강화 {first['boost']['n_control_after_N']}/{first['boost']['n_items']}, 차단 {first['block']['n_control_after_N']}/{first['block']['n_items']}).",
        f"     N-last에서는 항상 앞에 있다 (강화 {later['boost']['last']['n_items'] - later['boost']['last']['n_control_after_N']}/{later['boost']['last']['n_items']},"
        f" 차단 {later['block']['last']['n_items'] - later['block']['last']['n_control_after_N']}/{later['block']['last']['n_items']})."
        f" middle에서는 섞여 있다 (뒤: 강화 {later['boost']['middle']['n_control_after_N']}/{later['boost']['middle']['n_items']},"
        f" 차단 {later['block']['middle']['n_control_after_N']}/{later['block']['middle']['n_items']}).",
        "     그래서 C에는 'N이냐 P냐'의 차이와 '어느 자리의 이미지를 건드렸느냐'의 차이가 섞여 있다.",
        "     뒤쪽 이미지에 대한 개입이 margin을 더 크게 움직이는 위치 효과만으로도 N-first의 C 부호가 바뀔 수 있다.",
        f"  4. **N-first에서는 N 개입 자체의 효과가 작다** (강화 Δm(N) {fmt(first['boost']['dm_N'])}, middle {fmt(later['boost']['middle']['dm_N'])},"
        f" last {fmt(later['boost']['last']['dm_N'])}). 그래서 강화 N-first의 C 부호는 사실상 P 대조 쪽 값으로 정해진다.",
        "     P 대조는 'N과 무관한 같은 크기의 개입'이라는 기준선이다. 위치가 다르면 이 기준선 역할이 약해진다.",
        "  5. **개입은 attention 확률만 바꾼다.** 한 이미지를 강화하면 다른 토큰들의 attention 몫이 줄고, 차단하면 늘어난다.",
        "     따라서 P 강화의 효과에는 N을 포함한 나머지 이미지의 몫이 줄어든 영향이 함께 들어 있다.",
        "     이 분석으로는 둘을 나눌 수 없다.",
        "- 이 관찰을 확인하려면 개입 실험을 새로 설계해야 한다. 예: P 대조를 N과 같은 쪽 인접 위치로 맞춘 대조, 위치별 구간 선택.",
        "  이번 작업 범위(새 실행 금지) 밖이다.",
        "",
        "## 재현", "",
        f"- 코드: `experiments/step5_causal/explore_answer_flip.py` (sha256 {sha256_file(code)[:16]}…)",
        f"- 입력: 고정본 `{project_relative_path(frozen_manifest.parent)}` (manifest sha256 {sha256_file(frozen_manifest)[:16]}…),"
        f" 본 실행 `{config['main_dir']}` (실행 고정본 {run['meta']['frozen']['version']}), 평가 {analysis['evaluated_at']}",
        f"- 실행 시각 {started}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")



if __name__ == "__main__":
    main()
