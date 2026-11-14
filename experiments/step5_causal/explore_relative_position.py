"""STEP 5-3 실험 B 후속 탐색 (EXPLORATORY): P 대조 이미지의 N 대비 상대 위치별 재집계. 저장된 입력별 결과만 쓴다.

새 GPU 실행·모델 추론·개입·기존 평가 재실행 없음. 고정본 v3·사전 등록·확증 판정은 바꾸지 않는다 (전후 해시 확인).

python -m experiments.step5_causal.explore_relative_position
→ docs/results/step5/step5_3/exploratory_b_middle_relative_position.csv
  docs/results/step5/step5_3/exploratory_b_p_control_relative_position.csv
  docs/results/step5/step5_3/exploratory_b_relative_position_analysis.md

- 조건: 확증 절반, 선택 구간(analysis.json), 이미지 뒤 모든 토큰, 기존 제외 규칙, 같은 입력의 N/P 짝 유지.
- P-before-N: 계획의 P 대조 슬롯(control_index)이 N 슬롯(target_index)보다 앞. P-after-N: 뒤.
- A. N이 middle인 입력만 두 집단으로 나눈다. B. 모든 N 위치를 합쳐 두 집단으로 나누고 N 위치 구성과 층(위치)별 값을 함께 낸다.
- 지표·CI: explore_answer_flip.subgroup (causal_eval.summarize, 기존 평가와 같은 1,000회·seed 42, 95%).
"""
import math
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import read_yaml
from src.common.data import read_json, write_csv
from src.common.paths import project_path, project_relative_path
from src.core.causal_check import load_run
from src.core.causal_eval import SECONDARY_LEVEL, b_index, b_population, b_valid
from src.core.causal_freeze import changed_files, experiment_config, verify_causal_freeze
from src.core.causal_plan import load_plan
from src.core.feature_validation import require
from src.core.probe_freeze import sha256_file
from experiments.step5_causal.explore_answer_flip import CHANGE, INPUTS, QUERY, compare, fmt, subgroup


OUTPUT = Path("docs/results/step5/step5_3")
POSITIONS = ("first", "middle", "last")
RELATIVE = {"P-before-N": lambda i: i["control_index"] < i["target_index"],
            "P-after-N": lambda i: i["control_index"] > i["target_index"]}
COUNT_KEYS = ("n_items", "n_flip_N", "n_flip_P", "excluded_changed_no_intervention_answer")


def composition(chosen):
    counts = {p: sum(i["position"] == p for i in chosen) for p in POSITIONS}
    total = len(chosen)
    return {**{f"n_position_{p}": counts[p] for p in POSITIONS},
            **{f"share_position_{p}": counts[p] / total if total else None for p in POSITIONS}}


def main():
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _, config = experiment_config("B")
    manifest = verify_causal_freeze("B", config, with_baseline=True)
    frozen_manifest = project_path(config["frozen_dir"]) / "manifest.json"
    analysis = read_json(project_path(config["evaluation_dir"]) / "analysis.json")
    require(analysis["preregistration"]["version"] == manifest["version"], "analysis.json is not from the frozen version")
    reps, seed = manifest["preregistration"]["statistics"]["reps"], manifest["preregistration"]["statistics"]["seed"]
    plan = load_plan(read_yaml(project_path(config["split_config"])))
    run = load_run(project_path(config["main_dir"]))
    table, items = b_index(plan, run["batches"])
    valid = b_valid(plan, table, items)
    meta = {i["item_id"]: (i["anchor"], i["class"]) for i in items.values()}
    same_slot = [i["item_id"] for i in items.values() if i["control_index"] == i["target_index"]]
    checks = [{"check": "no item has the P control in the N slot", "passed": not same_slot, "detail": len(same_slot)}]

    middle_rows, all_rows, rows_by = [], [], {}
    for population in ("boost", "block"):
        band = tuple(analysis["selection"][population]["band"])
        base = {"label": "exploratory", "intervention": population, "inputs": INPUTS[population],
                "answer_change": CHANGE[population], "half": "confirmation", "band": f"{band[0]}-{band[1]}",
                "query": "all post-image tokens", "ci_level": SECONDARY_LEVEL,
                "ci_method": f"N-image and class cluster percentile bootstrap, {reps} reps, seed {seed} (same as evaluation)"}

        def cell(keep):
            planned = b_population(items, "confirmation", population, None, keep)
            chosen = b_population(items, "confirmation", population, valid, keep)
            row, stats = subgroup(table, chosen, len(planned), population, band, meta, reps, seed)
            return {**row, **composition(chosen)}, stats, chosen

        # A. middle 세분화
        whole, whole_stats, _ = cell(lambda i: i["position"] == "middle")
        rows_by[(population, "middle", "all")] = whole
        middle_rows.append({**base, "n_position": "middle", "relative_position": "all (middle)", **whole})
        for name, rel in RELATIVE.items():
            row, _, _ = cell(lambda i, rel=rel: i["position"] == "middle" and rel(i))
            rows_by[(population, "middle", name)] = row
            middle_rows.append({**base, "n_position": "middle", "relative_position": name, **row})
        stored = next(r for r in analysis["secondary"] if r["analysis"] == "N position middle"
                      and r["population"] == population and r["arm"] == "delta_P" and r["outcome"] == "margin")
        checks.append(compare(f"{population} middle (both groups) dm_P = stored secondary", whole_stats["dm_P"], stored))
        parts = [rows_by[(population, "middle", n)] for n in RELATIVE]
        checks.append({"check": f"{population} middle: P-before-N + P-after-N = middle total",
                       "passed": all(sum(p[k] for p in parts) == whole[k] for k in COUNT_KEYS),
                       "detail": {k: (sum(p[k] for p in parts), whole[k]) for k in COUNT_KEYS}})

        # B. 전체 위치
        overall, overall_stats, _ = cell(lambda i: True)
        rows_by[(population, "all", "all")] = overall
        all_rows.append({**base, "row_type": "group", "relative_position": "all", "n_position": "all", **overall})
        for name, rel in RELATIVE.items():
            row, _, _ = cell(rel)
            rows_by[(population, "all", name)] = row
            all_rows.append({**base, "row_type": "group", "relative_position": name, "n_position": "all", **row})
            for position in POSITIONS:
                stratum, _, chosen = cell(lambda i, rel=rel, position=position: rel(i) and i["position"] == position)
                if chosen:
                    rows_by[(population, "stratum", position, name)] = stratum
                    all_rows.append({**base, "row_type": "stratum (component of the group)", "relative_position": name,
                                     "n_position": position, **stratum})
        primary = next(r for r in analysis["primary"] if r["population"] == population)
        checks.append(compare(f"{population} all positions dm_P estimate and n = stored primary (97.5% CI not compared)",
                              overall_stats["dm_P"], primary["p_arm_reported"], keys=("n", "estimate")))
        groups = [rows_by[(population, "all", n)] for n in RELATIVE]
        checks.append({"check": f"{population} all positions: P-before-N + P-after-N = overall",
                       "passed": all(sum(g[k] for g in groups) == overall[k] for k in COUNT_KEYS),
                       "detail": {k: (sum(g[k] for g in groups), overall[k]) for k in COUNT_KEYS}})
        before, after = (rows_by[(population, "all", n)] for n in RELATIVE)
        checks.append({"check": f"{population} composition: N-first only in P-after-N, N-last only in P-before-N",
                       "passed": before["n_position_first"] == 0 and after["n_position_last"] == 0,
                       "detail": {"P-before-N first": before["n_position_first"], "P-after-N last": after["n_position_last"]}})
        for name in RELATIVE:
            group = rows_by[(population, "all", name)]
            strata = [rows_by[k] for k in rows_by if k[:2] == (population, "stratum") and k[3] == name]
            pooled = math.fsum(s["dm_P"] * s["n_items"] for s in strata) / group["n_items"]
            checks.append({"check": f"{population} {name}: n-weighted mean of strata dm_P = group dm_P",
                           "passed": math.isclose(pooled, group["dm_P"], rel_tol=0, abs_tol=1e-12),
                           "detail": (pooled, group["dm_P"])})
        middle_parts = [rows_by[(population, "middle", n)] for n in RELATIVE]
        strata_middle = [rows_by[(population, "stratum", "middle", n)] for n in RELATIVE]
        keys = ("n_items", "n_flip_N", "n_flip_P", "dm_P", "dm_P_ci_low", "dm_P_ci_high", "flip_P", "flip_P_ci_low")
        checks.append({"check": f"{population}: middle strata of the all-position split = A middle split (computed separately)",
                       "passed": all(a[k] == b[k] for a, b in zip(middle_parts, strata_middle) for k in keys),
                       "detail": keys})
        previous = {r["value"]: r for r in read_csv_rows(OUTPUT / "exploratory_b_answer_flip_by_position.csv")
                    if r["intervention"] == population}
        checks.append({"check": f"{population}: middle and overall counts = previous exploratory_b_answer_flip_by_position.csv",
                       "passed": all(int(previous[v][k]) == rows_by[(population, *key)][k]
                                     for v, key in (("middle", ("middle", "all")), ("all", ("all", "all")))
                                     for k in COUNT_KEYS),
                       "detail": "n_items, n_flip_N, n_flip_P, excluded"})

    changed = changed_files(manifest["files_sha256"]) + changed_files(manifest["baseline_sha256"])
    checks.append({"check": "frozen v3 files unchanged before and after this analysis", "passed": not changed,
                   "detail": changed or f"manifest sha256 {sha256_file(frozen_manifest)[:16]}..."})
    output = project_path(OUTPUT)
    write_csv(middle_rows, output / "exploratory_b_middle_relative_position.csv",
              list(dict.fromkeys(k for r in middle_rows for k in r)))
    write_csv(all_rows, output / "exploratory_b_p_control_relative_position.csv",
              list(dict.fromkeys(k for r in all_rows for k in r)))
    write_report(output / "exploratory_b_relative_position_analysis.md", rows_by, checks, analysis, config,
                 frozen_manifest, run, reps, seed, started)
    for c in checks:
        print(f"  {'OK  ' if c['passed'] else 'FAIL'} {c['check']}")
    print(f"저장: {project_relative_path(output)}/exploratory_b_*relative_position*")
    require(all(c["passed"] for c in checks), "Consistency checks failed")


def read_csv_rows(path):
    import csv
    with project_path(path).open(encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def ci2(row, metric, sign=True):
    """N 이미지 CI와 클래스 CI를 한 칸에."""
    return (f"{fmt(row[metric], sign=sign)} [{fmt(row[metric + '_ci_low'], sign=sign)}, {fmt(row[metric + '_ci_high'], sign=sign)}]"
            f"<br>클래스 [{fmt(row[metric + '_class_ci_low'], sign=sign)}, {fmt(row[metric + '_class_ci_high'], sign=sign)}]")


def size(row):
    return f"{row['n_inputs']} / {row['n_n_images']} / {row['n_classes']}"


def write_report(path, rows_by, checks, analysis, config, frozen_manifest, run, reps, seed, started):
    code = project_path("experiments/step5_causal/explore_relative_position.py")
    r = rows_by
    band = analysis["selection"]["boost"]["band"]
    lines = [
        "# STEP 5-3 실험 B 후속 분석: P 대조의 상대 위치별 재집계 (EXPLORATORY)",
        "",
        "> **Exploratory.** 사전 등록 판정과 무관하며, 다중 비교 보정은 하지 않았다.",
        "> 확증 판정(A-H1·A-H2 no meaningful difference, B-H1 inconclusive, B-H2 robust support)과 고정본 v3는 그대로다.",
        "> 새 GPU 실행·모델 추론·개입·기존 평가 재실행 없이, 저장된 입력별 결과(`attention/main/conditions.jsonl`)와",
        "> `attention/evaluation/analysis.json`만 다시 집계했다.",
        "",
        "## 조건",
        "",
        f"- 확증 절반, 선택 구간 L{band[0]}–{band[1]}, 이미지 뒤 모든 토큰. 기존 제외 규칙을 적용하고 같은 입력의 N/P 짝을 유지했다.",
        "- 강화는 기준 False-Yes 입력의 Yes → No 비율, 차단은 기준 Correct-No 입력의 No → Yes 비율이다.",
        "- 상대 위치는 계획 단계에서 정한 P 대조 슬롯이 N 슬롯보다 앞인지(P-before-N) 뒤인지(P-after-N)다.",
        "  P 대조는 이미지 토큰 수가 N과 가장 가까운 P이고, 동률이면 위치가 가까운 것이다.",
        "  상대 위치를 맞추려고 고른 것이 아니라서 두 집단의 크기와 구성이 다르다.",
        f"- CI는 기존 평가와 같은 함수(`causal_eval.summarize`)로 계산했다: N 이미지 군집과 클래스 군집 percentile 부트스트랩,"
        f" {reps}회, seed {seed}, 95%.",
        "  집단마다 그 집단의 군집으로 재표본을 만든다. 같은 집단 안의 N, P, N − P, Δm은 같은 재표본을 쓴다.",
        "- 두 집단 사이의 차이(P-after − P-before)에 대한 CI는 계산하지 않았다. CI가 겹치는지 여부를 검정으로 읽지 않는다.",
        "",
        "## A. N이 middle인 입력: P 대조 상대 위치별 (EXPLORATORY)",
        "",
    ]
    for population, head in (("boost", "강화 (Boost): 기준 False-Yes → No"), ("block", "차단 (Block): 기준 Correct-No → Yes")):
        lines += [f"### {head}", "",
                  "| 집단 | 입력 / N 이미지 / 클래스 | N 대상 변화율 | P 대조 변화율 | N − P | Δm(P) |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name in ("all", "P-before-N", "P-after-N"):
            row = r[(population, "middle", name)]
            lines.append(f"| {'middle 전체' if name == 'all' else name} | {size(row)} | {ci2(row, 'flip_N', False)} | "
                         f"{ci2(row, 'flip_P', False)} | {ci2(row, 'flip_diff')} | {ci2(row, 'dm_P')} |")
        lines.append("")
    lines += ["- 각 칸의 첫 줄은 추정 [95% N 이미지 군집 CI], 둘째 줄은 95% 클래스 군집 CI다. Δm(N)과 C는 CSV에 있다.", "",
              "## B. 모든 N 위치: P 대조 상대 위치별 Δm(P) (EXPLORATORY)", ""]
    for population, head in (("boost", "강화 (Boost)"), ("block", "차단 (Block)")):
        lines += [f"### {head}", "",
                  "| 집단 | 입력 / N 이미지 / 클래스 | N 위치 구성 (first / middle / last) | Δm(P) | P 대조 변화율 |",
                  "| --- | --- | --- | --- | --- |"]
        for name in ("all", "P-before-N", "P-after-N"):
            row = r[(population, "all", name)]
            mix = " / ".join(f"{row[f'n_position_{p}']} ({row[f'share_position_{p}']:.0%})" for p in POSITIONS)
            lines.append(f"| {'전체' if name == 'all' else name} | {size(row)} | {mix} | {ci2(row, 'dm_P')} | "
                         f"{ci2(row, 'flip_P', False)} |")
        lines += ["", "구성 요소 (각 집단을 N 위치로 나눈 층):", "",
                  "| 집단 | N 위치 | 입력 | Δm(P) | P 대조 변화율 |", "| --- | --- | --- | --- | --- |"]
        for name in ("P-before-N", "P-after-N"):
            for position in POSITIONS:
                if (population, "stratum", position, name) in r:
                    row = r[(population, "stratum", position, name)]
                    lines.append(f"| {name} | {position} | {size(row)} | {ci2(row, 'dm_P')} | {ci2(row, 'flip_P', False)} |")
        lines.append("")
    bb, ba = r[("boost", "all", "P-before-N")], r[("boost", "all", "P-after-N")]
    mb, ma = r[("boost", "middle", "P-before-N")], r[("boost", "middle", "P-after-N")]
    kb, ka = r[("block", "middle", "P-before-N")], r[("block", "middle", "P-after-N")]
    fb, fk = r[("boost", "stratum", "first", "P-after-N")], r[("block", "stratum", "first", "P-after-N")]
    lines += [
        "## 위치 구성 차이에 대한 주의",
        "",
        f"- 두 집단의 N 위치 구성은 구조적으로 다르다. N-first 입력은 모두 P-after-N에, N-last 입력은 모두 P-before-N에 들어간다.",
        f"  강화 기준으로 P-before-N은 middle {bb['share_position_middle']:.0%}·last {bb['share_position_last']:.0%},"
        f" P-after-N은 first {ba['share_position_first']:.0%}·middle {ba['share_position_middle']:.0%}이다.",
        "- 따라서 B의 전체 위치 비교는 'P가 N 앞이냐 뒤냐'와 'N이 몇 번째 자리냐'를 함께 비교한다. 상대 위치만의 차이로 읽지 않는다.",
        "  N 위치와 상대 위치가 둘 다 있는 층은 middle뿐이다 (A).",
        f"- 강화에서 전체 위치 Δm(P)는 P-before-N {fmt(bb['dm_P'])}, P-after-N {fmt(ba['dm_P'])}로 부호가 다르다.",
        f"  그러나 층별로 보면 이 차이는 대부분 N-first 층(Δm(P) {fmt(fb['dm_P'])})에서 온다.",
        f"  middle 안에서는 P-before-N {fmt(mb['dm_P'])}, P-after-N {fmt(ma['dm_P'])}로 부호가 같고 차이가 작다.",
        f"  차단 middle도 {fmt(kb['dm_P'])}와 {fmt(ka['dm_P'])}로 부호가 같다. 차단 N-first 층은 {fmt(fk['dm_P'])}다.",
        "",
        "## 해석 범위 (EXPLORATORY)",
        "",
        "- 허용하는 가장 강한 해석은 **'N 뒤에 위치한 이미지 토큰이 N 관련 정보를 싣고 있을 가능성이 있다'**까지다.",
        "  확정된 내부 메커니즘으로 주장하지 않는다.",
        "- 이번 자료가 그 가능성을 뒷받침하는 정도는 제한적이다.",
        "  - N-first에서는 뒤쪽 P 강화가 Yes margin을 낮추고, 뒤쪽 P 차단이 Yes 쪽으로 움직인다. 이는 그 가능성과 들어맞는다.",
        f"  - 하지만 middle 입력에서는 P가 N 뒤에 있어도 같은 방향의 효과가 나타나지 않았다.",
        f"    강화 Δm(P)는 두 집단 모두 양수다 ({fmt(mb['dm_P'])}, {fmt(ma['dm_P'])}).",
        f"    차단 Δm(P)는 두 집단 모두 음수다 ({fmt(kb['dm_P'])}, {fmt(ka['dm_P'])}).",
        "  - 따라서 N-first에서 본 역방향 대조를 '상대 위치(N 뒤)'만으로 설명할 수 없다.",
        "    N-first 입력에 국한된 현상일 수 있지만, 이번 자료로는 원인을 가를 수 없다.",
        "- 이번 비교로 분리되지 않는 것:",
        "  1. **P 대조의 상대 위치와 이미지 의미(P/N)가 분리되지 않았다.** N 앞뒤의 같은 자리에 N이 아닌 이미지만 놓은 대조는 없다.",
        "  2. **attention 재분배의 간접 효과가 있을 수 있다.** P를 강화하면 N과 다른 이미지의 attention 몫이 줄고, 차단하면 늘어난다.",
        "     Δm(P)에는 P 자체의 효과와 이 재분배 효과가 섞여 있다.",
        "  3. **집단 크기·구성이 다르다.** 장수, 클래스, N 이미지 구성이 두 집단에서 같지 않다. 맞추거나 보정하지 않았다.",
        "  4. **사후 하위 집단이다.** 구간 L21–24는 위치를 합친 선택 절반에서 골랐다.",
        "     여러 하위 집단을 보면서 다중 비교를 보정하지 않았으므로, 개별 CI가 0을 배제해도 일반화된 유의성으로 읽지 않는다.",
        "- 확인하려면 새 개입 설계가 필요하다. 예: 같은 슬롯에 N이 아닌 이미지를 둔 위치 맞춤 대조, N 앞뒤 대조를 모두 갖춘 입력. 이번 범위(새 실행 금지) 밖이다.",
        "",
        "## 기존 집계와의 일치 검증",
        "",
        "| 검사 | 결과 |", "| --- | --- |",
        *[f"| {c['check']} | {'일치' if c['passed'] else '불일치: ' + str(c['detail'])} |" for c in checks],
        "",
        "## 재현",
        "",
        f"- 코드: `experiments/step5_causal/explore_relative_position.py` (sha256 {sha256_file(code)[:16]}…),"
        " `explore_answer_flip.py`의 집계 함수를 그대로 쓴다",
        f"- 입력: 고정본 `{project_relative_path(frozen_manifest.parent)}` (manifest sha256 {sha256_file(frozen_manifest)[:16]}…),"
        f" 본 실행 `{config['main_dir']}` (실행 고정본 {run['meta']['frozen']['version']}), 평가 {analysis['evaluated_at']}",
        f"- 실행 시각 {started}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
