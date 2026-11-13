"""STEP 5-3 그림·표: 평가 결과(analysis.json)에 이미 계산된 값만 쓴다. 새 평가·부트스트랩은 하지 않는다.

모두 판정과 무관하며 이름과 제목에 descriptive(기술 통계) 또는 exploratory(탐색)를 붙인다.
- A 층별 답 변화율 곡선 (같은 묶음 donor, C1 donor; 복원·오염): descriptive
- B 4층 구간별 평균 C 곡선 (차단·강화; 이미지 뒤 모든 토큰, 마지막 토큰): descriptive
- B 답 변화율 (강화: False-Yes → No, 차단: Correct-No → Yes; N 대상과 P 대조, 95% CI): exploratory
- B N 위치별·장수별 C와 Δm(N) (95% CI): exploratory
"""
from src.visualization.probe import GRID, SERIES, TEXT_SECONDARY, plt, save


A_TITLES = {
    "restore": "restore: False-Yes input <- donor h_q\nrate of answer 'No'",
    "contaminate": "contaminate: N-last Correct-No input <- donor h_q\nrate of answer 'Yes'",
}
B_TITLES = {
    "block": "block (Correct-No inputs; preregistered direction C > 0)",
    "boost": "boost x4 (False-Yes inputs; preregistered direction C < 0)",
}
HALF_STYLE = {"confirmation": {"linestyle": "-", "linewidth": 2.0, "alpha": 1.0},
              "selection": {"linestyle": ":", "linewidth": 1.4, "alpha": 0.85}}


def _finish(fig, axes, report, lines):
    """범례와 설명을 그림 아래에 둔다 (곡선을 가리지 않게)."""
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8, frameon=False)
    pre = report["preregistration"]
    fig.text(0.5, -0.11, f"status: {report['status']} (this figure: DESCRIPTIVE, no decision) | preregistration "
             f"{pre['version']} | evaluated {report['evaluated_at']}\n" + "\n".join(lines),
             ha="center", va="top", fontsize=8, color=TEXT_SECONDARY)


def _style(axis):
    axis.grid(True, axis="y", color=GRID, linewidth=0.8)
    axis.grid(False, axis="x")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def plot_a_layer_curves(report, path):
    """실험 A: 층 ℓ에서 h_q를 이식했을 때 목표 답 비율. 같은 묶음 donor와 C1 donor는 같은 단위(C1 있는 단위)."""
    curves = report["descriptive"]["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for axis, direction in zip(axes, ("restore", "contaminate")):
        layer_star = report["selection"][direction]["layer"]
        counts = {}
        for half in ("confirmation", "selection"):
            rows = sorted((c for c in curves if c["direction"] == direction and c["half"] == half),
                          key=lambda c: c["layer"])
            layers = [c["layer"] for c in rows]
            style = HALF_STYLE[half]
            for key, color, name in (("same_set_rate_with_c1", SERIES[0], "same-set donor"),
                                     ("c1_rate", SERIES[1], "C1 donor (other N image)")):
                axis.plot(layers, [c[key] for c in rows], color=color, solid_capstyle="round",
                          label=f"{name}, {half} half", **style)
            counts[half] = rows[0]["n_units_with_c1"]
        axis.axvline(layer_star, color=TEXT_SECONDARY, linewidth=0.8)
        axis.text(layer_star + 0.4, 0.93, f"l* = {layer_star}\n(selection half)", fontsize=8, color=TEXT_SECONDARY,
                  va="top")
        axis.axhline(0.5, color=GRID, linewidth=1.0)
        axis.set_xlim(0, 36)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks(range(0, 37, 4))
        axis.set_xlabel("patched layer l")
        axis.set_title(f"{A_TITLES[direction]} | units: {counts['confirmation']} confirmation, "
                       f"{counts['selection']} selection", fontsize=9.5)
        _style(axis)
    axes[0].set_ylabel("share of units with the target answer")
    fig.suptitle("DESCRIPTIVE - Experiment A: answer change rate by patched layer", fontsize=11)
    _finish(fig, axes, report, [
        "l = residual stream after l decoder layers (0 = embeddings, 36 = before the final norm); horizontal line = 0.5",
        "units with a C1 donor that are valid under the frozen exclusion rule; same-set and C1 curves use the same units"])
    save(fig, path)
    return path


def plot_b_band_curves(report, path):
    """실험 B: 구간별 평균 C = Δm(N) − Δm(P). 평균만 있고 CI는 없다 (선택 구간의 CI는 주 판정 표)."""
    curves = report["descriptive"]["band_curves"]
    bands = list(dict.fromkeys(c["band"] for c in curves))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for axis, population in zip(axes, ("block", "boost")):
        selected = "-".join(str(v) for v in report["selection"][population]["band"])
        index = bands.index(selected)
        axis.axvspan(index - 0.4, index + 0.4, color=GRID, alpha=0.6, linewidth=0)
        counts = {}
        for query, color, name in (("post_image", SERIES[0], "all post-image tokens (primary)"),
                                   ("last_token", SERIES[2], "last token only")):
            for half in ("confirmation", "selection"):
                rows = {c["band"]: c for c in curves
                        if c["population"] == population and c["query"] == query and c["half"] == half}
                axis.plot(range(len(bands)), [rows[b]["mean_C"] for b in bands], color=color,
                          marker="o" if half == "confirmation" else None, markersize=4,
                          label=f"{name}, {half} half", **HALF_STYLE[half])
                counts[half] = rows[bands[0]]["n"]
        minimum = 0.5 if population == "block" else -0.5
        axis.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        axis.axhline(minimum, color=GRID, linewidth=1.2)
        axis.text(len(bands) - 0.6, minimum, f"minimum effect {minimum:+.1f}", fontsize=7.5, color=TEXT_SECONDARY,
                  ha="right", va="bottom" if minimum > 0 else "top")
        axis.set_ylim(*((-0.05, 0.62) if population == "block" else (-0.82, 0.05)))
        axis.set_xticks(range(len(bands)), [f"{b}\n(selected)" if b == selected else b for b in bands])
        axis.set_xlabel("4-layer band (decoder layers); shaded = band selected on the selection half")
        axis.set_title(f"{B_TITLES[population]}\ninputs: {counts['confirmation']} confirmation, "
                       f"{counts['selection']} selection", fontsize=9.5)
        _style(axis)
    axes[0].set_ylabel("mean C = dm(N) - dm(P)  (fp32 variant margin, logit)")
    fig.suptitle("DESCRIPTIVE - Experiment B: mean contrast C by attention band", fontsize=11)
    _finish(fig, axes, report, ["means over inputs valid under the frozen exclusion rule, no CI "
                                "(CIs for the selected band are in the primary table)"])
    save(fig, path)
    return path


def _ci_row(row):
    return {k: row.get(k) for k in ("n", "n_anchors", "estimate", "ci_low", "ci_high", "class_ci_low",
                                    "class_ci_high", "level")}


def b_answer_change_rows(report):
    """강화(False-Yes 입력)에서 No로, 차단(Correct-No 입력)에서 Yes로 바뀐 비율. N 대상과 P 대조."""
    rows = []
    for row in report["secondary"]:
        if row["outcome"] != "answer_change_rate":
            continue
        population = row["population"]
        rows.append({"label": "exploratory", "intervention": population, "band": "-".join(map(str, row["band"])),
                     "inputs": "False-Yes (baseline Yes)" if population == "boost" else "Correct-No (baseline No)",
                     "change": "Yes -> No" if population == "boost" else "No -> Yes",
                     "target": "N image" if row["arm"] == "N" else "P image (control)", **_ci_row(row)})
    return rows


def b_subgroup_rows(report):
    """N 위치별·장수별 C와 Δm(N) (선택 구간, 이미지 뒤 모든 토큰)."""
    rows = []
    for row in report["secondary"]:
        name = row["analysis"]
        if row["arm"] not in ("C", "delta_N") or row["outcome"] != "margin":
            continue
        if name.startswith("N position "):
            group, value = "N position", name.removeprefix("N position ")
        elif name.endswith(" images"):
            group, value = "image count", name.removesuffix(" images")
        else:
            continue
        rows.append({"label": "exploratory", "intervention": row["population"], "band": "-".join(map(str, row["band"])),
                     "group": group, "value": value, "arm": "C" if row["arm"] == "C" else "dm(N)", **_ci_row(row)})
    return rows
