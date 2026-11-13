"""STEP 5-2 그림: 누적 곡선, 주 판정, 민감도, H2, 탐색 (analysis.json만 읽는다)."""
from src.visualization.probe import GRID, SEQUENTIAL, SERIES, SURFACE, TEXT, TEXT_SECONDARY, plt, save


PLACEMENTS = ("N-first", "N-last")
STEPS = (1, 2, 3, 4)
OUTCOMES = {
    "false_yes_rate": "False-Yes rate",
    "yes_rate": "Yes rate",
    "margin_fp32_variant": "Yes - No margin (logit, fp32 variant)",
    "margin_bf16": "Yes - No margin (logit, bf16)",
}
STRENGTHS = {"High": (SERIES[0], 1.4, "High-P series"), "Low": (SERIES[1], 1.4, "Low-P series"),
             "pooled": (TEXT, 2.2, "Pooled (High + Low), 95% CI")}
TICKS = [f"{k + 1}\n(P{k})" for k in STEPS]


def footer(fig, report, scope="preregistered analysis", class_ci=False):
    counts, pre = report["counts"], report["preregistration"]
    bars = "; thin offset bars = class-level cluster bootstrap" if class_ci else ""
    fig.supxlabel(
        f"status: {report['status']} | {scope} | preregistration {pre['version']} | {counts['series']} series = "
        f"{counts['n_images_anchor']} N images x High/Low, {counts['classes']} classes\n"
        f"pairs within series; 95% CI = N-image cluster bootstrap ({pre['statistics']['reps']} reps){bars}",
        fontsize=8, color=TEXT_SECONDARY)


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:+.{digits}f}"


def primary_text(row):
    return f"P4 - P1 = {fmt(row['estimate'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]: {row['label']}"


def plot_curves(report, output_dir):
    """one-N False-Yes·margin과 All-P 대조군을 이미지 장수별로 (High / Low / 합침)."""
    lookup = {(r["condition"], r["placement"], r["outcome"], r["strength"], r["series_step"]): r
              for r in report["levels"]}
    primary = {(r["placement"], r["outcome"]): r for r in report["primary"]}
    panels = (("one_negative", "false_yes_rate", "One-N inputs: False-Yes rate"),
              ("one_negative", "margin_fp32_variant", "One-N inputs: Yes - No margin"),
              ("all_positive", "margin_fp32_variant", "All-P control: Yes - No margin"),
              ("all_positive", "yes_rate", "All-P control: Yes rate"))
    fig, axes = plt.subplots(len(panels), 2, figsize=(10.5, 13), sharex=True, constrained_layout=True)
    for row, (condition, outcome, title) in enumerate(panels):
        for col, placement in enumerate(PLACEMENTS):
            axis = axes[row, col]
            for strength in ("High", "Low", "pooled"):
                color, width, label = STRENGTHS[strength]
                points = [lookup[(condition, placement, outcome, strength, k)] for k in STEPS]
                x = [point["n_images"] for point in points]
                if strength == "pooled":
                    axis.fill_between(x, [p["ci_low"] for p in points], [p["ci_high"] for p in points],
                                      color=TEXT, alpha=0.10, linewidth=0)
                axis.plot(x, [p["estimate"] for p in points], color=color, linewidth=width, marker="o",
                          markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
            if condition == "one_negative" and outcome.startswith("margin"):
                axis.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
            subtitle = (f"\n{primary_text(primary[(placement, outcome)])}"
                        if condition == "one_negative" and (placement, outcome) in primary else "")
            axis.set_title(f"{placement} | {title}{subtitle}", fontsize=9)
            axis.set_ylabel(OUTCOMES[outcome])
            axis.set_xticks([k + 1 for k in STEPS], TICKS)
            if row == len(panels) - 1:
                axis.set_xlabel("Images in the input (P images in one-N inputs)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=8.5)
    footer(fig, report, scope="primary outcomes and All-P control by image count")
    return save(fig, output_dir / "accumulation_curves.png")


def forest(axis, rows, minimum=None, highlight=None):
    """rows: [(라벨, 행)] 위에서 아래로. 굵은 막대 = N 이미지 CI, 얇은 막대 = 클래스 CI(있으면)."""
    for index, (label, row) in enumerate(rows):
        y = len(rows) - 1 - index
        if row.get("estimate") is None:
            continue
        color = TEXT if highlight and highlight(label) else SERIES[0]
        if row.get("class_ci_low") is not None:
            axis.plot([row["class_ci_low"], row["class_ci_high"]], [y - 0.2, y - 0.2], color=TEXT_SECONDARY,
                      linewidth=1.2, label="class-level CI" if index == 0 else None)
        axis.plot([row["ci_low"], row["ci_high"]], [y, y], color=color, linewidth=4.5, solid_capstyle="round",
                  label="N-image CI" if index == 0 else None)
        axis.plot(row["estimate"], y, "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    if minimum is not None:
        axis.axvline(minimum, color=SERIES[1], linewidth=1.3, linestyle="--", label=f"minimum effect ({minimum:g})")
    axis.legend(fontsize=8, loc="lower right")
    axis.set_yticks(range(len(rows)), [label for label, _ in reversed(rows)], fontsize=8.5)
    axis.set_ylim(-0.7, len(rows) - 0.3)
    axis.grid(axis="y", visible=False)


def plot_decision(report, output_dir):
    """주 판정: 배치별 P4 - P1 (False-Yes, margin), 최소 효과선, 판정 표기."""
    minimum = report["preregistration"]["minimum_effect"]
    limits = {"false_yes_rate": minimum["false_yes_rate"], "margin_fp32_variant": minimum["margin"]}
    decision = report["decision"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4), constrained_layout=True)
    for axis, outcome, verdict in zip(axes, limits, (decision["false_yes"], decision["margin_co_primary"])):
        rows = {row["placement"]: row for row in report["primary"] if row["outcome"] == outcome}
        forest(axis, [(f"{placement}\n{rows[placement]['label']}", rows[placement]) for placement in PLACEMENTS],
               limits[outcome])
        axis.set_xlabel(f"P4 - P1 change in {OUTCOMES[outcome]}")
        role = "primary (decides H1)" if outcome == "false_yes_rate" else "co-primary"
        axis.set_title(f"{role}: {verdict['verdict']}", fontsize=9.5)
    footer(fig, report, scope="preregistered decision rule", class_ci=True)
    return save(fig, output_dir / "primary_decision.png")


def plot_sensitivity(report, output_dir):
    """주 대비(P4 - P1)를 민감도 분석과 비전 fp32 부분집합에서 다시 본다."""
    sensitivity = report["sensitivity"]
    pick = {(r["variant"], r["placement"], r["outcome"]): r for r in sensitivity["rows"]
            if r["analysis"] == "P4_minus_P1" and r["condition"] == "one_negative"}
    main = {(r["placement"], r["outcome"]): r for r in report["primary"]}
    vision = {(r["run"], r["placement"], r["outcome"]): r for r in sensitivity["vision_fp32"]["rows"]}
    minimum = report["preregistration"]["minimum_effect"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    for axis, outcome, limit in zip(axes, ("false_yes_rate", "margin_fp32_variant"),
                                    (minimum["false_yes_rate"], minimum["margin"])):
        rows = []
        for placement in PLACEMENTS:
            rows.append((f"{placement}: main analysis", main[(placement, outcome)]))
            if outcome == "margin_fp32_variant":
                rows.append((f"{placement}: bf16 'Yes'/'No' margin", pick[("bf16_margin", placement, "margin_bf16")]))
            rows.append((f"{placement}: sign-consistent inputs", pick[("sign_consistent", placement, outcome)]))
            rows.append((f"{placement}: integrity exclusion", pick[("integrity_exclusion", placement, outcome)]))
            rows.append((f"{placement}: fp32 subset, bf16 vision", vision[("main_run_bf16_vision", placement, outcome)]))
            rows.append((f"{placement}: fp32 subset, fp32 vision", vision[("vision_fp32", placement, outcome)]))
        forest(axis, rows, limit, highlight=lambda label: label.endswith("main analysis"))
        for index in range(1, len(PLACEMENTS)):
            axis.axhline(len(rows) - index * len(rows) // len(PLACEMENTS) - 0.5, color=GRID, linewidth=1)
        axis.set_xlabel(f"P4 - P1 change in {OUTCOMES[outcome]}")
        axis.set_title(f"Sensitivity: {OUTCOMES[outcome]}", fontsize=9.5)
    excluded = {s["variant"]: s["excluded_inputs"] for s in sensitivity["summary"]}
    counts = sensitivity["vision_fp32"]["counts"]
    footer(fig, report, class_ci=True, scope=(
        f"sensitivity (excluded inputs: sign {excluded['sign_consistent']}, integrity "
        f"{excluded['integrity_exclusion']}; fp32 subset {counts['series']} series)"))
    return save(fig, output_dir / "sensitivity.png")


def plot_h2(report, output_dir):
    """H2: 같은 N 이미지의 High - Low (P 장수별과 평균)."""
    rows = [r for r in report["contrasts"] if r["analysis"] == "H2_high_minus_low"]
    slopes = {(r["placement"], r["outcome"]): r for r in report["contrasts"] if r["analysis"] == "H2_slope_of_difference"}
    lookup = {(r["placement"], r["outcome"], str(r["p_count"])): r for r in rows}
    keys = [str(k) for k in STEPS] + ["mean_1_to_4"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    for row, outcome in enumerate(("false_yes_rate", "margin_fp32_variant")):
        for col, placement in enumerate(PLACEMENTS):
            axis = axes[row, col]
            for x, key in enumerate(keys):
                point = lookup[(placement, outcome, key)]
                color = TEXT if key == "mean_1_to_4" else SERIES[0]
                axis.plot([x, x], [point["ci_low"], point["ci_high"]], color=color, linewidth=4,
                          solid_capstyle="round")
                axis.plot(x, point["estimate"], "o", color=color, markersize=8, markeredgecolor=SURFACE,
                          markeredgewidth=2)
            axis.axhline(0, color=TEXT_SECONDARY, linewidth=1)
            slope = slopes[(placement, outcome)]
            axis.set_title(f"{placement} | High - Low {OUTCOMES[outcome].split(' (')[0]}\n"
                           f"slope over P: {fmt(slope['estimate'])} [{fmt(slope['ci_low'])}, {fmt(slope['ci_high'])}]",
                           fontsize=9)
            axis.set_xticks(range(len(keys)), [f"P{k}" for k in STEPS] + ["mean"])
            axis.set_ylabel(f"High - Low, {OUTCOMES[outcome].split(' (')[0]}")
            axis.grid(axis="x", visible=False)
    footer(fig, report, scope="secondary H2, no decision rule")
    return save(fig, output_dir / "h2_high_minus_low.png")


def plot_classes(report, output_dir):
    """탐색: 클래스별 P4 - P1."""
    rows = report["exploratory"]["per_class"]
    if not rows:
        return None
    classes = sorted({r["class"] for r in rows})
    lookup = {(r["class"], r["placement"], r["outcome"]): r for r in rows}
    order = sorted(classes, key=lambda name: sum(lookup[(name, p, "false_yes_rate")]["estimate"] or 0
                                                 for p in PLACEMENTS))
    sizes = sorted({r["n_units"] for r in rows})
    size = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]}-{sizes[-1]}"
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.5), sharey=True, constrained_layout=True)
    for axis, outcome in zip(axes, ("false_yes_rate", "margin_fp32_variant")):
        for offset, color, placement in zip((0.17, -0.17), SERIES, PLACEMENTS):
            for y, name in enumerate(order):
                point = lookup[(name, placement, outcome)]
                if point["estimate"] is None:
                    continue
                axis.plot([point["ci_low"], point["ci_high"]], [y + offset] * 2, color=color, linewidth=1.6)
                axis.plot(point["estimate"], y + offset, "o", color=color, markersize=6, markeredgecolor=SURFACE,
                          markeredgewidth=1.2, label=placement if y == 0 else None)
        axis.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        axis.set_yticks(range(len(order)), order)
        axis.set_xlabel(f"P4 - P1 change in {OUTCOMES[outcome]}")
        axis.set_title(f"Per class ({size} series each): {OUTCOMES[outcome].split(' (')[0]}", fontsize=9.5)
        axis.legend(fontsize=8, loc="lower right")
        axis.grid(axis="y", visible=False)
    footer(fig, report, scope="exploratory, not part of any decision")
    return save(fig, output_dir / "exploratory_per_class.png")


def plot_z_n(report, output_dir):
    """탐색: N 이미지 고정 z_i 점수(logit)의 변화, 같은 계열 2장 입력 기준."""
    rows = [r for r in report["exploratory"]["z_n"] if r["analysis"] == "change_vs_two_image_input"
            and r["scale"] == "logit"]
    if not rows:
        return None
    layers = sorted({r["layer"] for r in rows})
    colors = [SEQUENTIAL(0.35 + 0.65 * i / max(len(layers) - 1, 1)) for i in range(len(layers))]
    lookup = {(r["placement"], r["layer"], r["p_count"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
    for axis, placement in zip(axes, PLACEMENTS):
        for offset, color, layer in zip((-0.09, -0.03, 0.03, 0.09), colors, layers):
            points = [lookup[(placement, layer, k)] for k in STEPS[1:]]
            x = [p["n_images"] + offset for p in points]
            axis.vlines(x, [p["ci_low"] for p in points], [p["ci_high"] for p in points], color=color, linewidth=1.2)
            axis.plot(x, [p["estimate"] for p in points], color=color, marker="o", markersize=5, linewidth=1.6,
                      label=f"L{layer}")
        axis.axhline(0, color=TEXT_SECONDARY, linewidth=1)
        axis.set_xticks([k + 1 for k in STEPS[1:]], TICKS[1:])
        axis.set_xlabel("Images in the input (P images)")
        axis.set_title(f"{placement}: N image z_i logit change vs. the 2-image input", fontsize=9.5)
        axis.legend(fontsize=8, ncol=4, loc="lower left")
    axes[0].set_ylabel("Change in frozen z_i probe logit (N image)")
    footer(fig, report, scope="exploratory; N-first change is bf16 numerical noise by construction")
    return save(fig, output_dir / "exploratory_z_n.png")


def plot_accumulation_report(report, output_dir):
    paths = [plot_curves(report, output_dir), plot_decision(report, output_dir), plot_sensitivity(report, output_dir),
             plot_h2(report, output_dir), plot_classes(report, output_dir), plot_z_n(report, output_dir)]
    return [path for path in paths if path is not None]
