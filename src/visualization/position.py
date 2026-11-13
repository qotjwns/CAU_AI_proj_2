"""STEP 5-2b 그림: 위치 격자, 주 판정(H1b·H2b), 실행 간 위치별 N 영향력, z_N 변화 (analysis.json만 읽는다)."""
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb

from src.visualization.probe import GRID, SEQUENTIAL, SERIES, SURFACE, TEXT, TEXT_SECONDARY, plt, save


COUNTS = (3, 4, 5)
DIVERGING = LinearSegmentedColormap.from_list("orange_gray_blue", [SERIES[1], "#f3f2ee", SERIES[0]])


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:+.{digits}f}"


def hypothesis_title(report, name):
    """사전 등록 표기 그대로: 'H1b: N-last advantage' 형식."""
    statement = report["preregistration"]["hypotheses"][name]
    return f"{name}: {statement.split(':')[0]}"


def footer(fig, report, scope, class_ci=False):
    counts, pre = report["counts"], report["preregistration"]
    bars = "; thin offset bars = class-level cluster bootstrap" if class_ci else ""
    fig.supxlabel(
        f"status: {report['status']} | {scope} | preregistration {pre['version']} | {counts['series']} series = "
        f"{counts['n_images_anchor']} N images x High/Low, {counts['classes']} classes\n"
        f"pairs within series; N-image cluster bootstrap ({pre['statistics']['reps']} reps){bars}",
        fontsize=8, color=TEXT_SECONDARY)


def ink(cmap, norm, value):
    """배경 밝기에 맞춘 글자색 (어두운 칸은 밝은 글자)."""
    red, green, blue = to_rgb(cmap(norm(value)))
    return SURFACE if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5 else TEXT


def heatmap(axis, values, cmap, norm, title, digits=3, colorbar_label=""):
    """values[(n_images, a)] = (estimate, ci_low, ci_high). 행: 3~5장, 열: a = 0..4."""
    grid = [[values.get((n, a), (None,))[0] for a in range(5)] for n in COUNTS]
    shown = [[float("nan") if v is None else v for v in row] for row in grid]
    if norm is None:
        present = [v[0] for v in values.values()]
        norm = Normalize(min(present), max(present) if max(present) > min(present) else min(present) + 1e-9)
    image = axis.imshow(shown, cmap=cmap, norm=norm, aspect="auto")
    for i, n in enumerate(COUNTS):
        for a in range(5):
            if (n, a) not in values:
                axis.text(a, i, "-", ha="center", va="center", color=TEXT_SECONDARY, fontsize=8)
                continue
            estimate, low, high = values[(n, a)]
            role = "first" if a == 0 else "last" if a == n - 1 else "middle"
            axis.text(a, i, f"{estimate:.{digits}f}\n[{low:.{digits}f}, {high:.{digits}f}]\n{role}", ha="center",
                      va="center", fontsize=7.5, color=ink(cmap, norm, estimate))
    axis.set_xticks(range(5), [f"a={a}" for a in range(5)])
    axis.set_yticks(range(len(COUNTS)), [f"{n} images" for n in COUNTS])
    axis.set_xlabel("a = P images before N (b = images - 1 - a after N)")
    axis.set_title(title, fontsize=9.5)
    axis.grid(False)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.02, label=colorbar_label)


def plot_grid(report, output_dir):
    rows = [r for r in report["exploratory"]["grid"] if r["strength"] == "pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for axis, outcome, title in ((axes[0], "false_yes_rate", "False-Yes rate by N position"),
                                 (axes[1], "margin_fp32_variant", "Yes - No margin (fp32 variant) by N position")):
        values = {(r["n_images"], r["a"]): (r["estimate"], r["ci_low"], r["ci_high"])
                  for r in rows if r["outcome"] == outcome}
        if outcome == "false_yes_rate":
            top = max(v[0] for v in values.values())
            heatmap(axis, values, SEQUENTIAL, Normalize(0, max(top, 0.01)), title, 3, "False-Yes rate")
        else:
            span = max(abs(v[0]) for v in values.values()) or 1.0
            heatmap(axis, values, DIVERGING, TwoSlopeNorm(0, -span, span), title, 2, "logit")
    footer(fig, report, "cell levels with 95% CI (descriptive)")
    return save(fig, output_dir / "position_grid.png")


def forest(axis, rows, minimum=None, legend=True):
    for index, (label, row) in enumerate(rows):
        y = len(rows) - 1 - index
        if row.get("estimate") is None:
            continue
        if row.get("class_ci_low") is not None:
            axis.plot([row["class_ci_low"], row["class_ci_high"]], [y - 0.2] * 2, color=TEXT_SECONDARY, linewidth=1.2,
                      label="class-level CI" if index == 0 else None)
        axis.plot([row["ci_low"], row["ci_high"]], [y, y], color=SERIES[0], linewidth=4.5, solid_capstyle="round",
                  label="N-image CI" if index == 0 else None)
        axis.plot(row["estimate"], y, "o", color=SERIES[0], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    if minimum is not None:
        axis.axvline(minimum, color=SERIES[1], linewidth=1.3, linestyle="--", label=f"minimum effect ({minimum:g})")
    if legend:
        axis.legend(fontsize=8, loc="lower right")
    axis.set_yticks(range(len(rows)), [label for label, _ in reversed(rows)], fontsize=8.5)
    axis.set_ylim(-0.7, len(rows) - 0.3)
    axis.grid(axis="y", visible=False)


def plot_decision(report, output_dir):
    minimum = report["preregistration"]["minimum_effect"]["false_yes_rate"]
    primary = {row["hypothesis"]: row for row in report["primary"]}
    secondary = [r for r in report["secondary"]["contrasts"] if r["outcome"] == "false_yes_rate"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.1]})
    forest(axes[0], [(f"{hypothesis_title(report, name)}\n({row['contrast']})\n{row['label']}", row)
                     for name, row in primary.items()], minimum, legend=False)
    axes[0].set_xlabel("Difference in False-Yes rate (3-5 images, equal weight)")
    axes[0].set_title("Primary: H1b and H2b (97.5% CI, Bonferroni)", fontsize=9.5)
    forest(axes[1], [(f"{r['hypothesis']} {r['scope']}", {k: v for k, v in r.items() if not k.startswith("class_")})
                     for r in secondary], minimum, legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=8.5)
    axes[1].set_xlabel("Difference in False-Yes rate")
    axes[1].set_title("Secondary: by image count (95% CI, no decision)", fontsize=9.5)
    footer(fig, report, "preregistered decision rule", class_ci=True)
    return save(fig, output_dir / "primary_decision.png")


def plot_position_influence(report, output_dir):
    rows = report["exploratory"]["position_influence"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for axis, outcome, title, digits in ((axes[0], "yes_rate", "Yes-rate gap: STEP 5-2 All-P minus STEP 5-2b one-N", 3),
                                         (axes[1], "margin_fp32_variant", "Margin gap: STEP 5-2 All-P minus STEP 5-2b one-N", 2)):
        values = {(r["n_images"], r["a"]): (r["estimate"], r["ci_low"], r["ci_high"]) for r in rows if r["outcome"] == outcome}
        heatmap(axis, values, SEQUENTIAL, None, title, digits, "All-P minus one-N")
    footer(fig, report, "cross-run comparison, exploratory (95% CI)")
    return save(fig, output_dir / "exploratory_position_influence.png")


def plot_z_n_by_a(report, output_dir):
    rows = [r for r in report["exploratory"].get("z_n_by_a", []) if r.get("a", 0) > 0 and "logit_estimate" in r]
    if not rows:
        return None
    layers = sorted({r["layer"] for r in rows})
    colors = [SEQUENTIAL(0.35 + 0.65 * i / max(len(layers) - 1, 1)) for i in range(len(layers))]
    fig, axis = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    for color, layer in zip(colors, layers):
        for run, style in (("step5_2b", "-"), ("step5_2 N-last (same format, a = k)", "--")):
            points = sorted((r for r in rows if r["layer"] == layer and r["run"] == run), key=lambda r: r["a"])
            if points:
                axis.plot([p["a"] for p in points], [p["logit_estimate"] for p in points], style, color=color,
                          marker="o", markersize=5, label=f"L{layer} {'5-2b' if run == 'step5_2b' else '5-2'}")
                axis.vlines([p["a"] for p in points], [p["logit_ci_low"] for p in points],
                            [p["logit_ci_high"] for p in points], color=color, linewidth=1)
    axis.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    axis.set_xticks(sorted({r["a"] for r in rows}))
    axis.set_xlabel("a = P images before N")
    axis.set_ylabel("Change in frozen z_i probe logit vs a = 0")
    axis.set_title("Exploratory: N image representation by preceding P images", fontsize=9.5)
    axis.legend(fontsize=7.5, ncol=2)
    footer(fig, report, "exploratory (95% CI); content and absolute position not separated")
    return save(fig, output_dir / "exploratory_z_n_by_a.png")


def plot_position_report(report, output_dir):
    paths = [plot_grid(report, output_dir), plot_decision(report, output_dir),
             plot_position_influence(report, output_dir), plot_z_n_by_a(report, output_dir)]
    return [path for path in paths if path is not None]
