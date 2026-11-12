import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
])
COMPOSITION_ORDER = ("PPP", "PPN", "PNP", "NPP", "PNN", "NNN")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "font.size": 9,
})


def footer(fig, report, extra=""):
    scope = ("all images unseen by the frozen probes" if "preregistration" in report
             else "target image_id unseen by its probe; prefixes may contain training images")
    note = f"status: {report['status']} | {scope}"
    fig.supxlabel(f"{note}\n{extra}" if extra else note, fontsize=8, color=TEXT_SECONDARY)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_image_level(report, output_dir):
    layers = report["extraction_settings"]["layers"]
    rows = [row for row in report["image_level"] if row["measure"] == "position1"]
    classes = ["all", *sorted({row["cls"] for row in rows if row["cls"] != "all"})]
    lookup = {(row["cls"], row["layer"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 9), constrained_layout=True)
    for axis, metric, title in zip(axes, ("accuracy", "auroc"), ("Accuracy", "AUROC")):
        values = [[lookup[(name, layer)][metric] for layer in layers] for name in classes]
        image = axis.imshow([[v if v is not None else float("nan") for v in row] for row in values],
                            cmap=SEQUENTIAL, vmin=0.5, vmax=1.0, aspect="auto")
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                if value is None:
                    axis.text(j, i, "N/A", ha="center", va="center", fontsize=7, color=TEXT_SECONDARY)
                else:
                    axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                              color=SURFACE if value >= 0.8 else TEXT)
        axis.set_xticks(range(len(layers)), [f"L{layer}" for layer in layers])
        labels = [f"All (n={lookup[('all', layers[0])]['n']})" if name == "all"
                  else f"{name} (n={lookup[(name, layers[0])]['n']})" for name in classes]
        axis.set_yticks(range(len(classes)), labels if axis is axes[0] else [])
        axis.axhline(0.5, color=SURFACE, linewidth=2)
        axis.grid(False)
        axis.set_title(f"Image-level {title} (position-1 vectors)")
        fig.colorbar(image, ax=axis, shrink=0.6, label=f"{title} (values below 0.5 shown at the lightest step)")
    footer(fig, report, "n = (class, image) pairs that appear at position 1; OOF predictions from 4 image-grouped folds")
    return save(fig, output_dir / "image_level.png")


def plot_position_context(report, output_dir):
    layers = report["extraction_settings"]["layers"]
    rows = [row for row in report["context_level"] if row["cls"] == "all"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey="row", constrained_layout=True)
    for column, position in enumerate((1, 2, 3)):
        contexts = ("all",) if position == 1 else ("pure", "mixed")
        for row_index, metric in enumerate(("accuracy", "auroc")):
            axis = axes[row_index, column]
            for color, context in zip(SERIES, contexts):
                values = sorted((row for row in rows if row["position"] == position and row["context"] == context),
                                key=lambda row: row["layer"])
                if not values or not values[0]["n"]:
                    continue
                label = "pure (no preceding image)" if position == 1 else context
                axis.plot([row["layer"] for row in values], [row[metric] for row in values], marker="o",
                          markersize=6, linewidth=2, color=color,
                          label=f"{label} (vectors={values[0]['n']}, targets={values[0]['n_targets']})")
            axis.set_xticks(layers)
            axis.set_title(f"Position {position}" if row_index == 0 else "")
            axis.set_ylabel("Accuracy" if metric == "accuracy" else "AUROC")
            if row_index == 1:
                axis.set_xlabel("Decoder layer")
            if row_index == 0:
                axis.legend(fontsize=8, loc="best")
    footer(fig, report, "Position 3 pure contexts are few; treat that series as reference only")
    return save(fig, output_dir / "position_context.png")


def plot_false_yes(report, output_dir):
    layers = report["extraction_settings"]["layers"]
    rows = [row for row in report["false_yes"] if row["level"] == "unique_input"]
    fig, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for color, group in zip(SERIES, ("False-Yes", "Correct-No")):
        values = sorted((row for row in rows if row["vlm_group"] == group), key=lambda row: row["layer"])
        axis.plot([row["layer"] for row in values], [row["negative_detection_rate"] for row in values],
                  marker="o", markersize=6, linewidth=2, color=color,
                  label=f"VLM {group} (inputs={values[0]['n_cases']}, N slots={values[0]['n_negative_slots']})")
    axis.set_xticks(layers)
    axis.set_ylim(0, 1)
    axis.set(xlabel="Decoder layer", ylabel="Negative detection rate (p < threshold)",
             title="Probe detection of negative images, unique inputs")
    axis.legend(fontsize=8, loc="lower right")
    footer(fig, report)
    outputs = [save(fig, output_dir / "false_yes_detection.png")]

    probabilities = report["negative_probabilities"]["unique_input"]
    threshold = report["settings"]["threshold"]
    fig, axes = plt.subplots(1, len(layers), figsize=(3.2 * len(layers), 3.6), sharey=True, constrained_layout=True)
    bins = [i / 20 for i in range(21)]
    for axis, layer in zip(axes, layers):
        for color, group in zip(SERIES, ("False-Yes", "Correct-No")):
            values = probabilities[str(layer)][group]
            axis.hist(values, bins=bins, density=True, histtype="step", linewidth=2, color=color,
                      label=f"{group} (n={len(values)})")
        axis.axvline(threshold, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        axis.set(title=f"L{layer}", xlabel="Probe P(object) for N image")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8, loc="upper right")
    footer(fig, report, "Dashed line: decision threshold; left of it = negative detected")
    outputs.append(save(fig, output_dir / "false_yes_probabilities.png"))
    return outputs


def plot_probe_and(report, output_dir):
    layers = report["extraction_settings"]["layers"]
    rows = [row for row in report["probe_and"] if row["level"] == "unique_input" and row["scope"] == "composition"]
    fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 4.2), sharey=True, constrained_layout=True)
    width = 0.38
    for axis, layer in zip(axes, layers):
        lookup = {row["value"]: row for row in rows if row["layer"] == layer}
        names = [name for name in COMPOSITION_ORDER if name in lookup]
        for offset, color, key, label in ((-width / 2, SERIES[0], "vlm_accuracy", "VLM"),
                                          (width / 2, SERIES[1], "probe_and_accuracy", "Probe AND")):
            axis.bar([i + offset for i in range(len(names))], [lookup[name][key] for name in names],
                     width=width - 0.04, color=color, label=label)
        axis.set_xticks(range(len(names)), [f"{name}\nn={lookup[name]['n']}" for name in names], fontsize=8)
        axis.set_ylim(0, 1)
        axis.grid(axis="x", visible=False)
        axis.set_title(f"L{layer}")
    axes[0].set_ylabel("Accuracy (unique inputs)")
    axes[0].legend(fontsize=8, loc="lower left")
    footer(fig, report, "Probe AND: Yes only if all three image probabilities are at or above the threshold")
    return save(fig, output_dir / "probe_and.png")


def plot_strict_by_class(report, output_dir):
    layers = report["extraction_settings"]["layers"]
    rows = [row for row in report["strict_discrepancy"]
            if row["level"] == "unique_input" and row["group_type"] in ("all", "class")]
    groups = ["all", *sorted({row["group"] for row in rows if row["group"] != "all"})]
    lookup = {(row["group"], row["layer"]): row for row in rows}
    fig, axis = plt.subplots(figsize=(6.2, 9), constrained_layout=True)
    values = [[lookup[(group, layer)]["strict_discrepancy_rate"] for layer in layers] for group in groups]
    image = axis.imshow([[v if v is not None else float("nan") for v in row] for row in values],
                        cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i, group in enumerate(groups):
        for j, layer in enumerate(layers):
            row = lookup[(group, layer)]
            value = row["strict_discrepancy_rate"]
            text = "N/A" if value is None else f"{value:.2f}\n{row['n_strict']}/{row['n_false_yes']}"
            axis.text(j, i, text, ha="center", va="center", fontsize=6.8,
                      color=SURFACE if value is not None and value >= 0.6 else TEXT)
    axis.set_xticks(range(len(layers)), [f"L{layer}" for layer in layers])
    axis.set_yticks(range(len(groups)), [
        f"All (N imgs={lookup[('all', layers[0])]['n_fy_negative_images']})" if group == "all"
        else f"{group} (N imgs={lookup[(group, layers[0])]['n_fy_negative_images']})" for group in groups
    ])
    axis.axhline(0.5, color=SURFACE, linewidth=2)
    axis.grid(False)
    axis.set_title("Strict discrepancy: VLM False-Yes with all 3 images correct")
    fig.colorbar(image, ax=axis, shrink=0.5, label="Strict discrepancy rate")
    footer(fig, report, "Cell: rate and strict / VLM False-Yes (unique inputs); N imgs = distinct negative images in those cases")
    return save(fig, output_dir / "strict_discrepancy_by_class.png")


def plot_key_vs_baselines(report, output_dir):
    layers = report["extraction_settings"]["layers"]
    metrics = (("image_accuracy", "Image accuracy"), ("fy_negative_detection_rate", "False-Yes N detection"),
               ("strict_discrepancy_rate", "Strict discrepancy"), ("fy_correction_rate", "False-Yes correction"),
               ("ppp_retention_rate", "PPP retention"))
    key = {row["layer"]: row for row in report["key_table"] if row["level"] == "unique_input"}
    nulls = {(row["layer"], row["metric"]): row for row in report["baselines"] if row["baseline"] == "label_permutation"}
    constants = {}
    for row in report["baselines"]:
        if row["baseline"] in ("majority_image_label", "always_no") and row["metric"] != "image_auroc":
            constants[row["metric"]] = row["value"]
    fig, axes = plt.subplots(1, len(layers), figsize=(3.4 * len(layers), 3.8), sharey=True, constrained_layout=True)
    positions = list(range(len(metrics)))[::-1]
    for axis, layer in zip(axes, layers):
        for y, (metric, _) in zip(positions, metrics):
            observed = key[layer][metric]
            low, high = key[layer].get(f"{metric}_ci_low"), key[layer].get(f"{metric}_ci_high")
            if observed is not None:
                axis.errorbar(observed, y + 0.18, xerr=[[observed - low], [high - observed]] if low is not None else None,
                              fmt="o", markersize=6, color=SERIES[0], capsize=3, linewidth=2,
                              label="Observed (95% image bootstrap CI)" if y == positions[0] else None)
            null = nulls.get((layer, metric))
            if null and null.get("value") is not None:
                axis.errorbar(null["value"], y, xerr=[[null["value"] - null["null_q025"]], [null["null_q975"] - null["value"]]],
                              fmt="o", markersize=6, color=SERIES[1], capsize=3, linewidth=2,
                              label=f"Label permutation (runs={null['n_runs']})" if y == positions[0] else None)
            if constants.get(metric) is not None:
                axis.plot(constants[metric], y - 0.18, marker="D", markersize=6, color=SERIES[2], linestyle="none",
                          label="Constant (majority / always No)" if y == positions[0] else None)
        axis.set_xlim(-0.03, 1.03)
        axis.set_yticks(positions, [label for _, label in metrics])
        axis.grid(axis="y", visible=False)
        axis.set(title=f"L{layer}", xlabel="Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: ("Observed" not in labels[i], "Constant" in labels[i]))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="outside upper center", ncol=3, fontsize=8)
    footer(fig, report, "Unique inputs; permutation bars span the 2.5-97.5% of the null runs")
    return save(fig, output_dir / "key_metrics_vs_baselines.png")


GROUP_ORDER = ("False-Yes", "Correct-No (one N)", "Correct-No", "Correct-Yes", "False-No")


def plot_group_comparison(report, output_dir):
    layers = report["extraction_settings"]["layers"]
    rows = [row for row in report["group_comparison"]
            if row["level"] == "unique_input" and row["analysis"] == "group" and row["metric"] == "strict"]
    fig, axes = plt.subplots(1, len(layers), figsize=(3.4 * len(layers), 3.6), sharey=True, constrained_layout=True)
    positions = list(range(len(GROUP_ORDER)))[::-1]
    for axis, layer in zip(axes, layers):
        lookup = {row["group"]: row for row in rows if row["layer"] == layer}
        for y, group in zip(positions, GROUP_ORDER):
            row = lookup[group]
            if row["value"] is None or row["ci_low"] is None:
                axis.text(0.02, y, "N/A", va="center", fontsize=8, color=TEXT_SECONDARY)
                continue
            color = SERIES[0] if group == "False-Yes" else TEXT_SECONDARY
            axis.errorbar(row["value"], y, xerr=[[row["value"] - row["ci_low"]], [row["ci_high"] - row["value"]]],
                          fmt="o", markersize=6, color=color, capsize=3, linewidth=2)
        axis.set_yticks(positions, [f"{group} (n={lookup[group]['n_cases']})" for group in GROUP_ORDER])
        axis.set_xlim(0, 1)
        axis.grid(axis="y", visible=False)
        axis.set(title=f"L{layer}", xlabel="All 3 images correct (strict)")
    footer(fig, report, "VLM response groups, unique inputs; 95% image bootstrap CI (product weights over the 3 images)")
    outputs = [save(fig, output_dir / "group_comparison.png")]

    series = (("matched_class_composition", "strict", "Matched class x composition: strict"),
              ("paired_negative_image", "strict", "Same N image: strict"),
              ("paired_negative_image", "negative_detection", "Same N image: N detection"))
    rows = [row for row in report["group_comparison"] if row["level"] == "unique_input"]
    fig, axis = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    for offset, color, (analysis, metric, label) in zip((-2, 0, 2), SERIES, series):
        values = sorted((row for row in rows if row["analysis"] == analysis and row["metric"] == metric
                         and row["difference"] is not None and row["ci_low"] is not None),
                        key=lambda row: row["layer"])
        if not values:
            continue
        x = [row["layer"] + offset for row in values]
        y = [row["difference"] for row in values]
        error = [[row["difference"] - row["ci_low"] for row in values], [row["ci_high"] - row["difference"] for row in values]]
        units = values[0]["n_units"]
        axis.errorbar(x, y, yerr=error, fmt="o", markersize=6, capsize=3, linewidth=2, color=color,
                      label=f"{label} ({'cells' if analysis.startswith('matched') else 'N images'}={units})")
    axis.axhline(0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    axis.set_xticks(layers)
    axis.set(xlabel="Decoder layer", ylabel="False-Yes minus Correct-No",
             title="Probe readability: VLM False-Yes vs Correct-No (one-N compositions)")
    axis.legend(fontsize=8, loc="upper left")
    footer(fig, report, "Positive = probe reads images better when the VLM is wrong; 95% image bootstrap CI")
    outputs.append(save(fig, output_dir / "group_difference.png"))
    return outputs


STEP4_METRICS = (("image_accuracy", "Image accuracy"), ("fy_negative_detection_rate", "False-Yes N detection"),
                 ("strict_discrepancy_rate", "Strict discrepancy"), ("ppp_retention_rate", "PPP retention"))


def plot_step3_comparison(report, output_dir):
    """STEP 3(OOF, 탐색적) 값과 STEP 4(고정 Probe, 독립 이미지) 값을 레이어별로 비교한다."""
    layers = report["extraction_settings"]["layers"]
    rows = {(row["layer"], row["metric"]): row for row in report["step3_comparison"]}
    primary = report["preregistration"]["primary_layer"]
    fig, axes = plt.subplots(1, len(STEP4_METRICS), figsize=(3.4 * len(STEP4_METRICS), 3.8), sharey=True,
                             constrained_layout=True)
    for axis, (metric, title) in zip(axes, STEP4_METRICS):
        for offset, color, key, label in ((-1.2, SERIES[1], "step3", "STEP 3 (OOF, exploratory)"),
                                          (1.2, SERIES[0], "step4", "STEP 4 (frozen probe, new images)")):
            points = [(layer, rows[(layer, metric)]) for layer in layers
                      if rows.get((layer, metric)) and rows[(layer, metric)][key] is not None]
            if not points:
                continue
            x = [layer + offset for layer, _ in points]
            y = [row[key] for _, row in points]
            error = None
            if key == "step4" and all(row["step4_ci_low"] is not None for _, row in points):
                error = [[row[key] - row["step4_ci_low"] for _, row in points],
                         [row["step4_ci_high"] - row[key] for _, row in points]]
            axis.errorbar(x, y, yerr=error, fmt="o", markersize=6, capsize=3, linewidth=2, color=color,
                          label=label if axis is axes[0] else None)
        axis.axvline(primary, color=GRID, linewidth=6, zorder=0)
        axis.set_xticks(layers)
        axis.set_ylim(0, 1)
        axis.set(title=title, xlabel="Decoder layer")
    axes[0].set_ylabel("Rate (unique inputs)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=8)
    fig.supxlabel(f"Shaded: pre-registered primary layer L{primary}; STEP 4 bars are 95% image bootstrap CIs | "
                  f"status: {report['status']}", fontsize=8, color=TEXT_SECONDARY)
    return save(fig, output_dir / "step3_vs_step4.png")


def plot_probe_report(report, output_dir):
    return [
        plot_strict_by_class(report, output_dir),
        *plot_group_comparison(report, output_dir),
        plot_key_vs_baselines(report, output_dir),
        plot_image_level(report, output_dir),
        plot_position_context(report, output_dir),
        *plot_false_yes(report, output_dir),
        plot_probe_and(report, output_dir),
    ]
