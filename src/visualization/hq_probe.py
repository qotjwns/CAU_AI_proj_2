from src.visualization.probe import GRID, SERIES, TEXT_SECONDARY, plt, save

FEATURES = (("z_i", "z_i (image tokens)"), ("h_q_pooled", "h_q pooled over classes"),
            ("h_q_per_class", "h_q per class"))


def plot_hq_readability(report, output_dir, readability=None):
    """위치별 이미지 P/N AUROC: z_i vs h_q. pure 행이 있으면(STEP 3 진단) 두 번째 줄로 그린다."""
    layers = report["extraction_settings"]["layers"]
    rows = readability if readability is not None else report["readability"]
    contexts = [context for context in ("all", "pure") if any(row["context"] == context for row in rows)]
    fig, axes = plt.subplots(len(contexts), 3, figsize=(12, 3.4 * len(contexts) + 0.4), sharey=True,
                             constrained_layout=True, squeeze=False)
    for row_index, context in enumerate(contexts):
        for column, position in enumerate((1, 2, 3)):
            axis = axes[row_index, column]
            values = sorted((row for row in rows if row["position"] == position and row["context"] == context),
                            key=lambda row: row["layer"])
            for color, (feature, label) in zip(SERIES, FEATURES):
                if f"{feature}_auroc" not in values[0]:
                    continue
                axis.plot([row["layer"] for row in values], [row[f"{feature}_auroc"] for row in values],
                          marker="o", markersize=6, linewidth=2, color=color, label=label)
            axis.axhline(0.5, color=GRID, linewidth=1.5)
            axis.set_xticks(layers)
            axis.set_ylim(0.4, 1.0)
            name = "all inputs" if context == "all" else "pure inputs only"
            axis.set_title(f"Position {position} | {name} (n={values[0]['n']})", fontsize=9)
            if column == 0:
                axis.set_ylabel("AUROC (per-image P/N)")
            if row_index == len(contexts) - 1:
                axis.set_xlabel("Decoder layer")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=8)
    note = (f"status: {report['status']} | pure = all three images in the held-out fold; mixed inputs can leak through "
            "training images in other positions" if "pure" in contexts
            else f"status: {report['status']} | all images unseen by the frozen probes; same inputs for z_i and h_q")
    fig.supxlabel(note, fontsize=8, color=TEXT_SECONDARY)
    return save(fig, output_dir / "hq_vs_zi_readability.png")


def plot_localization(report, output_dir, rows=None, inputs=None):
    """N 1장 구성(PPN·PNP·NPP)에서 N 위치 정확도 (우연 1/3): 전체, VLM False-Yes, Correct-No."""
    layers = report["extraction_settings"]["layers"]
    rows = rows if rows is not None else report["localization"]
    if inputs is not None:
        rows = [row for row in rows if row.get("inputs") == inputs]
    panels = (("all", "all", "All one-N inputs"), ("vlm_group", "False-Yes", "VLM False-Yes"),
              ("vlm_group", "Correct-No", "VLM Correct-No"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, constrained_layout=True)
    for axis, (scope, value, title) in zip(axes, panels):
        n = None
        for offset, color, (feature, label) in zip((-1.2, 0, 1.2), SERIES, FEATURES):
            values = sorted((row for row in rows if row["feature"] == feature and row["scope"] == scope
                             and row["value"] == value and row["accuracy"] is not None), key=lambda row: row["layer"])
            if not values:
                continue
            n = values[0]["n"]
            error = None
            if all(row["ci_low"] is not None for row in values):
                error = [[row["accuracy"] - row["ci_low"] for row in values],
                         [row["ci_high"] - row["accuracy"] for row in values]]
            axis.errorbar([row["layer"] + offset for row in values], [row["accuracy"] for row in values], yerr=error,
                          fmt="o-", markersize=6, capsize=3, linewidth=2, color=color, label=label)
        axis.axhline(1 / 3, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        axis.set_xticks(layers)
        axis.set_ylim(0, 1)
        axis.set(title=f"{title} (n={n})", xlabel="Decoder layer")
    axes[0].set_ylabel("N localization accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=8)
    suffix = f" | inputs: {inputs}" if inputs else ""
    fig.supxlabel(f"status: {report['status']}{suffix} | argmin of the three image probabilities = N position; "
                  "dashed line = chance 1/3; 95% image bootstrap CI", fontsize=8, color=TEXT_SECONDARY)
    name = f"localization_{inputs}.png" if inputs else "localization.png"
    return save(fig, output_dir / name)
