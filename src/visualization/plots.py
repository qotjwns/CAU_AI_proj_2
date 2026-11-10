import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.metrics import COMPOSITIONS


def save_figure(fig, output_dir, name):
    fig.savefig(output_dir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def rate_bars(axis, labels, values, title, ylabel):
    heights = [value * 100 if value is not None else 0 for value in values]
    bars = axis.bar(range(len(labels)), heights, color="#4477aa")
    axis.set_xticks(range(len(labels)), labels)
    axis.set(title=title, ylabel=ylabel, ylim=(0, 110))
    for bar, value in zip(bars, values):
        label = f"{value:.1%}" if value is not None else "N/A"
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                  label, ha="center", va="bottom", fontsize=8)


def plot_accuracy(report, output_dir):
    modes = [row["mode"] for row in report["by_mode"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = ["All records", *modes]
    rows = [report["overall"], *report["by_mode"]]
    rate_bars(axes[0], labels, [row["accuracy"] for row in rows], "Accuracy", "Accuracy (%)")
    rate_bars(axes[1], labels, [row["invalid_rate"] for row in rows],
              "Invalid responses", "Invalid (%)")
    save_figure(fig, output_dir, "overview.png")

    categories = sorted({row["object"] for row in report["by_class"]})
    fig, axis = plt.subplots(figsize=(max(10, len(categories) * 0.65), 5), constrained_layout=True)
    width = 0.8 / len(modes)
    for index, mode in enumerate(modes):
        by_class = {row["object"]: row for row in report["by_class"] if row["mode"] == mode}
        positions = [i - 0.4 + width / 2 + index * width for i in range(len(categories))]
        values = [by_class[name]["accuracy"] * 100 if name in by_class else float("nan")
                  for name in categories]
        axis.bar(positions, values, width=width, label=f"Mode {mode}")
    axis.set_xticks(range(len(categories)), categories, rotation=45, ha="right")
    axis.set(title="Accuracy by class", ylabel="Accuracy (%)", ylim=(0, 105))
    axis.legend()
    save_figure(fig, output_dir, "accuracy_by_class.png")

    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4),
                             squeeze=False, constrained_layout=True)
    for axis, mode in zip(axes[0], modes):
        by_composition = {row["composition"]: row for row in report["by_composition"]
                          if row["mode"] == mode}
        labels = COMPOSITIONS[mode]
        values = [by_composition[label]["accuracy"] if label in by_composition else None
                  for label in labels]
        rate_bars(axis, labels, values, f"Mode {mode}: composition", "Accuracy (%)")
    save_figure(fig, output_dir, "accuracy_by_composition.png")


def plot_confusion(report, output_dir):
    rows = report["by_mode"]
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4),
                             squeeze=False, constrained_layout=True)
    for axis, row in zip(axes[0], rows):
        matrix = [[row["confusion_matrix"][gt][prediction] for prediction in ("Yes", "No", "Invalid")]
                  for gt in ("Yes", "No")]
        heatmap = axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_xticks(range(3), ["Yes", "No", "Invalid"])
        axis.set_yticks(range(2), ["Yes", "No"])
        axis.set(title=f"Mode {row['mode']}: confusion matrix", xlabel="Prediction", ylabel="Ground truth")
        threshold = max(max(values) for values in matrix) / 2
        for y, values in enumerate(matrix):
            for x, value in enumerate(values):
                axis.text(x, y, str(value), ha="center", va="center",
                          color="white" if value > threshold else "black")
        fig.colorbar(heatmap, ax=axis, shrink=0.8)
    save_figure(fig, output_dir, "confusion_matrix.png")


def plot_negative_positions(report, output_dir):
    rows = report["by_negative_position"]
    if not rows:
        (output_dir / "negative_position_error.png").unlink(missing_ok=True)
        return
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    rate_bars(axis, [f"Image {row['negative_position']}" for row in rows],
              [row["error_rate"] for row in rows],
              "Mode B: error rate by negative image position", "Error (%)")
    save_figure(fig, output_dir, "negative_position_error.png")


def plot_margins(report, output_dir):
    rows = report["logit_margins"]
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4),
                             squeeze=False, constrained_layout=True)
    for axis, row in zip(axes[0], rows):
        samples, labels = [], []
        for key in ("correct", "incorrect"):
            if row[key]:
                samples.append(row[key])
                labels.append(f"{key} (n={len(row[key])})")
        axis.hist(samples, bins=25, histtype="step", linewidth=1.6, label=labels)
        axis.axvline(0, color="gray", linestyle="--", linewidth=1)
        axis.set(title=f"Mode {row['mode']}: first-token margins",
                 xlabel="Yes logit - No logit", ylabel="Case count")
        axis.legend()
    save_figure(fig, output_dir, "logit_margin_distribution.png")


def plot_pairs(report, output_dir):
    rows = report["pair_transitions"]
    if not rows:
        (output_dir / "pair_changes.png").unlink(missing_ok=True)
        return
    ranks = {mode: {composition: i for i, composition in enumerate(order)}
             for mode, order in COMPOSITIONS.items()}
    rows = sorted(rows, key=lambda row: (row["mode"], ranks[row["mode"]][row["from_composition"]],
                                        ranks[row["mode"]][row["to_composition"]]))
    labels = [f"{row['mode']}: {row['from_composition']} -> {row['to_composition']}" for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    rate_bars(axes[0], labels, [row["prediction_change_rate"] for row in rows],
              "Prediction changes (valid responses only)", "Changed (%)")
    deltas = [row["mean_margin_delta"] for row in rows]
    axes[1].bar(range(len(rows)), deltas, color=["#228833" if value >= 0 else "#cc6677" for value in deltas])
    axes[1].set_xticks(range(len(rows)), labels)
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set(title="Mean paired margin change", ylabel="Target margin - source margin")
    for axis in axes:
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    save_figure(fig, output_dir, "pair_changes.png")


def plot_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for plot in (plot_accuracy, plot_confusion, plot_negative_positions, plot_margins, plot_pairs):
        plot(report, output_dir)
