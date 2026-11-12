from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.data import normalize_category


def plot_distances(report, output_dir):
    groups = defaultdict(list)
    for row in report["summary"]:
        groups[(row["mode"], row["category"], row["comparison"])].append(row)
    outputs = []
    for (mode, category, comparison), rows in sorted(groups.items()):
        fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        labels = sorted({row["response_group"] for row in rows}, key=lambda label: (label != "All", label))
        for label in labels:
            values = sorted((row for row in rows if row["response_group"] == label), key=lambda row: row["layer"])
            counts = sorted({row["n"] for row in values})
            count_text = str(counts[0]) if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
            axis.errorbar(
                [row["layer"] for row in values], [row["mean"] for row in values],
                yerr=[row["std"] if row["std"] is not None else 0 for row in values],
                label=f"{label} (n={count_text})", marker="o", capsize=3,
                linestyle="--" if label == "All" else "-", color="black" if label == "All" else None,
            )
        axis.set_xticks(report["extraction_settings"]["layers"])
        axis.set(xlabel="Decoder layer", ylabel="Cosine distance (same N1)",
                 title=f"Mode {mode} | {category} | {comparison}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
        fig.supxlabel("Mean ± sample SD (SD only for n > 1); incomplete comparison groups excluded", fontsize=9)
        folder = output_dir / mode / normalize_category(category).replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"{comparison}.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

    reference = report["reference_comparison"]
    if reference and reference["rows"]:
        rows = reference["rows"]
        fig, axis = plt.subplots(figsize=(max(10, 2.1 * len(rows)), 5.5), constrained_layout=True)
        axis.bar([i - 0.2 for i in range(len(rows))], [row["expected_mean"] for row in rows],
                 width=0.4, label="Historical summary")
        for index, row in enumerate(rows):
            if row["observed_mean"] is not None:
                axis.bar(index + 0.2, row["observed_mean"], width=0.4, color="#ff7f0e")
                axis.annotate(f"n={row['n']}", (index + 0.2, row["observed_mean"]),
                              xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
            else:
                axis.text(index + 0.2, 0, "N/A", ha="center", va="bottom", fontsize=8)
        axis.bar([], [], color="#ff7f0e", label="Current sample")
        axis.set_xticks(range(len(rows)), [
            f"{row['mode']} | {row['comparison']}\nL{row['layer']} | {row['response_group']}" for row in rows
        ], fontsize=8)
        axis.set(ylabel="Mean cosine distance", title="Historical values: reference comparison only")
        axis.legend()
        note = "Original cohort is unverified; Mode A grouping by PPN response is estimated"
        if not reference["model_and_layers_compatible"]:
            note += "\nModel or layer settings differ from the historical experiment"
        fig.supxlabel(note, fontsize=9)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "reference_comparison.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
    return outputs
