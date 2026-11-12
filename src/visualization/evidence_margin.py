import torch

from src.visualization.probe import GRID, SERIES, TEXT_SECONDARY, plt, save


TERMS = (("z_N", "z_N (negative image)"), ("z_P_mean", "z_P mean (positive images)"))


def residual(x, values):
    return values - x @ torch.linalg.lstsq(x, values.unsqueeze(-1)).solution.squeeze(-1)


def plot_evidence_margin(report, tables, output_dir):
    """왼쪽: 레이어별 표준화 계수(주 모형) / 오른쪽: 주 레이어 added-variable plot (z_P_mean, z_N·클래스·위치 통제)."""
    analysis = report["analysis"]
    sample, layer, model = analysis["primary_sample"], analysis["primary_layer"], analysis["primary_model"]
    rows = [row for row in report["coefficients"] if row["sample"] == sample and row["model"] == model]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    axis = axes[0]
    for offset, color, (term, label) in zip((-0.9, 0.9), SERIES, TERMS):
        values = sorted((row for row in rows if row["term"] == term), key=lambda row: row["layer"])
        scale = [row["sd_term"] for row in values]
        centers = [row["estimate_per_sd"] for row in values]
        low = [c - row["ci_low"] * s for c, row, s in zip(centers, values, scale)]
        high = [row["ci_high"] * s - c for c, row, s in zip(centers, values, scale)]
        axis.errorbar([row["layer"] + offset for row in values], centers, yerr=[low, high], marker="o", markersize=6,
                      linewidth=0, elinewidth=2, capsize=3, color=color, label=label)
    axis.axhline(0, color=GRID, linewidth=1.5)
    axis.set_xticks(report["layers"])
    axis.set_xlabel("Decoder layer")
    axis.set_ylabel("Margin change per 1 SD of evidence (logit units)")
    axis.set_title(f"{model}: class FE + N position + z_N + z_P mean (n={rows[0]['n']})", fontsize=9)
    bottom, top = axis.get_ylim()
    axis.set_ylim(bottom - 0.25 * (top - bottom), top)
    axis.legend(fontsize=8, loc="lower center", ncol=2)

    axis = axes[1]
    items = tables[sample]
    terms = {name: torch.tensor([item["terms"][layer][name] for item in items], dtype=torch.float64)
             for name in ("z_N", "z_P_mean")}
    classes = sorted({item["class"] for item in items})
    columns = [torch.tensor([item["class"] == name for item in items], dtype=torch.float64) for name in classes]
    position = torch.tensor([item["labels"].index("N") + 1 for item in items])
    columns += [(position == value).double() for value in (2, 3)] + [terms["z_N"]]
    x = torch.stack(columns, dim=1)
    margin = residual(x, torch.tensor([item["margin"] for item in items], dtype=torch.float64))
    positive = residual(x, terms["z_P_mean"])
    yes = torch.tensor([item["prediction"] == "Yes" for item in items])
    for mask, color, label in ((~yes, SERIES[0], "VLM Correct-No"), (yes, SERIES[1], "VLM False-Yes")):
        axis.scatter(positive[mask], margin[mask], s=10, alpha=0.5, color=color, linewidths=0,
                     label=f"{label} (n={int(mask.sum())})")
    slope = float((positive @ margin) / (positive @ positive))
    span = torch.linspace(float(positive.min()), float(positive.max()), 2)
    axis.plot(span, slope * span, color=TEXT_SECONDARY, linewidth=1.5, label=f"slope {slope:.3f}")
    axis.axhline(0, color=GRID, linewidth=1)
    axis.set_xlabel("z_P mean, residual")
    axis.set_ylabel("Yes - No margin, residual")
    axis.set_title(f"Added-variable plot, L{layer} (controls: class, N position, z_N)", fontsize=9)
    axis.legend(fontsize=8, loc="upper left")
    fig.supxlabel(f"status: {report['status']} | STEP 4 one-N inputs (GT No); 95% cluster bootstrap CI over image IDs; "
                  "association only, same-N contrast not identified in STEP 4", fontsize=8, color=TEXT_SECONDARY)
    return save(fig, output_dir / "evidence_margin.png")
