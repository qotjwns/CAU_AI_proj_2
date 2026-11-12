from src.visualization.probe import GRID, SERIES, TEXT_SECONDARY, plt, save


SAMPLE = "STEP 4 one-N inputs (PPN/PNP/NPP, GT No), all unique inputs (sign-inconsistent included)"


def footer(fig, report, extra):
    fig.supxlabel(f"status: {report['status']} | sample: {SAMPLE} | {extra}", fontsize=7.5, color=TEXT_SECONDARY)


def error_bar(axis, x, row, color, label=None, low="ci_low", high="ci_high", value="estimate", marker="o"):
    axis.errorbar([x], [row[value]], yerr=[[row[value] - row[low]], [row[high] - row[value]]], marker=marker,
                  markersize=7, linewidth=0, elinewidth=2, capsize=3, color=color, label=label)


def plot_coefficients(report, results, output_dir):
    """A1(β_N, β_P, 차이)와 A3(위치 × 역할 계수), 주 레이어."""
    layer = report["primary_layer"]
    rows = results["all"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axis = axes[0]
    terms = (("beta_N", "beta_N"), ("beta_P", "beta_P"), ("beta_P_minus_beta_N", "beta_P - beta_N"))
    for x, (term, _) in enumerate(terms):
        row = next(r for r in rows["a1"] if r["layer"] == layer and r["term"] == term)
        error_bar(axis, x, row, SERIES[min(x, 2)])
    axis.set_xticks(range(3), [label for _, label in terms])
    axis.axhline(0, color=GRID, linewidth=1.5)
    axis.set_ylabel("Margin change per 1 class-SD of z_i logit")
    axis.set_title(f"A1: class FE + N position + z_N + z_P mean, L{layer} (n={rows['n_inputs']})", fontsize=9)

    axis = axes[1]
    for offset, color, role in ((-0.12, SERIES[1], "P"), (0.12, SERIES[0], "N")):
        for position in (1, 2, 3):
            row = next(r for r in rows["a3"] if r["layer"] == layer and r["term"] == f"{role}_pos{position}")
            error_bar(axis, position + offset, row, color, label=f"{role} slot" if position == 1 else None)
    axis.set_xticks((1, 2, 3), ["position 1", "position 2", "position 3"])
    axis.axhline(0, color=GRID, linewidth=1.5)
    axis.set_ylabel("Margin change per 1 class-SD of z_i logit")
    axis.set_title(f"A3: position x role coefficients, L{layer} (class FE + N position FE)", fontsize=9)
    bottom, top = axis.get_ylim()
    axis.set_ylim(bottom, top + 0.2 * (top - bottom))
    axis.legend(fontsize=8, loc="upper center", ncol=2)
    footer(fig, report, "95% image-ID cluster bootstrap CI")
    return save(fig, output_dir / "a1_a3_coefficients.png")


def plot_models(report, results, output_dir):
    """A2 held-out 설명력(margin R², False-Yes log loss)과 STEP 4 신뢰도 곡선(원 확률 vs 보정)."""
    layer = report["primary_layer"]
    rows = [r for r in results["all"]["a2"] if r["layer"] == layer]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    names = [r["model"] for r in rows]
    for axis, metric, label in ((axes[0], "margin_r2", "Held-out R2 (margin)"),
                                (axes[1], "fy_log_loss", "Held-out log loss (False-Yes, lower is better)")):
        for x, row in enumerate(rows):
            error_bar(axis, x, row, SERIES[0] if row["model"] != "minimum" else SERIES[1],
                      low=f"{metric}_ci_low", high=f"{metric}_ci_high", value=metric)
        axis.set_xticks(range(len(rows)), [name.replace("soft_and_", "soft-AND\n") for name in names], fontsize=8)
        axis.set_ylabel(label)
        axis.set_title(f"A2: {report['settings']['a2']['folds']}-fold pair-grouped CV, L{layer} "
                       f"(n={rows[0]['n']})", fontsize=9)
    axis = axes[2]
    calibration = {r["source"]: r["ece"] for r in results["all"]["calibration"] if r["layer"] == layer}
    for color, source, label in ((SERIES[0], "step4_one_negative_raw", "frozen probe, raw"),
                                 (SERIES[1], "step4_one_negative_calibrated", "after STEP 3 Platt")):
        curve = [r for r in results["all"]["reliability"] if r["layer"] == layer and r["source"] == source]
        axis.plot([r["mean_predicted"] for r in curve], [r["observed_P_rate"] for r in curve], marker="o",
                  color=color, label=f"{label} (ECE {calibration[source]:.3f})")
    axis.plot([0, 1], [0, 1], color=GRID, linewidth=1.5)
    axis.set_xlabel("Mean predicted P probability")
    axis.set_ylabel("Observed P rate")
    axis.set_title(f"Probe reliability on STEP 4 one-N slots, L{layer}", fontsize=9)
    axis.legend(fontsize=8)
    footer(fig, report, "all models share class FE + N position FE; 95% image-ID cluster bootstrap CI on OOF predictions")
    return save(fig, output_dir / "a2_models.png")


def plot_terciles(report, results, output_dir):
    """A4(전체)·A5(N 확률 기준 한정)에서 z_P 상위·하위 1/3의 False-Yes 비율."""
    rows = [r for r in results["all"]["a4"] + results["all"]["a5"] if r["outcome"] == "false_yes"]
    fig, axis = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    labels = []
    for x, row in enumerate(rows):
        axis.bar(x - 0.18, row["bottom_tercile"], width=0.36, color=SERIES[0], label="bottom z_P third" if not x else None)
        axis.bar(x + 0.18, row["top_tercile"], width=0.36, color=SERIES[1], label="top z_P third" if not x else None)
        ci = f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
        axis.text(x, max(row["bottom_tercile"], row["top_tercile"]) + 0.02, f"diff {row['difference']:+.3f}\n{ci}",
                  ha="center", fontsize=7.5)
        name = "all one-N" if row["analysis"] == "A4" else f"p_N < {row['threshold']}"
        labels.append(f"{name}\n(n={row['n_inputs']})")
    axis.set_xticks(range(len(rows)), labels, fontsize=8)
    axis.set_ylabel("VLM False-Yes rate")
    axis.set_ylim(0, max(max(r["top_tercile"], r["bottom_tercile"]) for r in rows) + 0.12)
    axis.set_title(f"A4/A5: False-Yes by mean P score tercile within class x N position, L{report['primary_layer']}",
                   fontsize=9)
    axis.legend(fontsize=8, loc="upper right")
    footer(fig, report, "diff = stratum-size weighted top - bottom; 95% image-ID cluster bootstrap CI")
    return save(fig, output_dir / "a4_a5_terciles.png")
