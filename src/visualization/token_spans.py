import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.common.data import normalize_category
from src.common.paths import project_path
from src.core.metrics import COMPOSITIONS


def group_span_records(records, category=None):
    groups = {}
    seen = set()
    for record in records:
        case = record["case"]
        if category and normalize_category(case["class"]) != normalize_category(category):
            continue
        key = (case["mode"], case["case_id"])
        if key in seen:
            raise ValueError(f"Duplicate extracted case: {key}")
        seen.add(key)
        group_key = (case["mode"], case["class"], case["pair_id"])
        groups.setdefault(group_key, []).append(record)

    for (mode, _, _), group in groups.items():
        order = COMPOSITIONS[mode]
        group.sort(key=lambda record: order.index("".join(record["case"]["image_labels"])))
    return list(groups.values())


def validate_spans(record):
    case = record["case"]
    spans = record["image_spans"]
    if not (len(spans) == len(case["image_ids"]) == len(case["image_order"])
            == len(case["image_labels"]) == 3):
        raise ValueError(f"Expected three images and spans: {case['case_id']}")
    previous_end = 0
    for index, span in enumerate(spans):
        if (span["image_index"] != index or span["image_id"] != case["image_ids"][index]
                or not previous_end <= span["start"] < span["end"] <= record["query_index"]
                or span["token_count"] != span["end"] - span["start"]):
            raise ValueError(f"Invalid saved span for image {index + 1}: {case['case_id']}")
        previous_end = span["end"]
    if sum(span["token_count"] for span in spans) != record["image_token_count"]:
        raise ValueError(f"Image token count does not match saved spans: {case['case_id']}")


def plot_token_spans(records, output):
    for record in records:
        validate_spans(record)
    first = records[0]["case"]
    image_ids = list(dict.fromkeys(
        image_id for record in records for image_id in record["case"]["image_ids"]
    ))
    palette = plt.get_cmap("tab10")
    colors = {image_id: palette(index % 10) for index, image_id in enumerate(image_ids)}
    first_positions = {image_id: index + 1 for index, image_id in enumerate(first["image_ids"])}
    max_length = max(record["query_index"] + 1 for record in records)

    fig = plt.figure(figsize=(16, 5.5 * len(records)), constrained_layout=True)
    grid = fig.add_gridspec(2 * len(records), 3, height_ratios=[3.5, 0.85] * len(records))
    expected = len(COMPOSITIONS[first["mode"]])
    fig.suptitle(
        f"Mode {first['mode']} | {first['class']} | {first['pair_id']}\n"
        f"{len(records)}/{expected} saved cases | colors track image ID",
        fontsize=13,
    )

    for row, record in enumerate(records):
        case = record["case"]
        image_dir = project_path(case["image_dir"])
        for column, span in enumerate(record["image_spans"]):
            image_id = span["image_id"]
            color = colors[image_id]
            name = case["image_order"][column]
            label = case["image_labels"][column]
            reference = (
                f"first input: image {first_positions[image_id]}"
                if image_id in first_positions else "not in first input"
            )
            axis = fig.add_subplot(grid[2 * row, column])
            axis.imshow(plt.imread(image_dir / name))
            axis.set_title(
                f"input {row + 1}: image {column + 1} ({reference})\n"
                f"{name} | {label}\n"
                f"[{span['start']}, {span['end']}) / {span['token_count']} tokens",
                fontsize=10, color=color,
            )
            axis.axis("off")

        axis = fig.add_subplot(grid[2 * row + 1, :])
        axis.axvspan(0, record["query_index"] + 1, color="#dddddd", zorder=0)
        for index, span in enumerate(record["image_spans"]):
            axis.barh(
                0.5, span["token_count"], left=span["start"], height=0.9,
                color=colors[span["image_id"]], edgecolor="white", linewidth=1,
            )
            label = (
                f"image {index + 1}\n{span['start']}:{span['end']}"
                if span["token_count"] / max_length >= 0.045 else str(index + 1)
            )
            axis.text(
                (span["start"] + span["end"]) / 2, 0.5, label,
                ha="center", va="center", color="white", fontsize=8,
            )
        axis.axvline(record["query_index"], color="#333333", linestyle="--", linewidth=1)
        axis.set_xlim(0, max_length)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
        composition = "".join(case["image_labels"])
        prediction = record["prediction"] or "Invalid"
        axis.set_title(
            f"Input {row + 1}: {composition} | GT: {case['gt']} | prediction: {prediction} "
            f"| h_q: {record['query_index']}\n{case['case_id']}",
            loc="left", fontsize=9,
        )
        axis.set_xlabel(
            "Token index (zero-based, end exclusive); gray = non-image tokens; dashed line = h_q",
            fontsize=9,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
