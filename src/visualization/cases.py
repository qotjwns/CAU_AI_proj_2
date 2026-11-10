"""Group and render the existing image cases by category and pair."""

import matplotlib.pyplot as plt

from src.common.data import normalize_category
from src.common.paths import project_path


def group_cases_by_class(cases, category=None):
    classes = {}
    for case in cases:
        name = case["class"]
        if category and normalize_category(name) != normalize_category(category):
            continue
        pairs = classes.setdefault(name, {})
        pairs.setdefault(case["pair_id"], []).append(case)
    return classes


def plot_pair(cases, output):
    first = cases[0]
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 3 * len(cases)), squeeze=False, constrained_layout=True)
    fig.suptitle(f"Mode {first['mode']} | {first['class']}\n{first['pair_id']}", fontsize=11)
    for row, case in enumerate(cases):
        image_dir = project_path(case["image_dir"])
        for column, (name, label) in enumerate(zip(case["image_order"], case["image_labels"])):
            axis = axes[row, column]
            axis.imshow(plt.imread(image_dir / name))
            title = f"Image {column + 1}: {label} | {name}"
            if column == 0:
                title = f"{case['composition']} | GT: {case['gt']}\n{title}"
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    fig.savefig(output, dpi=150)
    plt.close(fig)
