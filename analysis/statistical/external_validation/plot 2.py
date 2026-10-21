from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

RESULTS_DIR = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis/statistical/external_validation/"
    "friedman_posthoc_results"
)

FIGURE_DIR = RESULTS_DIR / "comparison_matrices"

DATASETS = ["HCC", "LiTS", "TCIA", "WORC", "CAIRO5"]

DATASET_LABELS = {
    "HCC": "HCC-TACE-Seg",
    "LiTS": "LiTS",
    "TCIA": "TCIA-CRLM",
    "WORC": "WORC-CRLM",
    "CAIRO5": "CAIRO5",
}

METRICS = ["dice", "precision", "recall", "detectability"]

METRIC_LABELS = {
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Sensitivity",
    "detectability": "Detectability",
}


BASELINES = ["TSLL", "TSLT", "AtlasNet", "COALA"]

# Cell colours indicate significance and direction.
COLOURS = {
    "reduction": "#E6A09B",
    "not_significant": "#EEEEEE",
    "improvement": "#9DCEAE",
}

CMAP = ListedColormap([
    COLOURS["reduction"],
    COLOURS["not_significant"],
    COLOURS["improvement"],
])

SHOW_FIGURES = True

sns.set_theme(style="white", font_scale=1.05)


# -------------------------------------------------------------------
# Load and validate statistical results
# -------------------------------------------------------------------

def load_results(dataset):
    path = RESULTS_DIR / f"posthoc_wilcoxon_{dataset}.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    results = pd.read_csv(path)

    required = {
        "model_a",
        "model_b",
        "metric",
        "mean_difference_a_minus_b",
        "significant_after_holm",
    }

    missing = required.difference(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    keys = ["model_a", "model_b", "metric"]

    if results.duplicated(keys).any():
        raise ValueError(
            f"Duplicate comparisons found in {path}"
        )

    return results


def parse_significance(value):
    """Read boolean flags consistently from CSV files."""
    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "1.0"}:
        return True

    if text in {"false", "0", "0.0"}:
        return False

    raise ValueError(
        f"Unrecognised significance value: {value!r}"
    )


# -------------------------------------------------------------------
# Prepare a matrix cell
# -------------------------------------------------------------------

def get_cell(results, model_a, model_b, metric):
    # Find the pair in either direction.
    selected = results[
        (results["metric"] == metric)
        & (
            (
                (results["model_a"] == model_a)
                & (results["model_b"] == model_b)
            )
            | (
                (results["model_a"] == model_b)
                & (results["model_b"] == model_a)
            )
        )
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Expected one result for {model_a} versus "
            f"{model_b}, metric {metric}; "
            f"found {len(selected)}."
        )

    row = selected.iloc[0]

    # Reverse the sign when the CSV stores the opposite direction.
    direction = 1 if row["model_a"] == model_a else -1

    # Keep differences on the original metric scale.
    difference = (
        direction
        * float(row["mean_difference_a_minus_b"])
    )

    if not np.isfinite(difference):
        return 0, "NA"

    significant = parse_significance(
        row["significant_after_holm"]
    )

    if significant and difference > 0:
        colour_code = 1
    elif significant and difference < 0:
        colour_code = -1
    else:
        colour_code = 0

    annotation = f"{difference:+.4f}"

    if significant:
        annotation += "*"

    return colour_code, annotation


# -------------------------------------------------------------------
# Draw and save a heatmap
# -------------------------------------------------------------------

def draw_heatmap(codes, annotations, title, filename, figsize):
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        codes,
        annot=annotations,
        fmt="",
        cmap=CMAP,
        vmin=-1.5,
        vmax=1.5,
        cbar=False,
        linewidths=2,
        linecolor="white",
        annot_kws={
            "fontsize": 13,
            "color": "#222222",
        },
        ax=ax,
    )

    ax.set_title(
        title,
        fontsize=16,
        weight="bold",
        pad=16,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(axis="both", length=0)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        fontsize=11,
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=11,
    )

    legend = [
        Patch(
            facecolor=COLOURS["improvement"],
            label="Significant increase",
        ),
        Patch(
            facecolor=COLOURS["reduction"],
            label="Significant decrease",
        ),
        Patch(
            facecolor=COLOURS["not_significant"],
            label="Not significant",
        ),
    ]

    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.15),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    fig.text(
        0.5,
        0.025,
        "Voxel level: Dice, Precision, Sensitivity. "
        "Lesion level: Detectability (IoU ≥ 0.20).\n"
        "Values: mean paired differences (named model − comparison model).\n"
        "* Holm-adjusted Wilcoxon p < 0.05.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.27, 1, 1])

    png_path = FIGURE_DIR / f"{filename}.png"
    svg_path = FIGURE_DIR / f"{filename}.svg"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


# -------------------------------------------------------------------
# DP versus SP
# -------------------------------------------------------------------

def plot_dp_vs_sp(results, dataset):
    columns = [METRIC_LABELS[m] for m in METRICS]

    codes = pd.DataFrame(
        0,
        index=["DP − SP"],
        columns=columns,
    )

    annotations = pd.DataFrame(
        "",
        index=codes.index,
        columns=codes.columns,
    )

    for metric in METRICS:
        column = METRIC_LABELS[metric]

        code, annotation = get_cell(
            results,
            model_a="DP",
            model_b="SP",
            metric=metric,
        )

        codes.loc["DP − SP", column] = code
        annotations.loc["DP − SP", column] = annotation

    draw_heatmap(
        codes=codes,
        annotations=annotations,
        title=f"{DATASET_LABELS.get(dataset, dataset)}: Single-Phase versus Dual-Phase",        filename=f"{dataset}_DP_vs_SP_matrix",
        figsize=(10, 3.8),
    )


# -------------------------------------------------------------------
# SP or DP versus benchmarks
# -------------------------------------------------------------------

def plot_model_vs_benchmarks(results, dataset, model):
    rows = [METRIC_LABELS[m] for m in METRICS]

    codes = pd.DataFrame(
        0,
        index=rows,
        columns=BASELINES,
    )

    annotations = pd.DataFrame(
        "",
        index=codes.index,
        columns=codes.columns,
    )

    for metric in METRICS:
        row_label = METRIC_LABELS[metric]

        for baseline in BASELINES:
            code, annotation = get_cell(
                results,
                model_a=model,
                model_b=baseline,
                metric=metric,
            )

            codes.loc[row_label, baseline] = code
            annotations.loc[row_label, baseline] = annotation

    draw_heatmap(
        codes=codes,
        annotations=annotations,
        title=f"{DATASET_LABELS.get(dataset, dataset)}: {model} versus Benchmark Models",        filename=f"{dataset}_{model}_vs_benchmarks_matrix",
        figsize=(9, 5.5),
    )


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"\nCreating matrices for {dataset}")

        results = load_results(dataset)

        plot_dp_vs_sp(results, dataset)

        for model in ["SP", "DP"]:
            plot_model_vs_benchmarks(
                results,
                dataset,
                model,
            )

    print(f"\nAll figures saved to:\n{FIGURE_DIR}")


if __name__ == "__main__":
    main()