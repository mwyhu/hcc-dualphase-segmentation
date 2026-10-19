from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

BASE_DIR = Path(
    "/Users/michellehu/Desktop/"
    "hcc-dualphase-segmentation/analysis/statistical/internal_testset/"
)

RESULTS_DIR = Path(
    "/Users/michellehu/Desktop/"
    "hcc-dualphase-segmentation/benchmark/SP_vs_DP/bootstrap"
)

RESULTS_FILE = RESULTS_DIR / "paired_ttest_results.csv"
FIGURE_DIR = RESULTS_DIR / "comparison_matrices"

METRICS = ["dice", "precision", "recall", "detectability"]

METRIC_LABELS = {
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Sensitivity",
    "detectability": "Detectability",
}

P_VALUE_COLUMN = "bootstrap_holm_adjusted_p_value"

COLOURS = {
    "reduction": "#E6A09B",
    "not_significant": "#EEEEEE",
    "improvement": "#9DCEAE",
    "unavailable": "#FFFFFF",
}

CMAP = ListedColormap([
    COLOURS["reduction"],
    COLOURS["not_significant"],
    COLOURS["improvement"],
])

NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], CMAP.N)

SHOW_FIGURES = True
ALPHA = 0.05

# Add extensions here if needed.
SAVE_FORMATS = ["png", "pdf"]

sns.set_theme(style="white", font_scale=1.05)


# -------------------------------------------------------------------
# Load and validate results
# -------------------------------------------------------------------

def load_results():
    if not RESULTS_FILE.is_file():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            "Run the internal test-set bootstrap analysis first."
        )

    results = pd.read_csv(RESULTS_FILE)

    required = {
        "metric",
        "mean_difference_dual_minus_single",
        P_VALUE_COLUMN,
        "n_paired_cases",
    }

    missing = required.difference(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    results = results[
        results["metric"].isin(METRICS)
    ].copy()

    if results["metric"].duplicated().any():
        raise ValueError("Duplicate metric results found.")

    missing_metrics = set(METRICS) - set(results["metric"])

    if missing_metrics:
        raise ValueError(
            f"Missing metrics: {sorted(missing_metrics)}"
        )

    numeric_columns = [
        "mean_difference_dual_minus_single",
        P_VALUE_COLUMN,
        "n_paired_cases",
    ]

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="raise",
        )

    p_values = results[P_VALUE_COLUMN]

    invalid_p = (
        p_values.notna()
        & (
            ~np.isfinite(p_values)
            | (p_values < 0)
            | (p_values > 1)
        )
    )

    if invalid_p.any():
        raise ValueError("Invalid Holm-adjusted bootstrap p-values.")

    counts = results["n_paired_cases"]

    if (
        (~np.isfinite(counts))
        | (counts < 0)
        | (counts != np.floor(counts))
    ).any():
        raise ValueError("Invalid paired case counts.")

    return results.set_index("metric")


# -------------------------------------------------------------------
# Prepare matrix
# -------------------------------------------------------------------

def prepare_matrix(results):
    codes = np.zeros((1, len(METRICS)), dtype=float)
    annotations = np.empty((1, len(METRICS)), dtype=object)
    unavailable = np.zeros((1, len(METRICS)), dtype=bool)

    for column, metric in enumerate(METRICS):
        row = results.loc[metric]

        difference = float(
            row["mean_difference_dual_minus_single"]
        )
        adjusted_p = float(row[P_VALUE_COLUMN])

        if not np.isfinite(difference):
            unavailable[0, column] = True
            annotations[0, column] = "NA"
            continue

        if not np.isfinite(adjusted_p):
            unavailable[0, column] = True
            annotations[0, column] = (
                f"{difference:+.4f}\nTest unavailable"
            )
            continue

        significant = adjusted_p < ALPHA

        if significant and difference > 0:
            codes[0, column] = 1
        elif significant and difference < 0:
            codes[0, column] = -1
        else:
            codes[0, column] = 0

        marker = "*" if significant else ""
        annotations[0, column] = f"{difference:+.4f}{marker}"

    return codes, annotations, unavailable


# -------------------------------------------------------------------
# Draw and save
# -------------------------------------------------------------------

def plot_internal_comparison(results):
    codes, annotations, unavailable = prepare_matrix(results)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_facecolor(COLOURS["unavailable"])

    sns.heatmap(
        codes,
        mask=unavailable,
        annot=False,
        cmap=CMAP,
        norm=NORM,
        cbar=False,
        linewidths=2,
        linecolor="white",
        xticklabels=[METRIC_LABELS[m] for m in METRICS],
        yticklabels=["DP − SP"],
        ax=ax,
    )

    # Manual annotations also display unavailable results.
    for column in range(len(METRICS)):
        ax.text(
            column + 0.5,
            0.5,
            annotations[0, column],
            ha="center",
            va="center",
            fontsize=13,
            color="#222222",
        )

    ax.set_title(
        "Internal test set: Single-Phase versus Dual-Phase",
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

    if unavailable.any():
        legend.append(
            Patch(
                facecolor=COLOURS["unavailable"],
                edgecolor="#999999",
                label="Test unavailable",
            )
        )

    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.15),
        ncol=len(legend),
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

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for extension in SAVE_FORMATS:
        path = (
            FIGURE_DIR
            / f"Internal_DP_vs_SP_bootstrap_matrix.{extension}"
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Saved: {path}")

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


def main():
    results = load_results()
    plot_internal_comparison(results)


if __name__ == "__main__":
    main()