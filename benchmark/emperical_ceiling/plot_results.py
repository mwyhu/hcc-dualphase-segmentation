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

RESULTS_DIR = Path(
    "/Users/michellehu/Desktop/"
    "hcc-dualphase-segmentation/benchmark/emperical_ceiling/bootstrap"
)

RESULTS_FILE = RESULTS_DIR / "paired_ttest_results.csv"
FIGURE_DIR = RESULTS_DIR / "comparison_matrices"

COMPARISONS = [
    "SP ceiling versus SP",
    "DP ceiling versus DP",
]

COMPARISON_LABELS = {
    "SP ceiling versus SP": "SP ceiling − SP",
    "DP ceiling versus DP": "DP ceiling − DP",
}

METRICS = ["dice", "precision", "recall", "detectability"]

METRIC_LABELS = {
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Sensitivity",
    "detectability": "Detectability",
}

DIFFERENCE_COLUMN = "mean_difference_ceiling_minus_original"
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

SAVE_FORMATS = ["png", "pdf"]

sns.set_theme(style="white", font_scale=1.05)


# -------------------------------------------------------------------
# Load and validate results
# -------------------------------------------------------------------

def load_results():
    if not RESULTS_FILE.is_file():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            "Run the ceiling test-set bootstrap analysis first."
        )

    results = pd.read_csv(RESULTS_FILE)

    required = {
        "comparison",
        "metric",
        DIFFERENCE_COLUMN,
        P_VALUE_COLUMN,
        "n_paired_cases",
    }

    missing = required.difference(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    results = results[
        results["comparison"].isin(COMPARISONS)
        & results["metric"].isin(METRICS)
    ].copy()

    if results.duplicated(["comparison", "metric"]).any():
        raise ValueError(
            "Duplicate comparison/metric results found."
        )

    expected = {
        (comparison, metric)
        for comparison in COMPARISONS
        for metric in METRICS
    }

    observed = set(
        results[["comparison", "metric"]].itertuples(
            index=False,
            name=None,
        )
    )

    missing_results = expected - observed

    if missing_results:
        raise ValueError(
            f"Missing comparison/metric results: "
            f"{sorted(missing_results)}"
        )

    numeric_columns = [
        DIFFERENCE_COLUMN,
        P_VALUE_COLUMN,
        "n_paired_cases",
    ]

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="raise",
        )

    differences = results[DIFFERENCE_COLUMN]

    invalid_difference = (
        differences.notna()
        & (
            ~np.isfinite(differences)
            | (differences < -1)
            | (differences > 1)
        )
    )

    if invalid_difference.any():
        raise ValueError("Invalid mean paired differences.")

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
        raise ValueError(
            "Invalid Holm-adjusted bootstrap p-values."
        )

    counts = results["n_paired_cases"]

    if (
        (~np.isfinite(counts))
        | (counts < 0)
        | (counts != np.floor(counts))
    ).any():
        raise ValueError("Invalid paired case counts.")

    return results.set_index(["comparison", "metric"])


# -------------------------------------------------------------------
# Prepare matrix
# -------------------------------------------------------------------

def prepare_matrix(results):
    shape = (len(COMPARISONS), len(METRICS))

    codes = np.zeros(shape, dtype=float)
    annotations = np.empty(shape, dtype=object)
    unavailable = np.zeros(shape, dtype=bool)

    for row_index, comparison in enumerate(COMPARISONS):
        for column, metric in enumerate(METRICS):
            result = results.loc[(comparison, metric)]

            difference = float(result[DIFFERENCE_COLUMN])
            adjusted_p = float(result[P_VALUE_COLUMN])

            if not np.isfinite(difference):
                unavailable[row_index, column] = True
                annotations[row_index, column] = "NA"
                continue

            if not np.isfinite(adjusted_p):
                unavailable[row_index, column] = True
                annotations[row_index, column] = (
                    f"{difference:+.4f}\nTest unavailable"
                )
                continue

            significant = adjusted_p < ALPHA

            if significant and difference > 0:
                codes[row_index, column] = 1
            elif significant and difference < 0:
                codes[row_index, column] = -1
            else:
                codes[row_index, column] = 0

            marker = "*" if significant else ""

            annotations[row_index, column] = (
                f"{difference:+.4f}{marker}"
            )

    return codes, annotations, unavailable


# -------------------------------------------------------------------
# Draw and save
# -------------------------------------------------------------------

def plot_ceiling_comparison(results):
    codes, annotations, unavailable = prepare_matrix(results)

    fig, ax = plt.subplots(figsize=(11, 5))
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
        yticklabels=[
            COMPARISON_LABELS[c] for c in COMPARISONS
        ],
        ax=ax,
    )

    # Manual annotations also display unavailable results.
    for row_index in range(len(COMPARISONS)):
        for column in range(len(METRICS)):
            ax.text(
                column + 0.5,
                row_index + 0.5,
                annotations[row_index, column],
                ha="center",
                va="center",
                fontsize=13,
                color="#222222",
            )

    ax.set_title(
        "Ceiling test set: Primary versus Empirical Ceiling models",
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
        bbox_to_anchor=(0.5, 0.13),
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

    fig.tight_layout(rect=[0, 0.25, 1, 1])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for extension in SAVE_FORMATS:
        path = (
            FIGURE_DIR
            / f"Ceiling_vs_original_bootstrap_matrix.{extension}"
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
    plot_ceiling_comparison(results)


if __name__ == "__main__":
    main()