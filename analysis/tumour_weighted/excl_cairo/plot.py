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
    "hcc-dualphase-segmentation/analysis/tumour_weighted/excl_cairo"
)

GROUPS = {
    "CRLM": BASE_DIR / "crlm_sp_vs_dp_results",
    "HCC": BASE_DIR / "hcc_sp_vs_dp_results",
    "Mixed": BASE_DIR / "mixed_sp_vs_dp_results",
}

METRICS = ["dice", "precision", "recall", "detectability"]

METRIC_LABELS = {
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Sensitivity",
    "detectability": "Detectability",
}

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

sns.set_theme(style="white", font_scale=1.05)


# -------------------------------------------------------------------
# Load and validate results
# -------------------------------------------------------------------

def load_results(results_file, group):
    results = pd.read_csv(results_file)

    required = {
        "metric",
        "mean_difference_dual_minus_single",
        "holm_adjusted_p_value",
        "n_paired_cases",
    }

    missing = required.difference(results.columns)

    if missing:
        raise ValueError(
            f"{group}: missing columns: {sorted(missing)}"
        )

    # Check the group label when present.
    if "tumour_type" in results.columns:
        observed_groups = set(
            results["tumour_type"].dropna().astype(str)
        )

        if observed_groups != {group}:
            raise ValueError(
                f"{results_file}: expected tumour_type "
                f"{group!r}, found {sorted(observed_groups)}"
            )

    results = results.loc[
        results["metric"].isin(METRICS)
    ].copy()

    if results["metric"].duplicated().any():
        raise ValueError(
            f"{group}: duplicate metric results found."
        )

    missing_metrics = set(METRICS) - set(results["metric"])

    if missing_metrics:
        raise ValueError(
            f"{group}: missing metrics: "
            f"{sorted(missing_metrics)}"
        )

    numeric_columns = [
        "mean_difference_dual_minus_single",
        "holm_adjusted_p_value",
        "n_paired_cases",
    ]

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="raise",
        )

    p_values = results["holm_adjusted_p_value"]

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
            f"{group}: invalid Holm-adjusted p-values."
        )

    differences = results[
        "mean_difference_dual_minus_single"
    ]

    invalid_difference = (
        differences.notna()
        & (
            ~np.isfinite(differences)
            | (differences < -1)
            | (differences > 1)
        )
    )

    if invalid_difference.any():
        raise ValueError(
            f"{group}: invalid mean differences."
        )

    counts = results["n_paired_cases"]

    if (
        (~np.isfinite(counts))
        | (counts < 0)
        | (counts != np.floor(counts))
    ).any():
        raise ValueError(
            f"{group}: invalid paired case counts."
        )

    return results.set_index("metric").loc[METRICS]


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
        adjusted_p = float(row["holm_adjusted_p_value"])

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
        annotations[0, column] = (
            f"{difference:+.4f}{marker}"
        )

    return codes, annotations, unavailable


# -------------------------------------------------------------------
# Read detectability threshold for the figure note
# -------------------------------------------------------------------

def detectability_note(results):
    if "iou_threshold" not in results.columns:
        return "Lesion level: Detectability."

    value = pd.to_numeric(
        results.loc["detectability", "iou_threshold"],
        errors="coerce",
    )

    if not np.isfinite(value):
        return "Lesion level: Detectability."

    return (
        f"Lesion level: Detectability (IoU ≥ {value:.2f})."
    )


# -------------------------------------------------------------------
# Draw and save
# -------------------------------------------------------------------

def plot_group_comparison(results, group, results_dir):
    codes, annotations, unavailable = prepare_matrix(results)

    figure_dir = results_dir / "comparison_matrices"

    fig, ax = plt.subplots(figsize=(10, 4.2))
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
        xticklabels=[
            METRIC_LABELS[metric]
            for metric in METRICS
        ],
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
        f"{group}: Single-Phase versus Dual-Phase",
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
        bbox_to_anchor=(0.5, 0.18),
        ncol=len(legend),
        frameon=False,
        fontsize=10,
    )

    fig.text(
        0.5,
        0.025,
        "Voxel level: Dice, Precision, Sensitivity. "
        f"{detectability_note(results)}\n"
        "Values: equal-dataset-weighted mean paired "
        "differences (DL - SP).\n"
        f"* Holm-adjusted p < {ALPHA:.2f}; "
        "t-test with Satterthwaite degrees of freedom.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.31, 1, 1])

    figure_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{group}_DP_vs_SP_matrix"

    for extension in ("png", "svg", "pdf"):
        path = figure_dir / f"{filename}.{extension}"

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


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    plotted_groups = []

    for group, results_dir in GROUPS.items():
        results_file = (
            results_dir / "paired_ttest_results.csv"
        )

        if not results_file.is_file():
            print(
                f"\nSkipping {group}: no results file found.\n"
                f"Expected: {results_file}\n"
                "The group may still be a placeholder."
            )
            continue

        results = load_results(results_file, group)

        plot_group_comparison(
            results,
            group,
            results_dir,
        )

        plotted_groups.append(group)

    if not plotted_groups:
        print(
            "\nNo figures created. Run the grouped "
            "SP-versus-DP statistical analysis first."
        )


if __name__ == "__main__":
    main()