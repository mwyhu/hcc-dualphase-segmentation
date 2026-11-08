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
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis/statistical/external_validation/excl_cairo/"
    "friedman_pairwise_valid_pairs_results"
)

FIGURE_DIR = RESULTS_DIR / "comparison_matrices"

DATASETS = ["HCC", "LiTS", "TCIA", "WORC"]

DATASET_LABELS = {
    "HCC": "HCC-TACE-Seg",
    "LiTS": "LiTS",
    "TCIA": "TCIA-CRLM",
    "WORC": "WORC-CRLM",
}

METRICS = ["dice", "precision", "recall", "detectability"]

METRIC_LABELS = {
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Sensitivity",
    "detectability": "Detectability",
}

BASELINES = ["TSLL", "TSLT", "AtlasNet", "COALA"]

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

NORM = BoundaryNorm(
    [-1.5, -0.5, 0.5, 1.5],
    CMAP.N,
)

SHOW_FIGURES = True

sns.set_theme(style="white", font_scale=1.05)


# -------------------------------------------------------------------
# Load and validate statistical results
# -------------------------------------------------------------------

def validate_results(results, path):
    required = {
        "model_a",
        "model_b",
        "metric",
        "mean_difference_a_minus_b",
        "holm_adjusted_p_value",
        "significant_after_holm",
    }

    missing = required.difference(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    # Detect duplicate pairs, including pairs stored in reverse.
    pairs = results.apply(
        lambda row: tuple(sorted([
            str(row["model_a"]),
            str(row["model_b"]),
        ])),
        axis=1,
    )

    keys = pd.DataFrame({
        "pair": pairs,
        "metric": results["metric"],
    })

    if keys.duplicated().any():
        raise ValueError(
            f"Duplicate comparisons found in {path}"
        )

    for column in [
        "mean_difference_a_minus_b",
        "holm_adjusted_p_value",
    ]:
        results[column] = pd.to_numeric(
            results[column],
            errors="raise",
        )

    differences = results["mean_difference_a_minus_b"]

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
            f"Invalid mean differences in {path}"
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
            f"Invalid adjusted p-values in {path}"
        )

    return results


def load_results(dataset):
    """Load the full benchmark results."""
    path = RESULTS_DIR / f"posthoc_wilcoxon_{dataset}.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark results file not found: {path}"
        )

    results = pd.read_csv(path)

    return validate_results(results, path)


def load_sp_dp_results(dataset):
    """
    Load dedicated SP–DP results.

    These use valid SP–DP pairs and Holm correction across
    four metrics within this dataset.
    """
    path = RESULTS_DIR / f"SP_DP_wilcoxon_{dataset}.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"SP–DP results file not found: {path}\n"
            "Run the updated standalone statistical script first."
        )

    results = pd.read_csv(path)

    required = {
        "metric",
        "mean_difference_dual_minus_single",
        "holm_adjusted_p_value",
        "significant_after_holm",
    }

    missing = required.difference(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    if results["metric"].duplicated().any():
        raise ValueError(
            f"Duplicate metric results in {path}"
        )

    missing_metrics = set(METRICS) - set(results["metric"])

    if missing_metrics:
        raise ValueError(
            f"Missing metrics in {path}: "
            f"{sorted(missing_metrics)}"
        )

    # Match the structure expected by get_cell().
    results = results.rename(columns={
        "mean_difference_dual_minus_single": (
            "mean_difference_a_minus_b"
        ),
    })

    results["model_a"] = "DP"
    results["model_b"] = "SP"

    return validate_results(results, path)


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
    selected = results.loc[
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

    difference = (
        direction
        * float(row["mean_difference_a_minus_b"])
    )

    if not np.isfinite(difference):
        return np.nan, "NA"

    test_available = np.isfinite(
        float(row["holm_adjusted_p_value"])
    )

    if not test_available:
        return (
            np.nan,
            f"{difference:+.4f}\nTest unavailable",
        )

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

def draw_heatmap(
    codes,
    annotations,
    title,
    filename,
    figsize,
    statistical_note,
):
    fig, ax = plt.subplots(figsize=figsize)

    unavailable = codes.isna()
    ax.set_facecolor(COLOURS["unavailable"])

    sns.heatmap(
        codes.fillna(0),
        mask=unavailable,
        annot=False,
        cmap=CMAP,
        norm=NORM,
        cbar=False,
        linewidths=2,
        linecolor="white",
        ax=ax,
    )

    # Manual annotations also display masked unavailable results.
    for row in range(len(codes.index)):
        for column in range(len(codes.columns)):
            ax.text(
                column + 0.5,
                row + 0.5,
                annotations.iloc[row, column],
                ha="center",
                va="center",
                fontsize=13,
                color="#222222",
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

    if unavailable.to_numpy().any():
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
        bbox_to_anchor=(0.5, 0.21),
        ncol=len(legend),
        frameon=False,
        fontsize=10,
    )

    fig.text(
        0.5,
        0.025,
        "Voxel level: Dice, Precision, Sensitivity. "
        "Lesion level: Detectability (IoU ≥ 0.20).\n"
        "Values: mean paired differences "
        + statistical_note,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.34, 1, 1])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "svg", "pdf"):
        path = FIGURE_DIR / f"{filename}.{extension}"

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
# Dedicated DP versus SP
# -------------------------------------------------------------------

def plot_dp_vs_sp(results, dataset):
    columns = [
        METRIC_LABELS[metric]
        for metric in METRICS
    ]

    codes = pd.DataFrame(
        0.0,
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

    versions = [
        (
            DATASET_LABELS.get(dataset, dataset),
            f"{dataset}_DP_vs_SP_matrix",
        )
    ]


    # Additional plots
    group_labels = {
        "HCC": "HCC",
        "LiTS": "Mixed",
    }

    if dataset in group_labels:
        group = group_labels[dataset]

        versions.append((
            group,
            f"{dataset}_DP_vs_SP_matrix_group_{group}",
        ))

    for label, filename in versions:
        draw_heatmap(
            codes=codes,
            annotations=annotations,
            title=(
                f"{label}: "
                "Single-Phase versus Dual-Phase"
            ),
            filename=filename,
            figsize=(10, 4.3),
            statistical_note=(
                "(DP - SP)\n"
                "* Holm-adjusted Wilcoxon p < 0.05; "
                "correction across four metrics within this dataset.\n"
                "Cases: valid pairs for each metric."
            ),
        )


# -------------------------------------------------------------------
# SP or DP versus benchmark models
# -------------------------------------------------------------------

def plot_model_vs_benchmarks(results, dataset, model):
    rows = [
        METRIC_LABELS[metric]
        for metric in METRICS
    ]

    codes = pd.DataFrame(
        0.0,
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
        title=(
            f"{DATASET_LABELS.get(dataset, dataset)}: "
            f"{model} versus Benchmark Models"
        ),
        filename=f"{dataset}_{model}_vs_benchmarks_matrix",
        figsize=(10, 6.3),
        statistical_note=(
            "(named model - comparison model)\n"
            "* Holm-adjusted Wilcoxon p < 0.05; "
            "correction across 15 model pairs per dataset and metric.\n"
            "Cases: valid pairs for the two compared models, "
            "separately for each metric."
        ),
    )



def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"\nCreating matrices for {dataset}")

        # Dedicated SP–DP comparison:
        # valid SP–DP pairs; Holm across four metrics.
        sp_dp_results = load_sp_dp_results(dataset)

        plot_dp_vs_sp(
            sp_dp_results,
            dataset,
        )

        # Full benchmark:
        # valid pairs for each model comparison and metric;
        # Holm across 15 pairs per dataset and metric;
        # no Friedman significance gate.
        benchmark_results = load_results(dataset)

        for model in ["SP", "DP"]:
            plot_model_vs_benchmarks(
                benchmark_results,
                dataset,
                model,
            )

    print(f"\nAll figures saved to:\n{FIGURE_DIR}")


if __name__ == "__main__":
    main()