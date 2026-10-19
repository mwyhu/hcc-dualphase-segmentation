from pathlib import Path
import argparse
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

DEFAULT_RESULTS = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis/statistical/external_validation/friedman_posthoc_results"
)

MODELS = ["COALA", "AtlasNet", "TSLT", "TSLL", "DP", "SP"]
DISPLAY_MODELS = ["COALA", "AN", "TSLT", "TSLL", "DP", "SP"]

DATASETS = ["HCC", "LiTS", "TCIA", "WORC", "CAIRO5"]

METRICS = ["dice", "precision", "recall", "detectability"]

METRIC_LABELS = {
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Sensitivity",
    "detectability": "Detectability",
}

COLOURS = {
    "decrease": "#E6A09B",
    "not_significant": "#EEEEEE",
    "increase": "#9DCEAE",
    "unavailable": "#FFFFFF",
}

CMAP = ListedColormap([
    COLOURS["decrease"],
    COLOURS["not_significant"],
    COLOURS["increase"],
    COLOURS["unavailable"],
])

NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], CMAP.N)

MASK = np.triu(
    np.ones((len(MODELS), len(MODELS)), dtype=bool)
)


# -------------------------------------------------------------------
# Read and validate results
# -------------------------------------------------------------------

def significant(value):
    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    if text in ("true", "1", "1.0"):
        return True

    if text in ("false", "0", "0.0"):
        return False

    raise ValueError(f"Invalid significance flag: {value!r}")


def load_results(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    required = {
        "model_a",
        "model_b",
        "metric",
        "mean_difference_a_minus_b",
        "significant_after_holm",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}"
        )

    aliases = {model.casefold(): model for model in MODELS}
    aliases.update({
        "an": "AtlasNet",
        "singlephase": "SP",
        "dualphase": "DP",
    })

    for column in ["model_a", "model_b"]:
        data[column] = (
            data[column]
            .astype(str)
            .str.strip()
            .map(lambda name: aliases.get(name.casefold(), name))
        )

    data["metric"] = (
        data["metric"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unknown = (
        set(data["model_a"]) | set(data["model_b"])
    ) - set(MODELS)

    if unknown:
        raise ValueError(
            f"{path}: unrecognised models: {sorted(unknown)}"
        )

    data["mean_difference_a_minus_b"] = pd.to_numeric(
        data["mean_difference_a_minus_b"],
        errors="raise",
    )

    seen = set()

    for row in data.itertuples(index=False):
        if row.model_a == row.model_b:
            raise ValueError(
                f"{path}: self-comparison for {row.model_a}"
            )

        key = (
            row.metric,
            tuple(sorted([row.model_a, row.model_b])),
        )

        if key in seen:
            raise ValueError(
                f"{path}: duplicate comparison {key}"
            )

        seen.add(key)

    print(f"Read: {path}")
    return data


# -------------------------------------------------------------------
# Construct matrices
# -------------------------------------------------------------------

def build_matrix(data, metric):
    n = len(MODELS)

    codes = np.zeros((n, n), dtype=float)
    annotations = np.full((n, n), "", dtype=object)
    missing = []

    for i in range(n):
        for j in range(i):
            a = MODELS[i]
            b = MODELS[j]

            selected = data[
                (data["metric"] == metric)
                & (
                    (
                        (data["model_a"] == a)
                        & (data["model_b"] == b)
                    )
                    | (
                        (data["model_a"] == b)
                        & (data["model_b"] == a)
                    )
                )
            ]

            if selected.empty:
                codes[i, j] = 2
                annotations[i, j] = "NT"
                missing.append(f"{a} vs {b}")
                continue

            row = selected.iloc[0]

            # Difference = row model minus column model
            direction = 1 if row["model_a"] == a else -1
            difference = (
                direction
                * float(row["mean_difference_a_minus_b"])
            )

            if (
                not np.isfinite(difference)
                or pd.isna(row["significant_after_holm"])
            ):
                codes[i, j] = 2
                annotations[i, j] = "NA"
                continue

            sig = significant(row["significant_after_holm"])
            codes[i, j] = np.sign(difference) if sig else 0

            # Avoid displaying negative zero
            displayed = (
                0.0 if abs(difference) < 0.0005 else difference
            )

            annotations[i, j] = (
                f"{displayed:+.3f}" + ("*" if sig else "")
            )

    return codes, annotations, missing


# -------------------------------------------------------------------
# Plot all metrics and datasets together
# -------------------------------------------------------------------

def plot_full_comparison(results, output_dir, show=False):
    sns.set_theme(
        style="white",
        font="DejaVu Sans",
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#252525",
            "axes.labelcolor": "#252525",
            "xtick.color": "#252525",
            "ytick.color": "#252525",
        },
    )

    datasets = list(results)
    nrows = len(METRICS)
    ncols = len(datasets)


    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols + 1.5, 6.0 * nrows + 2.8),
        squeeze=False,
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.095,
        top=0.93,
        wspace=0.22,
        hspace=0.25,
    )

    has_unavailable = False

    for row_index, metric in enumerate(METRICS):
        for col_index, dataset in enumerate(datasets):
            ax = axes[row_index, col_index]

            codes, annotations, missing = build_matrix(
                results[dataset],
                metric,
            )

            if missing:
                warnings.warn(
                    f"{dataset}, {metric}: untested pairs: "
                    + ", ".join(missing)
                )

            has_unavailable |= bool(
                np.isin(annotations, ["NA", "NT"]).any()
            )

            sns.heatmap(
                codes,
                mask=MASK,
                annot=annotations,
                fmt="",
                cmap=CMAP,
                norm=NORM,
                cbar=False,
                square=True,
                linewidths=1.8,
                linecolor="white",
                xticklabels=DISPLAY_MODELS,
                yticklabels=DISPLAY_MODELS,
                annot_kws={
                    "fontsize": 11,
                    "color": "#252525",
                },
                ax=ax,
            )

            ax.invert_yaxis()
            ax.set_xlabel("")
            ax.set_ylabel("")

            ax.tick_params(
                axis="both",
                which="both",
                length=0,
                labelsize=11,
                pad=6,
            )

            ax.set_xticklabels(DISPLAY_MODELS, rotation=0)
            ax.set_yticklabels(DISPLAY_MODELS, rotation=0)

            # Dataset headings above the first row
            if row_index == 0:
                ax.set_title(
                    dataset,
                    fontsize=19,
                    fontweight="bold",
                    pad=20,
                )

            for tick in (
                ax.get_xticklabels() + ax.get_yticklabels()
            ):
                if tick.get_text() in {"SP", "DP"}:
                    tick.set_fontweight("bold")

    # Update positions after square heatmaps adjust their axes
    fig.canvas.draw()

    # Metric labels along the left side
    for row_index, metric in enumerate(METRICS):
        position = axes[row_index, 0].get_position()
        centre_y = (position.y0 + position.y1) / 2

        fig.text(
            0.016,
            centre_y,
            METRIC_LABELS[metric],
            rotation=90,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )

    fig.suptitle(
        "Model Comparisons across Datasets and Metrics",
        fontsize=26,
        fontweight="bold",
        y=0.98,
    )

    handles = [
        Patch(
            facecolor=COLOURS["increase"],
            label="Significantly higher",
        ),
        Patch(
            facecolor=COLOURS["decrease"],
            label="Significantly lower",
        ),
        Patch(
            facecolor=COLOURS["not_significant"],
            label="Not significant",
        ),
    ]

    if has_unavailable:
        handles.append(
            Patch(
                facecolor=COLOURS["unavailable"],
                edgecolor="#CCCCCC",
                label="Unavailable / not tested",
            )
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        ncol=len(handles),
        frameon=False,
        fontsize=13,
        handlelength=1.5,
        columnspacing=2,
    )

    note = (
        "Voxel level: Dice, precision, sensitivity. "
        "Lesion level: detectability (IoU ≥ 0.20).\n"
        "Values: mean paired differences (row model − column model).\n"
        "* Holm-adjusted Wilcoxon p < 0.05."
    )

    if has_unavailable:
        note += "\nNA: unavailable result; NT: pair not tested."

    fig.text(
        0.5,
        0.013,
        note,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#555555",
        linespacing=1.4,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / "all_datasets_all_metrics_comparison"

    # Vector PDF
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )

    # PNG
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Saved: {output_stem.with_suffix('.pdf')}")
    print(f"Saved: {output_stem.with_suffix('.png')}")

    if show:
        plt.show()

    plt.close(fig)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot all model comparisons in one figure: "
            "datasets as columns and metrics as rows."
        )
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
    )
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    if not args.datasets:
        parser.error("Provide at least one dataset.")

    if len(args.datasets) != len(set(args.datasets)):
        parser.error("Dataset names must be unique.")

    output_dir = (
        args.output_dir
        or args.results_dir / "all_model_comparison_figures"
    )

    results = {}

    for dataset in args.datasets:
        path = (
            args.results_dir
            / f"posthoc_wilcoxon_{dataset}.csv"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Results file not found: {path}"
            )

        results[dataset] = load_results(path)

    plot_full_comparison(
        results,
        output_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()