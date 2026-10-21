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

NORM = BoundaryNorm(
    [-1.5, -0.5, 0.5, 1.5, 2.5],
    CMAP.N,
)

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

    print(f"Reading: {path.resolve()}")

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

    aliases = {
        model.casefold(): model
        for model in MODELS
    }
    aliases.update({
        "an": "AtlasNet",
        "atlasnet": "AtlasNet",
        "singlephase": "SP",
        "dualphase": "DP",
    })

    for column in ["model_a", "model_b"]:
        cleaned = data[column].astype(str).str.strip()
        data[column] = cleaned.map(
            lambda name: aliases.get(name.casefold(), name)
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
            f"{path}: unrecognised model names: {sorted(unknown)}"
        )

    data["mean_difference_a_minus_b"] = pd.to_numeric(
        data["mean_difference_a_minus_b"],
        errors="raise",
    )

    expected_rows = len(METRICS) * (
        len(MODELS) * (len(MODELS) - 1) // 2
    )

    print(
        f"Loaded {len(data)} comparison rows; "
        f"a full matrix expects {expected_rows}."
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

            # Express the difference as row model minus column model.
            direction = 1 if row["model_a"] == a else -1

            # Original metric scale; no percentage-point conversion.
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

            # Avoid displaying negative zero after rounding.
            displayed_difference = (
                0.0 if abs(difference) < 0.0005 else difference
            )

            annotations[i, j] = (
                f"{displayed_difference:+.3f}"
                + ("*" if sig else "")
            )

    return codes, annotations, missing


def warn_missing_comparisons(data, dataset):
    for metric in METRICS:
        _, _, missing = build_matrix(data, metric)

        if missing:
            warnings.warn(
                f"{dataset}, {metric}: untested pairs: "
                f'{", ".join(missing)}'
            )


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

def set_plot_style():
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


def draw_matrix(ax, data, metric, title):
    codes, annotations, _ = build_matrix(data, metric)

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
            "fontsize": 10,
            "color": "#252525",
        },
        ax=ax,
    )

    ax.invert_yaxis()

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelsize=10,
        pad=6,
    )

    ax.set_xticklabels(DISPLAY_MODELS, rotation=0)
    ax.set_yticklabels(DISPLAY_MODELS, rotation=0)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        if tick.get_text() in {"SP", "DP"}:
            tick.set_fontweight("bold")

    return bool(np.isin(annotations, ["NA", "NT"]).any())


def add_shared_footer(fig, has_unavailable):
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
        bbox_to_anchor=(0.5, 0.9 / fig.get_figheight()),
        ncol=len(handles),
        frameon=False,
        fontsize=10,
        handlelength=1.3,
        columnspacing=1.5,
    )

    note = (
        "Voxel level: Dice, Precision, Sensitivity. Lesions level: Detectability (IoU ≥ 0.20).\n"
        "Values: mean paired differences (row model − column model).\n"
        "* Significant after Holm correction, and Friedman test."
    )


    if has_unavailable:
        note += "\nNA: unavailable result; NT: pair not tested."

    fig.text(
        0.5,
        0.15 / fig.get_figheight(),
        note,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
        linespacing=1.4,
    )


def save_figure(fig, output_dir, filename, show=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        path = output_dir / f"{filename}.{extension}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Saved: {path}")

    if show:
        plt.show()

    plt.close(fig)


# -------------------------------------------------------------------
# All datasets together: one figure per metric
# -------------------------------------------------------------------

def plot_metric_across_datasets(results, output_dir, show=False):
    set_plot_style()

    datasets = list(results)

    ncols = len(datasets)
    nrows = 1

    for metric in METRICS:
        fig, axes = plt.subplots(
            1,
            len(datasets),
            figsize=(6.5 * len(datasets), 9),
            squeeze=False,
        )

        has_unavailable = False

        for ax, dataset in zip(axes.flat, datasets):
            unavailable = draw_matrix(
                ax,
                results[dataset],
                metric,
                title=DATASET_LABELS.get(dataset, dataset),
            )
            has_unavailable |= unavailable

        # Hide unused panels.
        for ax in list(axes.flat)[len(datasets):]:
            ax.set_visible(False)

        fig.suptitle(
            f"{METRIC_LABELS[metric]}: "
            "model comparisons across datasets",
            fontsize=20,
            fontweight="bold",
            y=0.99,
        )

        # Reserve space for the shared title, legend and explanation.
        footer_fraction = 1.6 / fig.get_figheight()
        title_fraction = 0.7 / fig.get_figheight()

        fig.tight_layout(
            rect=[0, 0.25, 1, 0.83],
            w_pad=2.5,
        )

        add_shared_footer(fig, has_unavailable)

        save_figure(
            fig,
            output_dir,
            f"all_datasets_{metric}",
            show,
        )


# -------------------------------------------------------------------
# All datasets and all metrics in one figure
# -------------------------------------------------------------------

def plot_all_datasets_all_metrics(results, output_dir, show=False):
    set_plot_style()

    datasets = list(results)

    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=len(METRICS),
        figsize=(6.5 * len(METRICS), 5.5 * len(datasets) + 2),
        squeeze=False,
    )

    has_unavailable = False

    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]

            unavailable = draw_matrix(
                ax,
                results[dataset],
                metric,
                title=METRIC_LABELS[metric] if i == 0 else "",
            )
            has_unavailable |= unavailable

            # Show the full dataset name beside each row.
            if j == 0:
                ax.set_ylabel(
                    DATASET_LABELS.get(dataset, dataset),
                    fontsize=15,
                    fontweight="bold",
                    labelpad=20,
                )

    fig.suptitle(
        "Model comparisons across datasets and metrics",
        fontsize=22,
        fontweight="bold",
        y=0.99,
    )

    # Reserve a fixed amount of space for the title and footer.
    footer_fraction = 1.8 / fig.get_figheight()
    title_fraction = 0.8 / fig.get_figheight()

    fig.tight_layout(
        rect=[0, footer_fraction, 1, 1 - title_fraction],
        h_pad=3,
        w_pad=2.5,
    )

    add_shared_footer(fig, has_unavailable)

    save_figure(
        fig,
        output_dir,
        "all_datasets_all_metrics_comparison",
        show,
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot all datasets together, separately for each metric."
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

    output_dir = (
        args.output_dir
        or args.results_dir / "all_model_comparison_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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

        data = load_results(path)
        warn_missing_comparisons(data, dataset)
        results[dataset] = data

    plot_metric_across_datasets(
        results,
        output_dir,
        show=args.show,
    )

    plot_all_datasets_all_metrics(
        results,
        output_dir,
        show=args.show,
    )

if __name__ == "__main__":
    main()