from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
BASE_DIR = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis/detectability/final"
)

MODELS = ["TSLL", "TSLT", "AN", "SP", "DP", "COALA"]
DATASETS = ["HCC", "WORC", "TCIA", "LiTS", "CAIRO5"]

PLOT_MODELS = ["TSLL", "TSLT", "COALA", "AN", "SP", "DP"]
PLOT_DATASETS = ["HCC", "LiTS", "TCIA", "WORC", "CAIRO5"]
CRLM_DATASETS = ["TCIA", "WORC", "CAIRO5"]

MODEL_LABELS = {
    "TSLL": "TSLL",
    "TSLT": "TSLT",
    "COALA": "COALA",
    "AN": "AtlasNet",
    "SP": "SP",
    "DP": "DP",
}

MODEL_PALETTE = {
    "TSLL": "#4C78A8",
    "TSLT": "#F58518",
    "COALA": "#72B7B2",
    "AN": "#54A24B",
    "SP": "#B279A2",
    "DP": "#E45756",
}

REPORT_THRESHOLD = 0.20

PLOT_DIR = BASE_DIR / "plots"

sns.set_theme(style="whitegrid")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# -------------------------------------------------------------------
# Load results
# -------------------------------------------------------------------
def load_results():
    """Load all model–dataset combinations."""
    frames = []
    missing_files = []

    required_columns = {
        "case",
        "detectability",
        "precision",
        "iou_threshold",
    }

    for dataset in DATASETS:
        for model in MODELS:
            filepath = BASE_DIR / f"Detect_{model}_{dataset}.csv"

            if not filepath.is_file():
                missing_files.append(str(filepath))
                continue

            df = pd.read_csv(filepath)

            missing_columns = required_columns.difference(
                df.columns
            )

            if missing_columns:
                raise ValueError(
                    f"{filepath.name}: missing columns "
                    f"{sorted(missing_columns)}"
                )

            for column in [
                "detectability",
                "precision",
                "iou_threshold",
            ]:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="raise",
                )

            if df[["case", "iou_threshold"]].isna().any().any():
                raise ValueError(
                    f"{filepath.name}: missing case IDs "
                    "or IoU thresholds."
                )

            for metric in ["detectability", "precision"]:
                values = df[metric].dropna()
                if not values.between(0, 1).all():
                    raise ValueError(
                        f"{filepath.name}: {metric} contains "
                        "values outside [0, 1]."
                    )

            duplicates = df.duplicated(
                subset=["case", "iou_threshold"],
                keep=False,
            )
            if duplicates.any():
                raise ValueError(
                    f"{filepath.name}: duplicate case–threshold rows:\n"
                    f"{df.loc[duplicates, ['case', 'iou_threshold']]}"
                )

            df["model"] = model
            df["dataset"] = dataset
            frames.append(df)

    if missing_files:
        raise FileNotFoundError(
            "Missing expected CSV files:\n"
            + "\n".join(missing_files)
        )

    df_all = pd.concat(frames, ignore_index=True)

    df_all["model"] = pd.Categorical(
        df_all["model"],
        categories=MODELS,
        ordered=True,
    )
    df_all["dataset"] = pd.Categorical(
        df_all["dataset"],
        categories=DATASETS,
        ordered=True,
    )

    df_all["case_key"] = (
        df_all["dataset"].astype(str)
        + "::"
        + df_all["case"].astype(str)
    )

    return df_all


# -------------------------------------------------------------------
# Summaries: original calculations
# -------------------------------------------------------------------
def create_summary(df, group_columns):
    """
    Summarise per-case scores; each scan has equal weight.
    """
    return (
        df.groupby(group_columns, observed=True)
        .agg(
            n_cases=("case_key", "nunique"),
            n_detectability=("detectability", "count"),
            n_precision=("precision", "count"),

            mean_case_detectability=("detectability", "mean"),
            std_case_detectability=("detectability", "std"),
            median_case_detectability=("detectability", "median"),
            q1_case_detectability=(
                "detectability",
                lambda x: x.quantile(0.25),
            ),
            q3_case_detectability=(
                "detectability",
                lambda x: x.quantile(0.75),
            ),

            mean_case_precision=("precision", "mean"),
        )
        .assign(
            iqr_case_detectability=lambda x: (
                x["q3_case_detectability"]
                - x["q1_case_detectability"]
            )
        )
    )


def print_dataset_model_tables(dataset_summary, threshold):
    """
    Print dataset X model tables at the reporting threshold
    """

    def as_table(series):
        table = (
            series.unstack("model")
            .reindex(
                index=PLOT_DATASETS,
                columns=PLOT_MODELS,
            )
            .fillna("—")
        )
        return table.rename(columns=MODEL_LABELS)

    def format_mean_sd(row):
        mean = row["mean_case_detectability"]
        sd = row["std_case_detectability"]

        if pd.isna(mean):
            return "—"
        if pd.isna(sd):
            return f"{mean:.4f} ± NA"
        return f"{mean:.4f} ± {sd:.4f}"

    def format_median_quartiles(row):
        median = row["median_case_detectability"]
        q1 = row["q1_case_detectability"]
        q3 = row["q3_case_detectability"]

        if pd.isna(median):
            return "—"
        return f"{median:.4f} [{q1:.4f}, {q3:.4f}]"

    mean_sd_table = as_table(
        dataset_summary.apply(format_mean_sd, axis=1)
    )

    median_iqr_table = as_table(
        dataset_summary.apply(
            format_median_quartiles,
            axis=1,
        )
    )

    precision_table = (
        dataset_summary["mean_case_precision"]
        .unstack("model")
        .reindex(
            index=PLOT_DATASETS,
            columns=PLOT_MODELS,
        )
        .rename(columns=MODEL_LABELS)
    )

    print(
        f"\n--- Mean per-case detectability ± SD "
        f"at IoU ≥ {threshold:.2f} ---"
    )
    print(mean_sd_table.to_string())

    print(
        f"\n--- Median per-case detectability [Q1, Q3] "
        f"at IoU ≥ {threshold:.2f} ---"
    )
    print(median_iqr_table.to_string())

    print(
        f"\n--- Mean per-case lesion-level precision "
        f"at IoU ≥ {threshold:.2f} ---"
    )
    print(precision_table.round(4).to_string())

# -------------------------------------------------------------------
# Detectability plots
# -------------------------------------------------------------------
def draw_detectability_panel(
    ax,
    df_report,
    datasets,
    title,
    equal_dataset_weighting=False,
):
    rng = np.random.default_rng(42)

    if equal_dataset_weighting:
        boxes = []

        for model in PLOT_MODELS:
            parts = [
                df_report.loc[
                    (df_report["dataset"] == dataset)
                    & (df_report["model"] == model),
                    "detectability",
                ].dropna().to_numpy(dtype=float)
                for dataset in datasets
            ]

            if any(len(a) == 0 for a in parts):
                raise ValueError(
                    f"Missing detectability scores for "
                    f"{model} in {datasets}"
                )

            values = np.concatenate(parts)

            # Each dataset receives the same total weight.
            weights = np.concatenate([
                np.full(
                    len(a),
                    1 / (len(datasets) * len(a)),
                )
                for a in parts
            ])

            order = np.argsort(values)
            sorted_values = values[order]

            cumulative = np.cumsum(weights[order])
            cumulative /= cumulative[-1]

            indices = np.searchsorted(
                cumulative,
                [0.25, 0.50, 0.75],
                side="left",
            )

            q1, median, q3 = sorted_values[indices]
            iqr = q3 - q1

            boxes.append({
                "q1": float(q1),
                "med": float(median),
                "q3": float(q3),
                "whislo": float(
                    values[values >= q1 - 1.5 * iqr].min()
                ),
                "whishi": float(
                    values[values <= q3 + 1.5 * iqr].max()
                ),
                "fliers": [],
            })

        artists = ax.bxp(
            boxes,
            positions=np.arange(len(PLOT_MODELS)),
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops={
                "color": "#222222",
                "linewidth": 1.5,
            },
        )

        for box, model in zip(
            artists["boxes"],
            PLOT_MODELS,
        ):
            box.set_facecolor(MODEL_PALETTE[model])

    else:
        subset = df_report.loc[
            df_report["dataset"].isin(datasets)
        ].dropna(subset=["detectability"])

        sns.boxplot(
            data=subset,
            x="model",
            y="detectability",
            order=PLOT_MODELS,
            hue="model",
            hue_order=PLOT_MODELS,
            palette=MODEL_PALETTE,
            dodge=False,
            width=0.6,
            showfliers=False,
            ax=ax,
        )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    # Individual cases, including outliers.
    subset = df_report.loc[
        df_report["dataset"].isin(datasets)
    ].dropna(subset=["detectability"])

    for position, model in enumerate(PLOT_MODELS):
        values = subset.loc[
            subset["model"] == model,
            "detectability",
        ].to_numpy(dtype=float)

        ax.scatter(
            position + rng.uniform(
                -0.2,
                0.2,
                size=len(values),
            ),
            values,
            color="black",
            alpha=0.25,
            s=8,
            linewidths=0,
            zorder=3,
        )

    ax.set_xticks(range(len(PLOT_MODELS)))
    ax.set_xticklabels(
        [MODEL_LABELS[model] for model in PLOT_MODELS],
        rotation=30,
        ha="right",
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="x", visible=False)


def finish_detectability_figure(fig, filename):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    path = PLOT_DIR / filename

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Saved: {path}")
    plt.show()
    plt.close(fig)


def plot_case_distributions(df_report, threshold):
    threshold_token = f"{threshold:.2f}".replace(".", "p")

    for dataset in PLOT_DATASETS:
        fig, ax = plt.subplots(figsize=(8, 5))

        draw_detectability_panel(
            ax,
            df_report,
            datasets=[dataset],
            title=(
                f"{dataset}: detectability "
                f"at IoU ≥ {threshold:.2f}"
            ),
        )

        ax.set_ylabel("Detectability")
        fig.tight_layout()

        finish_detectability_figure(
            fig,
            f"{dataset}_detectability_IoU_{threshold_token}.png",
        )

    # All 5 external datasets in one figure
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(23, 5),
        sharey=True,
    )

    for ax, dataset in zip(axes, PLOT_DATASETS):
        draw_detectability_panel(
            ax,
            df_report,
            datasets=[dataset],
            title=dataset,
        )

    axes[0].set_ylabel("Detectability")

    fig.suptitle(
        f"Detectability at IoU ≥ {threshold:.2f}",
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    finish_detectability_figure(
        fig,
        f"all_datasets_detectability_IoU_{threshold_token}.png",
    )

    # HCC, LiTS, and combined CRLM in one figure
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
        sharey=True,
    )

    draw_detectability_panel(
        axes[0],
        df_report,
        datasets=["HCC"],
        title="HCC",
    )

    draw_detectability_panel(
        axes[1],
        df_report,
        datasets=["LiTS"],
        title="LiTS: mixed tumours",
    )

    draw_detectability_panel(
        axes[2],
        df_report,
        datasets=CRLM_DATASETS,
        title="CRLM: TCIA, WORC and CAIRO5",
        equal_dataset_weighting=True,
    )

    axes[0].set_ylabel("Detectability")

    fig.suptitle(
        f"Detectability by tumour group "
        f"at IoU ≥ {threshold:.2f}",
        fontsize=16,
    )

    fig.text(
        0.5,
        0.02,
        "CRLM boxes give TCIA, WORC and CAIRO5 equal weight. "
        "Dots show individual cases; their density reflects sample size.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    finish_detectability_figure(
        fig,
        f"HCC_LiTS_CRLM_detectability_IoU_{threshold_token}.png",
    )


def plot_lesion_precision(df_report, threshold):
    fig, axes = plt.subplots(
        1,
        len(PLOT_DATASETS),
        figsize=(23, 5),
        sharey=True,
    )

    rng = np.random.default_rng(42)

    for ax, dataset in zip(axes, PLOT_DATASETS):
        subset = df_report.loc[
            df_report["dataset"] == dataset
        ].dropna(subset=["precision"])

        sns.boxplot(
            data=subset,
            x="model",
            y="precision",
            order=PLOT_MODELS,
            hue="model",
            hue_order=PLOT_MODELS,
            palette=MODEL_PALETTE,
            dodge=False,
            width=0.6,
            showfliers=False,
            ax=ax,
        )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        for position, model in enumerate(PLOT_MODELS):
            values = subset.loc[
                subset["model"] == model,
                "precision",
            ].to_numpy(dtype=float)

            ax.scatter(
                position + rng.uniform(
                    -0.2, 0.2, size=len(values)
                ),
                values,
                color="black",
                alpha=0.25,
                s=8,
                linewidths=0,
                zorder=3,
            )

        ax.set_xticks(range(len(PLOT_MODELS)))
        ax.set_xticklabels(
            [MODEL_LABELS[model] for model in PLOT_MODELS],
            rotation=30,
            ha="right",
        )

        ax.set_title(dataset)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Lesion-level precision")

    fig.suptitle(
        f"Lesion-level precision at IoU ≥ {threshold:.2f}",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    threshold_token = f"{threshold:.2f}".replace(".", "p")

    finish_detectability_figure(
        fig,
        f"all_datasets_lesion_precision_IoU_{threshold_token}.png",
    )


def plot_lesion_precision_by_tumour_group(df_report, threshold):
    precision_df = df_report.drop(
        columns=["detectability"]
    ).rename(columns={"precision": "detectability"})

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
        sharey=True,
    )

    draw_detectability_panel(
        axes[0],
        precision_df,
        datasets=["HCC"],
        title="HCC",
    )

    draw_detectability_panel(
        axes[1],
        precision_df,
        datasets=["LiTS"],
        title="LiTS: mixed tumours",
    )

    draw_detectability_panel(
        axes[2],
        precision_df,
        datasets=CRLM_DATASETS,
        title="CRLM: TCIA, WORC and CAIRO5",
        equal_dataset_weighting=True,
    )

    axes[0].set_ylabel("Lesion-level precision")

    fig.suptitle(
        f"Lesion-level precision by tumour group "
        f"at IoU ≥ {threshold:.2f}",
        fontsize=16,
    )

    fig.text(
        0.5,
        0.02,
        "CRLM boxes give TCIA, WORC and CAIRO5 equal weight. "
        "Dots show individual cases; their density reflects sample size.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    threshold_token = f"{threshold:.2f}".replace(".", "p")

    finish_detectability_figure(
        fig,
        f"HCC_LiTS_CRLM_lesion_precision_IoU_{threshold_token}.png",
    )


# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------
def main():
    df_all = load_results()

    model_threshold_summary = create_summary(
        df_all,
        ["model", "iou_threshold"],
    )

    print(
        "\n--- Overall pooled summary "
        "by model and IoU threshold ---"
    )
    print(model_threshold_summary.round(4))

    dataset_model_threshold_summary = create_summary(
        df_all,
        ["dataset", "model", "iou_threshold"],
    )

    print(
        "\n--- Summary by dataset, model "
        "and IoU threshold ---"
    )
    print(dataset_model_threshold_summary.round(4))

    df_report = df_all.loc[
        np.isclose(
            df_all["iou_threshold"],
            REPORT_THRESHOLD,
            rtol=0,
            atol=1e-8,
        )
    ].copy()

    available = set(
        df_report[["dataset", "model"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    expected = {
        (dataset, model)
        for dataset in DATASETS
        for model in MODELS
    }
    missing = expected - available

    if missing:
        raise ValueError(
            f"No results at IoU {REPORT_THRESHOLD:.2f} for: "
            f"{sorted(missing)}"
        )

    duplicates = df_report.duplicated(
        subset=["dataset", "model", "case"],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Multiple reporting-threshold rows "
            "for the same dataset/model/case:\n"
            + df_report.loc[
                duplicates,
                ["dataset", "model", "case", "iou_threshold"],
            ].to_string(index=False)
        )

    model_summary_report = create_summary(
        df_report,
        ["model"],
    )

    dataset_model_summary_report = create_summary(
        df_report,
        ["dataset", "model"],
    )

    print(
        f"\n--- Overall pooled model summary "
        f"at IoU {REPORT_THRESHOLD:.2f} ---"
    )
    print(model_summary_report.round(4))

    print(
        f"\n--- Dataset-specific summary "
        f"at IoU {REPORT_THRESHOLD:.2f} ---"
    )
    print(dataset_model_summary_report.round(4))

    print_dataset_model_tables(
        dataset_model_summary_report,
        REPORT_THRESHOLD,
    )

    plot_case_distributions(
        df_report,
        REPORT_THRESHOLD,
    )

    plot_case_distributions(
        df_report,
        REPORT_THRESHOLD,
    )

    plot_lesion_precision(
        df_report,
        REPORT_THRESHOLD,
    )

    plot_lesion_precision_by_tumour_group(
        df_report,
        REPORT_THRESHOLD,
    )

    print(f"\nPlots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()