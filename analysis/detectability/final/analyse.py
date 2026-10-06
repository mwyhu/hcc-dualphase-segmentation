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

# MODELS = ["TSLL", "TSLT", "AN", "SP", "DP", "COALA"]
MODELS = ["TSLL", "TSLT", "AN", "SP", "DP"]
DATASETS = ["HCC", "WORC", "TCIA", "LiTS", "CAIRO5"]

REPORT_THRESHOLD = 0.2

# MODEL_PALETTE = dict(
#     zip(MODELS, sns.color_palette("Set2", len(MODELS)))
# )

MODEL_PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    # "#72B7B2",
]

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

            if not filepath.exists():
                missing_files.append(str(filepath))
                continue

            df = pd.read_csv(filepath)

            missing_columns = required_columns.difference(df.columns)
            if missing_columns:
                raise ValueError(
                    f"{filepath.name}: missing columns "
                    f"{sorted(missing_columns)}"
                )

            for column in ["detectability", "precision", "iou_threshold"]:
                df[column] = pd.to_numeric(df[column], errors="raise")

            if df[["case", "iou_threshold"]].isna().any().any():
                raise ValueError(
                    f"{filepath.name}: missing case IDs or IoU thresholds."
                )

            for metric in ["detectability", "precision"]:
                values = df[metric].dropna()
                if not values.between(0, 1).all():
                    raise ValueError(
                        f"{filepath.name}: {metric} contains "
                        "values outside [0, 1]."
                    )

            duplicates = df.duplicated(
                subset=["case", "iou_threshold"], keep=False
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
            "Missing expected CSV files:\n" + "\n".join(missing_files)
        )

    df_all = pd.concat(frames, ignore_index=True)

    df_all["model"] = pd.Categorical(
        df_all["model"], categories=MODELS, ordered=True
    )
    df_all["dataset"] = pd.Categorical(
        df_all["dataset"], categories=DATASETS, ordered=True
    )

    # Unique cases
    df_all["case_key"] = (
        df_all["dataset"].astype(str)
        + "::"
        + df_all["case"].astype(str)
    )

    return df_all


# -------------------------------------------------------------------
# Summaries
# -------------------------------------------------------------------
def create_summary(df, group_columns):
    """
    Summarize per-case scores; each scan has equal weight.

    SD is the sample standard deviation (ddof=1).
    Q1 and Q3 are the 25th and 75th percentiles.
    Missing values are excluded separately for each metric.
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
                "detectability", lambda x: x.quantile(0.25)
            ),
            q3_case_detectability=(
                "detectability", lambda x: x.quantile(0.75)
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
    """Print dataset × model tables at the reporting threshold."""

    def as_table(series):
        return (
            series.unstack("model")
            .reindex(index=DATASETS, columns=MODELS)
            .fillna("—")
        )

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
        dataset_summary.apply(format_median_quartiles, axis=1)
    )

    precision_table = (
        dataset_summary["mean_case_precision"]
        .unstack("model")
        .reindex(index=DATASETS, columns=MODELS)
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
        f"\n--- Mean per-case precision "
        f"at IoU ≥ {threshold:.2f} ---"
    )
    print(precision_table.round(4).to_string())

# -------------------------------------------------------------------
# Plots: per-case distributions at the reporting threshold
# -------------------------------------------------------------------
def plot_metric_by_dataset(df_report, metric, ylabel, threshold):
    """
    Show each model's per-case score distribution per dataset.

    Boxes: median and 25th–75th percentiles.
    Whiskers: up to 1.5 × IQR.
    Points: individual scans, including outliers.
    """
    n_columns = 2
    n_rows = (len(DATASETS) + n_columns - 1) // n_columns

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(14, 5 * n_rows),
        sharey=True,
        squeeze=False,
    )

    rng = np.random.default_rng(42)

    for ax, dataset in zip(axes.flat, DATASETS):
        subset = df_report.loc[
            df_report["dataset"] == dataset
        ].dropna(subset=[metric])

        sns.boxplot(
            data=subset,
            x="model",
            y=metric,
            order=MODELS,
            hue="model",
            hue_order=MODELS,
            palette=MODEL_PALETTE,
            dodge=False,
            legend=False,
            width=0.6,
            showfliers=False,
            ax=ax,
        )

        # Draw all scan scores with horizontal jitter
        for position, model in enumerate(MODELS):
            scores = subset.loc[
                subset["model"] == model, metric
            ].to_numpy()

            x_positions = position + rng.uniform(
                -0.2, 0.2, size=len(scores)
            )

            ax.scatter(
                x_positions,
                scores,
                color="black",
                alpha=0.25,
                s=8,
                linewidths=0,
                zorder=3,
            )

        ax.set_title(dataset)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="x", visible=False)

    for ax in list(axes.flat)[len(DATASETS):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Per-case {ylabel.lower()} at IoU ≥ {threshold:.2f}",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_case_distributions(df_report, threshold):
    plot_metric_by_dataset(
        df_report,
        metric="detectability",
        ylabel="Detectability",
        threshold=threshold,
    )

    plot_metric_by_dataset(
        df_report,
        metric="precision",
        ylabel="Precision",
        threshold=threshold,
    )


# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------
def main():
    df_all = load_results()

    # Overall results across all datasets and thresholds
    model_threshold_summary = create_summary(
        df_all, ["model", "iou_threshold"]
    )

    print("\n--- Overall summary by model and IoU threshold ---")
    print(model_threshold_summary.round(4))

    # Separate results for each dataset and threshold
    dataset_model_threshold_summary = create_summary(
        df_all, ["dataset", "model", "iou_threshold"]
    )

    print("\n--- Summary by dataset, model and IoU threshold ---")
    print(dataset_model_threshold_summary.round(4))

    # Select the reporting threshold
    df_report = df_all.loc[
        np.isclose(df_all["iou_threshold"], REPORT_THRESHOLD)
    ].copy()

    # Verify
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

    model_summary_report = create_summary(df_report, ["model"])
    dataset_model_summary_report = create_summary(
        df_report, ["dataset", "model"]
    )

    print(
        f"\n--- Overall model summary at IoU "
        f"{REPORT_THRESHOLD:.2f} ---"
    )
    print(model_summary_report.round(4))

    print(
        f"\n--- Dataset-specific summary at IoU "
        f"{REPORT_THRESHOLD:.2f} ---"
    )
    print(dataset_model_summary_report.round(4))

    # Tables: datasets as rows, models as columns
    print_dataset_model_tables(
        dataset_model_summary_report, REPORT_THRESHOLD
    )

    # Boxplots of per-case detectability and precision per dataset
    plot_case_distributions(df_report, REPORT_THRESHOLD)


if __name__ == "__main__":
    main()