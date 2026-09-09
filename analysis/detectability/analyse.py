from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Settings
BASE_DIR = Path(
    "/Users/michellehu/Desktop/hcc_active_learning/analysis/detectability"
)

MODEL_FILES = {
    "Start1": "detectability_start1.csv",
    "Start2": "detectability_start2.csv",
    "Ceiling": "detectability_ceiling.csv",
    "TS": "detectability_TS.csv",
}

MODEL_ORDER = list(MODEL_FILES)
MODEL_PALETTE = dict(
    zip(MODEL_ORDER, sns.color_palette("Set2", len(MODEL_ORDER)))
)

sns.set_theme(style="whitegrid")


# Functions
def safe_divide(numerator, denominator):
    """Divide safely and return 0 when the denominator is 0."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def create_summary(df, group_columns):
    """
    Summarise detection performance

    Case-level metrics (each scan has equal weight):
      - mean_case_detectability: mean of TP / (TP + FN) per scan
      - mean_case_precision: mean of TP / (TP + FP) per scan

    Pooled lesion-level metrics (each lesion has equal weight):
      - lesion_precision: sum(TP) / [sum(TP) + sum(FP)]
      - lesion_recall: sum(TP) / [sum(TP) + sum(FN)]
      - lesion_f1: calculated from pooled TP, FP and FN
    """
    summary = (
        df.groupby(group_columns, observed=True)
        .agg(
            n_cases=("case", "nunique"),
            mean_case_detectability=("detectability", "mean"),
            mean_case_precision=("precision", "mean"),
            TP=("TP", "sum"),
            FP=("FP", "sum"),
            FN=("FN", "sum"),
        )
    )

    summary["lesion_precision"] = safe_divide(
        summary["TP"], summary["TP"] + summary["FP"]
    )
    summary["lesion_recall"] = safe_divide(
        summary["TP"], summary["TP"] + summary["FN"]
    )
    summary["lesion_f1"] = safe_divide(
        2 * summary["TP"],
        2 * summary["TP"] + summary["FP"] + summary["FN"],
    )

    return summary.round(3)


def load_results():
    frames = []

    for model, filename in MODEL_FILES.items():
        df = pd.read_csv(BASE_DIR / filename)
        df["model"] = model
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_all["source"] = df_all["case"].str.split("_").str[0]
    df_all["model"] = pd.Categorical(
        df_all["model"], categories=MODEL_ORDER, ordered=True
    )
    df_all["iou_threshold"] = pd.to_numeric(df_all["iou_threshold"])

    required_columns = {
        "case",
        "detectability",
        "precision",
        "TP",
        "FP",
        "FN",
        "iou_threshold",
    }
    missing = required_columns.difference(df_all.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df_all


def plot_case_level_metrics(df_all):
    """Plot averages in which every scan contributes equally."""
    plot_df = df_all.rename(
        columns={
            "detectability": "Case detectability",
            "precision": "Case precision",
        }
    ).melt(
        id_vars=["model", "iou_threshold"],
        value_vars=["Case detectability", "Case precision"],
        var_name="metric",
        value_name="score",
    )

    g = sns.relplot(
        data=plot_df,
        kind="line",
        x="iou_threshold",
        y="score",
        hue="model",
        hue_order=MODEL_ORDER,
        col="metric",
        markers=True,
        dashes=False,
        errorbar=None,
        palette=MODEL_PALETTE,
        height=4.5,
        aspect=1.1,
    )
    g.set_axis_labels("IoU threshold", "Mean score across cases")
    g.set(ylim=(0, 1))
    g.set_titles("{col_name}")
    g.fig.suptitle("Case-level performance (each scan has equal weight)", y=1.04)
    plt.show()


def plot_lesion_level_metrics(model_threshold_summary):
    """Plot pooled metrics in which every lesion contributes equally."""
    plot_df = (
        model_threshold_summary.reset_index()
        .melt(
            id_vars=["model", "iou_threshold"],
            value_vars=["lesion_precision", "lesion_recall", "lesion_f1"],
            var_name="metric",
            value_name="score",
        )
    )

    metric_names = {
        "lesion_precision": "Lesion precision",
        "lesion_recall": "Lesion recall",
        "lesion_f1": "Lesion F1",
    }
    plot_df["metric"] = plot_df["metric"].map(metric_names)

    g = sns.relplot(
        data=plot_df,
        kind="line",
        x="iou_threshold",
        y="score",
        hue="model",
        hue_order=MODEL_ORDER,
        col="metric",
        col_wrap=3,
        markers=True,
        dashes=False,
        palette=MODEL_PALETTE,
        height=4,
        aspect=1,
    )
    g.set_axis_labels("IoU threshold", "Pooled score")
    g.set(ylim=(0, 1))
    g.set_titles("{col_name}")
    g.fig.suptitle("Lesion-level performance (each lesion has equal weight)", y=1.04)
    plt.show()


def plot_summary_at_threshold(model_summary, threshold):
    """Show the main case-level and lesion-level results together."""
    columns = [
        "mean_case_detectability",
        "mean_case_precision",
        "lesion_precision",
        "lesion_recall",
        "lesion_f1",
    ]
    names = {
        "mean_case_detectability": "Mean case detectability",
        "mean_case_precision": "Mean case precision",
        "lesion_precision": "Lesion precision",
        "lesion_recall": "Lesion recall",
        "lesion_f1": "Lesion F1",
    }

    plot_df = model_summary.reset_index().melt(
        id_vars="model",
        value_vars=columns,
        var_name="metric",
        value_name="score",
    )
    plot_df["metric"] = plot_df["metric"].map(names)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="metric",
        y="score",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
    )
    plt.title(f"Detection summary at IoU threshold {threshold:.2f}")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


#
# Main analysis
df_all = load_results()

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

# One result per model and IoU threshold, pooled across all data sources
model_threshold_summary = create_summary(
    df_all, ["model", "iou_threshold"]
)

print("\n--- Overall summary by model and IoU threshold ---")
print(model_threshold_summary)

# Sep results for each data source
source_model_threshold_summary = create_summary(
    df_all, ["source", "model", "iou_threshold"]
)

print("\n--- Summary by source, model and IoU threshold ---")
print(source_model_threshold_summary)

# Main reporting threshold
REPORT_THRESHOLD = 0.15
df_report = df_all[np.isclose(df_all["iou_threshold"], REPORT_THRESHOLD)]

model_summary_report = create_summary(df_report, ["model"])
source_model_summary_report = create_summary(df_report, ["source", "model"])

print(f"\n--- Overall model summary at IoU {REPORT_THRESHOLD:.2f} ---")
print(model_summary_report)

print(f"\n--- Source-specific summary at IoU {REPORT_THRESHOLD:.2f} ---")
print(source_model_summary_report)

plot_case_level_metrics(df_all)
plot_lesion_level_metrics(model_threshold_summary)
plot_summary_at_threshold(model_summary_report, REPORT_THRESHOLD)
