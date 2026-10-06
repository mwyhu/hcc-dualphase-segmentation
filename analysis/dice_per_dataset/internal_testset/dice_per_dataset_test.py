from pathlib import Path
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Paths to nnU-Net evaluation JSON files
# -------------------------------------------------------------------

DICE_FILES = {
    "SP": "/Users/michellehu/Desktop/hcc-dualphase-segmentation/analysis/dice_per_dataset/internal_testset/SP_summary.json",
    "DP": "/Users/michellehu/Desktop/hcc-dualphase-segmentation/analysis/dice_per_dataset/internal_testset/DP_summary.json",
}

MODEL_ORDER = ["SP", "DP"]

MODEL_PALETTE = {
    "SP": "#C44E52",
    "DP": "#8172B2",
}


# -------------------------------------------------------------------
# Identify the dataset from the case filename
# -------------------------------------------------------------------

def get_dataset(case_name):
    """Determine the source dataset from the case filename."""
    dataset_prefixes = [
        "HCC_TACE",
    ]

    for dataset in dataset_prefixes:
        if case_name.startswith(dataset):
            return dataset

    return case_name.split("_")[0]


# -------------------------------------------------------------------
# Extract average Dice stored in an nnU-Net summary file
# -------------------------------------------------------------------

def extract_summary_average_dice(results, json_path):
    """Support common nnU-Net summary JSON formats."""
    if (
        "foreground_mean" in results
        and "Dice" in results["foreground_mean"]
    ):
        return results["foreground_mean"]["Dice"]

    if (
        "mean" in results
        and "1" in results["mean"]
        and "Dice" in results["mean"]["1"]
    ):
        return results["mean"]["1"]["Dice"]

    raise KeyError(
        f"Could not find the average Dice in:\n{json_path}\n"
        "Expected foreground_mean['Dice'] or mean['1']['Dice']."
    )


# -------------------------------------------------------------------
# Formatting helpers
# -------------------------------------------------------------------

def format_median_iqr(row, metric):
    """Format median and interquartile interval as median [Q1, Q3]."""
    median = row[f"median_{metric}"]
    q1 = row[f"q1_{metric}"]
    q3 = row[f"q3_{metric}"]

    if pd.isna(median):
        return "NaN"

    return f"{median:.4f} [{q1:.4f}, {q3:.4f}]"


def format_summary_for_print(summary, metrics):
    """Format median columns while retaining numeric source tables."""
    formatted = summary.copy()

    for metric in metrics:
        formatted[f"median_{metric}"] = summary.apply(
            lambda row: format_median_iqr(row, metric),
            axis=1,
        )

        formatted = formatted.drop(
            columns=[f"q1_{metric}", f"q3_{metric}"]
        )

    return formatted.round(4)


# -------------------------------------------------------------------
# Read per-case Dice, TP, FP, FN, precision, and recall
# -------------------------------------------------------------------

def load_dice_results(dice_files):
    case_rows = []
    summary_rows = []

    for model, json_path in dice_files.items():
        with open(json_path, "r") as file:
            results = json.load(file)

        summary_rows.append({
            "model": model,
            "summary_average_dice": extract_summary_average_dice(
                results,
                json_path,
            ),
        })

        for case_result in results["metric_per_case"]:
            case_name = Path(
                case_result["prediction_file"]
            ).name.removesuffix(".nii.gz")

            metrics = case_result["metrics"]["1"]

            dice = metrics["Dice"]
            tp = metrics["TP"]
            fp = metrics["FP"]
            fn = metrics["FN"]

            # Undefined when no positive voxels are predicted
            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else float("nan")
            )

            # Undefined when the ground truth contains no tumour voxels
            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else float("nan")
            )

            case_rows.append({
                "case": case_name,
                "source": get_dataset(case_name),
                "model": model,
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })

    df_dice = pd.DataFrame(case_rows)
    df_summary_dice = pd.DataFrame(summary_rows)

    df_dice["model"] = pd.Categorical(
        df_dice["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    df_summary_dice["model"] = pd.Categorical(
        df_summary_dice["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    df_summary_dice = (
        df_summary_dice
        .sort_values("model")
        .set_index("model")
    )

    return df_dice, df_summary_dice


# -------------------------------------------------------------------
# Precision and recall summary helper
# -------------------------------------------------------------------

def summarise_precision_recall(df, group_columns):
    """Calculate per-case statistics, excluding undefined values."""
    return (
        df
        .groupby(group_columns, observed=True)
        .agg(
            n_cases=("case", "nunique"),
            n_cases_with_prediction=("precision", "count"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            median_precision=("precision", "median"),
            q1_precision=("precision", lambda x: x.quantile(0.25)),
            q3_precision=("precision", lambda x: x.quantile(0.75)),
            n_cases_with_gt_tumour=("recall", "count"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            median_recall=("recall", "median"),
            q1_recall=("recall", lambda x: x.quantile(0.25)),
            q3_recall=("recall", lambda x: x.quantile(0.75)),
        )
    )


# -------------------------------------------------------------------
# Dice summary helper
# -------------------------------------------------------------------

def summarise_dice(df, group_columns):
    summary = (
        df
        .groupby(group_columns, observed=True)
        .agg(
            n_cases=("case", "nunique"),
            n_cases_with_dice=("dice", "count"),
            mean_dice=("dice", "mean"),
            median_dice=("dice", "median"),
            std_dice=("dice", "std"),
        )
    )

    summary["standard_error"] = (
        summary["std_dice"]
        / summary["n_cases_with_dice"] ** 0.5
    )

    summary["ci95_lower"] = (
        summary["mean_dice"]
        - 1.96 * summary["standard_error"]
    ).clip(lower=0)

    summary["ci95_upper"] = (
        summary["mean_dice"]
        + 1.96 * summary["standard_error"]
    ).clip(upper=1)

    return summary


# -------------------------------------------------------------------
# Load results
# -------------------------------------------------------------------

df_dice, summary_file_dice = load_dice_results(DICE_FILES)


# -------------------------------------------------------------------
# Precision and recall per model
# -------------------------------------------------------------------

overall_precision_recall = summarise_precision_recall(
    df_dice,
    ["model"],
)

print(
    "\n--- Overall Voxel-level Precision and Recall per Model "
    "(Median [Q1, Q3]) ---"
)

print(
    format_summary_for_print(
        overall_precision_recall,
        ["precision", "recall"],
    ).to_string()
)


# -------------------------------------------------------------------
# Precision and recall per dataset and model
# -------------------------------------------------------------------

dataset_precision_recall = summarise_precision_recall(
    df_dice,
    ["source", "model"],
)

print(
    "\n--- Voxel-level Precision and Recall per Dataset and Model "
    "(Median [Q1, Q3]) ---"
)

print(
    format_summary_for_print(
        dataset_precision_recall,
        ["precision", "recall"],
    ).to_string()
)


# -------------------------------------------------------------------
# Dice per dataset and model
# -------------------------------------------------------------------

dice_summary = summarise_dice(
    df_dice,
    ["source", "model"],
)

print("\n--- Dice per Dataset and Model ---")
print(dice_summary.round(4).to_string())


# -------------------------------------------------------------------
# Overall Dice per model
# -------------------------------------------------------------------

overall_dice_summary = summarise_dice(
    df_dice,
    ["model"],
)

overall_dice_summary = overall_dice_summary.join(
    summary_file_dice
)

overall_dice_summary["difference"] = (
    overall_dice_summary["mean_dice"]
    - overall_dice_summary["summary_average_dice"]
)

print("\n--- Overall Dice per Model ---")
print(overall_dice_summary.round(4).to_string())

print("\n--- Average Dice Stored in nnU-Net Summary Files ---")
print(summary_file_dice.round(4).to_string())


# -------------------------------------------------------------------
# Overall Dice, precision, and recall comparison
# -------------------------------------------------------------------

comparison_table = (
    overall_dice_summary[
        [
            "n_cases",
            "mean_dice",
            "std_dice",
            "median_dice",
            "ci95_lower",
            "ci95_upper",
        ]
    ]
    .join(
        overall_precision_recall.drop(columns="n_cases")
    )
)

print(
    "\n--- Dice, Voxel-level Precision and Recall per Model "
    "(Median Precision/Recall [Q1, Q3]) ---"
)

print(
    format_summary_for_print(
        comparison_table,
        ["precision", "recall"],
    ).to_string()
)


# -------------------------------------------------------------------
# Mean tables per dataset and model
# -------------------------------------------------------------------

METRIC_LABELS = {
    "dice": "Dice",
    "precision": "Voxel-level Precision",
    "recall": "Voxel-level Recall",
}

for metric, metric_label in METRIC_LABELS.items():
    mean_table = (
        df_dice
        .groupby(["source", "model"], observed=True)[metric]
        .mean()
        .unstack("model")
        .reindex(columns=MODEL_ORDER)
        .round(4)
    )

    print(f"\n--- Mean {metric_label} per Dataset and Model ---")
    print(mean_table.to_string())


# -------------------------------------------------------------------
# Median Dice per dataset and model
# -------------------------------------------------------------------

median_dice_table = (
    df_dice
    .groupby(["source", "model"], observed=True)["dice"]
    .median()
    .unstack("model")
    .reindex(columns=MODEL_ORDER)
    .round(4)
)

print("\n--- Median Dice per Dataset and Model ---")
print(median_dice_table.to_string())


# -------------------------------------------------------------------
# Median precision and recall per dataset with IQR [Q1, Q3]
# -------------------------------------------------------------------

for metric in ["precision", "recall"]:
    formatted_medians = dataset_precision_recall.apply(
        lambda row: format_median_iqr(row, metric),
        axis=1,
    )

    median_iqr_table = (
        formatted_medians
        .unstack("model")
        .reindex(columns=MODEL_ORDER)
    )

    print(
        f"\n--- Median Voxel-level {metric.title()} "
        "per Dataset and Model [Q1, Q3] ---"
    )
    print(median_iqr_table.to_string())


# -------------------------------------------------------------------
# Plot settings
# -------------------------------------------------------------------

sns.set_theme(style="whitegrid")


# -------------------------------------------------------------------
# Per-case Dice, precision, and recall distributions per dataset
# -------------------------------------------------------------------

for metric, metric_label in METRIC_LABELS.items():
    plt.figure(figsize=(11, 6))

    sns.boxplot(
        data=df_dice,
        x="source",
        y=metric,
        hue="model",
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
    )

    plt.title(f"{metric_label} by Dataset and Model")
    plt.xlabel("Dataset")
    plt.ylabel(f"Per-case {metric_label}")
    plt.ylim(0, 1)

    plt.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Mean Dice per dataset with values and 95% CI
# Seaborn uses bootstrap confidence intervals for this plot.
# -------------------------------------------------------------------

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    data=df_dice,
    x="source",
    y="dice",
    hue="model",
    hue_order=MODEL_ORDER,
    estimator="mean",
    errorbar=("ci", 95),
    seed=42,
    capsize=0.1,
    palette=MODEL_PALETTE,
)

for container in ax.containers:
    if hasattr(container, "datavalues"):
        ax.bar_label(
            container,
            fmt="%.3f",
            padding=8,
            fontsize=8,
            rotation=90,
        )

plt.title("Mean Dice Score by Dataset and Model")
plt.xlabel("Dataset")
plt.ylabel("Mean Per-case Dice")
plt.ylim(0, 1.12)

plt.legend(
    title="Model",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------
# Overall average Dice stored in the nnU-Net summary files
# -------------------------------------------------------------------

summary_plot = summary_file_dice.reset_index()

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    data=summary_plot,
    x="model",
    y="summary_average_dice",
    order=MODEL_ORDER,
    hue="model",
    hue_order=MODEL_ORDER,
    palette=MODEL_PALETTE,
    legend=False,
)

for container in ax.containers:
    if hasattr(container, "datavalues"):
        ax.bar_label(
            container,
            fmt="%.3f",
            padding=4,
            fontsize=9,
        )

plt.title("Overall Average Dice per Model")
plt.xlabel("Model")
plt.ylabel("Average Dice from nnU-Net Summary")
plt.ylim(0, 1.05)

plt.tight_layout()
plt.show()