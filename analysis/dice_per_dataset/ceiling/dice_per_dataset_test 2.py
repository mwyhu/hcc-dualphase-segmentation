from pathlib import Path
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Paths to nnU-Net evaluation JSON files
# -------------------------------------------------------------------

BASE_DIR = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis"
)

DICE_DIR = BASE_DIR / "dice_per_dataset" / "ceiling"
DETECT_DIR = BASE_DIR / "detectability" / "ceiling"

DICE_FILES = {
    "SP": DICE_DIR / "SP_on_ceiling_testset.json",
    "SP ceiling": DICE_DIR / "SPCeiling_on_ceiling_testset.json",
    "DP": DICE_DIR / "DP_on_ceiling_testset.json",
    "DP ceiling": DICE_DIR / "DPCeiling_on_ceiling_testset.json",
}

MODEL_ORDER = ["SP", "SP ceiling", "DP", "DP ceiling"]

MODEL_PALETTE = {
    "SP": "#B279A2",
    "SP ceiling": "#D9B9D0",
    "DP": "#E45756",
    "DP ceiling": "#F2AAA9",
}

METRIC_LABELS = {
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Sensitivity",
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

def format_mean_sd(row, metric):
    """Format the mean and sample standard deviation as mean ± SD."""
    mean = row[f"mean_{metric}"]
    sd = row[f"std_{metric}"]

    if pd.isna(mean):
        return "NaN"

    sd_text = "NaN" if pd.isna(sd) else f"{sd:.4f}"

    return f"{mean:.4f} ± {sd_text}"


def format_median_iqr(row, metric):
    """Format median and interquartile interval as median [Q1, Q3]."""
    median = row[f"median_{metric}"]
    q1 = row[f"q1_{metric}"]
    q3 = row[f"q3_{metric}"]

    if pd.isna(median):
        return "NaN"

    return f"{median:.4f} [{q1:.4f}, {q3:.4f}]"


def format_summary_for_print(summary, metrics):
    """Display mean ± SD and median [Q1, Q3]; retain numeric source data."""
    formatted = summary.copy()

    for metric in metrics:
        formatted[f"mean_{metric}"] = summary.apply(
            lambda row: format_mean_sd(row, metric),
            axis=1,
        )

        formatted[f"median_{metric}"] = summary.apply(
            lambda row: format_median_iqr(row, metric),
            axis=1,
        )

        formatted = formatted.drop(
            columns=[
                f"std_{metric}",
                f"q1_{metric}",
                f"q3_{metric}",
            ]
        )

        label = "sensitivity" if metric == "recall" else metric

        formatted = formatted.rename(columns={
            f"mean_{metric}": f"{label}: Mean ± SD",
            f"median_{metric}": f"{label}: Median [Q1, Q3]",
        })

    return formatted.round(4)


# -------------------------------------------------------------------
# Read per-case Dice, TP, FP, FN, precision, and sensitivity
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

            # Undefined when no positive voxels are predicted.
            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else float("nan")
            )

            # Undefined when the ground truth contains no tumour voxels.
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
# Precision and sensitivity summary helper
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
            std_dice=("dice", "std"),
            median_dice=("dice", "median"),
            q1_dice=("dice", lambda x: x.quantile(0.25)),
            q3_dice=("dice", lambda x: x.quantile(0.75)),
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

print(
    "\nTable format: mean ± SD and median [Q1, Q3]. "
    "Undefined values are excluded per metric."
)


# -------------------------------------------------------------------
# Precision and sensitivity per model
# -------------------------------------------------------------------

overall_precision_recall = summarise_precision_recall(
    df_dice,
    ["model"],
)

print(
    "\n--- Overall Voxel-level Precision and Sensitivity per Model "
    "(Mean ± SD; Median [Q1, Q3]) ---"
)

print(
    format_summary_for_print(
        overall_precision_recall,
        ["precision", "recall"],
    ).to_string()
)


# -------------------------------------------------------------------
# Precision and sensitivity per dataset and model
# -------------------------------------------------------------------

dataset_precision_recall = summarise_precision_recall(
    df_dice,
    ["source", "model"],
)

print(
    "\n--- Voxel-level Precision and Sensitivity per Dataset and Model "
    "(Mean ± SD; Median [Q1, Q3]) ---"
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

print(
    "\n--- Dice per Dataset and Model "
    "(Mean ± SD; Median [Q1, Q3]) ---"
)

print(
    format_summary_for_print(
        dice_summary,
        ["dice"],
    ).to_string()
)


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

print(
    "\n--- Overall Dice per Model "
    "(Mean ± SD; Median [Q1, Q3]) ---"
)

print(
    format_summary_for_print(
        overall_dice_summary,
        ["dice"],
    ).to_string()
)

print("\n--- Average Dice Stored in nnU-Net Summary Files ---")
print(summary_file_dice.round(4).to_string())


# -------------------------------------------------------------------
# Overall Dice, precision, and sensitivity comparison
# -------------------------------------------------------------------

comparison_table = (
    overall_dice_summary[
        [
            "n_cases",
            "n_cases_with_dice",
            "mean_dice",
            "std_dice",
            "median_dice",
            "q1_dice",
            "q3_dice",
            "ci95_lower",
            "ci95_upper",
        ]
    ]
    .join(
        overall_precision_recall.drop(columns="n_cases")
    )
)

print(
    "\n--- Dice, Voxel-level Precision and Sensitivity per Model "
    "(Mean ± SD; Median [Q1, Q3]) ---"
)

print(
    format_summary_for_print(
        comparison_table,
        ["dice", "precision", "recall"],
    ).to_string()
)


# -------------------------------------------------------------------
# Mean ± SD and median [Q1, Q3] tables per dataset and model
# -------------------------------------------------------------------

for metric, metric_label in METRIC_LABELS.items():
    summary = (
        df_dice
        .groupby(["source", "model"], observed=True)
        .agg(
            mean=(metric, "mean"),
            sd=(metric, "std"),
            median=(metric, "median"),
            q1=(metric, lambda x: x.quantile(0.25)),
            q3=(metric, lambda x: x.quantile(0.75)),
        )
        .rename(columns={
            "mean": f"mean_{metric}",
            "sd": f"std_{metric}",
            "median": f"median_{metric}",
            "q1": f"q1_{metric}",
            "q3": f"q3_{metric}",
        })
    )

    mean_table = (
        summary.apply(
            lambda row: format_mean_sd(row, metric),
            axis=1,
        )
        .unstack("model")
        .reindex(columns=MODEL_ORDER)
    )

    median_table = (
        summary.apply(
            lambda row: format_median_iqr(row, metric),
            axis=1,
        )
        .unstack("model")
        .reindex(columns=MODEL_ORDER)
    )

    print(
        f"\n--- Mean {metric_label} per Dataset and Model "
        "(Mean ± SD) ---"
    )
    print(mean_table.to_string())

    print(
        f"\n--- Median {metric_label} per Dataset and Model "
        "(Median [Q1, Q3]) ---"
    )
    print(median_table.to_string())


# -------------------------------------------------------------------
# Plot settings
# -------------------------------------------------------------------

sns.set_theme(style="whitegrid")


# -------------------------------------------------------------------
# Per-case Dice, precision, and sensitivity distributions per dataset
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