from pathlib import Path
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Paths to nnU-Net evaluation JSON files
# -------------------------------------------------------------------

DICE_FILES = {
    "Start1": "/Users/michellehu/Desktop/hcc_active_learning/analysis/dice_per_dataset/summary_1channel.json",
    "Start2": "/Users/michellehu/Desktop/hcc_active_learning/analysis/dice_per_dataset/summary_2channel.json",
    "Ceiling": "/Users/michellehu/Desktop/hcc_active_learning/analysis/dice_per_dataset/summary_ceiling.json",
    "TS": "/Users/michellehu/Desktop/hcc_active_learning/analysis/dice_per_dataset/summary_TS.json",
}

MODEL_ORDER = [
    "Start1",
    "Start2",
    "Ceiling",
    "TS",
]

MODEL_PALETTE = {
    "Start1": "#4C72B0",
    "Start2": "#55A868",
    "Ceiling": "#C44E52",
    "TS": "#8172B2",
}


# -------------------------------------------------------------------
# Identify the dataset from the case filename
# -------------------------------------------------------------------

def get_dataset(case_name):
    """
    Determine the source dataset from the case filename.

    The longest or most specific prefixes should appear first.
    Adjust these prefixes if your case names differ.
    """
    dataset_prefixes = [
        "HCC_TACE",
        "TCIA_CRLM",
        "WAW_TACE",
        "WORC_CRLM",
        "LiTS",
        "MSD",
    ]

    for dataset in dataset_prefixes:
        if case_name.startswith(dataset):
            return dataset

    return case_name.split("_")[0]


# Extract average Dice stored in an nnU-Net summary file
def extract_summary_average_dice(results, json_path):
    """
    Supports the nnU-Net summary JSON formats
    """
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


# Read per-case Dice and summary-average Dice
def load_dice_results(dice_files):
    case_rows = []
    summary_rows = []

    for model, json_path in dice_files.items():
        with open(json_path, "r") as file:
            results = json.load(file)

        # Average Dice stored directly in the nnU-Net summary file
        summary_average_dice = extract_summary_average_dice(
            results,
            json_path,
        )

        summary_rows.append({
            "model": model,
            "summary_average_dice": summary_average_dice,
        })

        # Per-case Dice scores
        for case_result in results["metric_per_case"]:
            case_name = Path(
                case_result["prediction_file"]
            ).name.removesuffix(".nii.gz")

            dice = case_result["metrics"]["1"]["Dice"]

            case_rows.append({
                "case": case_name,
                "source": get_dataset(case_name),
                "model": model,
                "dice": dice,
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


df_dice, summary_file_dice = load_dice_results(DICE_FILES)


# Calculate Dice summary per dataset and model
dice_summary = (
    df_dice
    .groupby(
        ["source", "model"],
        observed=True,
    )
    .agg(
        n_cases=("case", "nunique"),
        mean_dice=("dice", "mean"),
        median_dice=("dice", "median"),
        std_dice=("dice", "std"),
    )
)

# Standard error
dice_summary["standard_error"] = (
    dice_summary["std_dice"]
    / dice_summary["n_cases"] ** 0.5
)

# Normal-approximation 95% confidence interval
dice_summary["ci95_lower"] = (
    dice_summary["mean_dice"]
    - 1.96 * dice_summary["standard_error"]
).clip(lower=0)

dice_summary["ci95_upper"] = (
    dice_summary["mean_dice"]
    + 1.96 * dice_summary["standard_error"]
).clip(upper=1)

dice_summary = dice_summary.round(4)

print("\n--- Dice per Dataset and Model ---")
print(dice_summary.to_string())


# Overall Dice per model
overall_dice_summary = (
    df_dice
    .groupby("model", observed=True)
    .agg(
        n_cases=("case", "nunique"),
        mean_dice=("dice", "mean"),
        median_dice=("dice", "median"),
        std_dice=("dice", "std"),
    )
)

overall_dice_summary["standard_error"] = (
    overall_dice_summary["std_dice"]
    / overall_dice_summary["n_cases"] ** 0.5
)

overall_dice_summary["ci95_lower"] = (
    overall_dice_summary["mean_dice"]
    - 1.96 * overall_dice_summary["standard_error"]
).clip(lower=0)

overall_dice_summary["ci95_upper"] = (
    overall_dice_summary["mean_dice"]
    + 1.96 * overall_dice_summary["standard_error"]
).clip(upper=1)

# Add the average Dice stored in each summary JSON file
overall_dice_summary = overall_dice_summary.join(
    summary_file_dice
)

overall_dice_summary["difference"] = (
    overall_dice_summary["mean_dice"]
    - overall_dice_summary["summary_average_dice"]
)

overall_dice_summary = overall_dice_summary.round(4)

print("\n--- Overall Dice per Model ---")
print(overall_dice_summary.to_string())


# Average dice
print("\n--- Average Dice Stored in nnU-Net Summary Files ---")
print(summary_file_dice.round(4).to_string())


# Mean dice per dataset and model
dice_print_table = (
    df_dice
    .groupby(
        ["source", "model"],
        observed=True,
    )["dice"]
    .mean()
    .unstack("model")
    .reindex(columns=MODEL_ORDER)
    .round(4)
)

print("\n--- Mean Dice per Dataset and Model ---")
print(dice_print_table.to_string())



# Per-case Dice distributions per dataset
sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 6))

sns.boxplot(
    data=df_dice,
    x="source",
    y="dice",
    hue="model",
    hue_order=MODEL_ORDER,
    palette=MODEL_PALETTE,
)

plt.title("Dice Score by Dataset and Model")
plt.xlabel("Dataset")
plt.ylabel("Per-case Dice")
plt.ylim(0, 1)

plt.legend(
    title="Model",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()
plt.show()


# Mean Dice per dataset with values and 95% CI
plt.figure(figsize=(12, 6))

ax = sns.barplot(
    data=df_dice,
    x="source",
    y="dice",
    hue="model",
    hue_order=MODEL_ORDER,
    estimator="mean",
    errorbar=("ci", 95),
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



# Overall average Dice
summary_plot = (
    summary_file_dice
    .reset_index()
)

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