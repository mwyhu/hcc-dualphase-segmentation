from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# -------------------------------------------------------------------
# Paths to nnU-Net evaluation JSON files
# -------------------------------------------------------------------

DICE_FILES = {
    "fold0": (
        "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
        "analysis/dice_per_dataset/fold1-4/"
        "Poly_DP/"
        "DP_1e3_fold0_summary.json"
    ),
    "fold1": (
        "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
        "analysis/dice_per_dataset/fold1-4/"
        "Poly_DP/"
        "DP_1e3_fold1_summary.json"
    ),
    "fold2": (
        "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
        "analysis/dice_per_dataset/fold1-4/"
        "Poly_DP/"
        "DP_1e3_fold2_summary.json"
    ),
    "fold3": (
        "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
        "analysis/dice_per_dataset/fold1-4/"
        "Poly_DP/"
        "DP_1e3_fold3_summary.json"
    ),
    "fold4": (
        "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
        "analysis/dice_per_dataset/fold1-4/"
        "Poly_DP/"
        "DP_1e3_fold4_summary.json"
    ),
}

#
# DICE_FILES = {
#     "fold0": (
#         "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
#         "analysis/dice_per_dataset/fold1-4/"
#         "CA_DP/"
#         "DP_CA_1e3_fold0_summary.json"
#     ),
#     "fold1": (
#         "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
#         "analysis/dice_per_dataset/fold1-4/"
#         "CA_DP/"
#         "DP_CA_1e3_fold1_summary.json"
#     ),
#     "fold2": (
#         "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
#         "analysis/dice_per_dataset/fold1-4/"
#         "CA_DP/"
#         "DP_CA_1e3_fold2_summary.json"
#     ),
#     "fold3": (
#         "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
#         "analysis/dice_per_dataset/fold1-4/"
#         "CA_DP/"
#         "DP_CA_1e3_fold3_summary.json"
#     ),
#     "fold4": (
#         "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
#         "analysis/dice_per_dataset/fold1-4/"
#         "CA_DP/"
#         "DP_CA_1e3_fold4_summary.json"
#     ),
# }

MODEL_ORDER = [
    "fold0",
    "fold1",
    "fold2",
    "fold3",
    "fold4",
]

MODEL_PALETTE = {
    "fold0": "#C44E52",
    "fold1": "#8172B2",
    "fold2": "#FF3366",
    "fold3": "#20A4F3",
    "fold4": "#D496A7",
}

AVERAGE_COLOR = "#4C4C4C"


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
        "Liver_Lesions",
        "MCT_LTDiag",
        "WAW_TACE",
        "MSD08",
    ]

    for dataset in dataset_prefixes:
        if case_name.startswith(dataset):
            return dataset

    return case_name.split("_")[0]


# -------------------------------------------------------------------
# Extract average Dice stored in an nnU-Net summary file
# -------------------------------------------------------------------

def extract_summary_average_dice(results, json_path):
    """
    Support different nnU-Net summary JSON formats.
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


# -------------------------------------------------------------------
# Read per-case Dice and summary-average Dice
# -------------------------------------------------------------------

def load_dice_results(dice_files):
    case_rows = []
    summary_rows = []

    for fold, json_path in dice_files.items():
        with open(json_path, "r") as file:
            results = json.load(file)

        # Average Dice stored directly in the nnU-Net summary file
        summary_average_dice = extract_summary_average_dice(
            results,
            json_path,
        )

        summary_rows.append({
            "model": fold,
            "summary_average_dice": summary_average_dice,
        })

        # Per-case Dice scores
        for case_result in results["metric_per_case"]:
            case_name = (
                Path(case_result["prediction_file"])
                .name
                .removesuffix(".nii.gz")
                .removesuffix(".nii")
            )

            dice = case_result["metrics"]["1"]["Dice"]

            case_rows.append({
                "case": case_name,
                "source": get_dataset(case_name),
                "model": fold,
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


# -------------------------------------------------------------------
# Dice summary per dataset and fold
# -------------------------------------------------------------------

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

# Standard error within each dataset and fold
dice_summary["standard_error"] = (
    dice_summary["std_dice"]
    / dice_summary["n_cases"] ** 0.5
)

# Normal-approximation 95% CI within each dataset and fold
dice_summary["ci95_lower"] = (
    dice_summary["mean_dice"]
    - 1.96 * dice_summary["standard_error"]
).clip(lower=0)

dice_summary["ci95_upper"] = (
    dice_summary["mean_dice"]
    + 1.96 * dice_summary["standard_error"]
).clip(upper=1)

dice_summary = dice_summary.round(4)

print("\n--- Dice per Dataset and Fold ---")
print(dice_summary.to_string())


# -------------------------------------------------------------------
# Overall Dice per fold
# -------------------------------------------------------------------

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

# Add average Dice stored in each summary JSON file
overall_dice_summary = overall_dice_summary.join(
    summary_file_dice
)

# Check whether manually calculated and JSON Dice values agree
overall_dice_summary["difference"] = (
    overall_dice_summary["mean_dice"]
    - overall_dice_summary["summary_average_dice"]
)

overall_dice_summary = overall_dice_summary.round(4)

print("\n--- Overall Dice per Fold ---")
print(overall_dice_summary.to_string())


print("\n--- Average Dice Stored in nnU-Net Summary Files ---")
print(summary_file_dice.round(4).to_string())


# -------------------------------------------------------------------
# Mean Dice per dataset and fold
# -------------------------------------------------------------------

dice_print_table = (
    df_dice
    .groupby(
        ["source", "model"],
        observed=True,
    )["dice"]
    .mean()
    .unstack("model")
    .reindex(columns=MODEL_ORDER)
)


# -------------------------------------------------------------------
# Average and SD across folds per dataset
# -------------------------------------------------------------------
#
# Each entry in dice_print_table is the mean Dice for one dataset
# in one fold.
#
# The following mean and SD therefore describe variation between the
# five fold-level dataset means.
# -------------------------------------------------------------------

dice_print_table["mean_over_folds"] = (
    dice_print_table[MODEL_ORDER]
    .mean(axis=1)
)

dice_print_table["sd_over_folds"] = (
    dice_print_table[MODEL_ORDER]
    .std(axis=1, ddof=1)
)

print("\n--- Mean Dice per Dataset, Fold, and Across Folds ---")
print(dice_print_table.round(4).to_string())


# Separate summary table
dataset_fold_summary = (
    dice_print_table[
        ["mean_over_folds", "sd_over_folds"]
    ]
    .copy()
)

print("\n--- Dataset Mean ± SD Across Folds ---")

for dataset, row in dataset_fold_summary.iterrows():
    print(
        f"{dataset}: "
        f"{row['mean_over_folds']:.4f} "
        f"± {row['sd_over_folds']:.4f}"
    )


# -------------------------------------------------------------------
# Overall average and SD across folds
# -------------------------------------------------------------------

fold_summary_values = (
    summary_file_dice["summary_average_dice"]
    .astype(float)
)

overall_average_over_folds = fold_summary_values.mean()

# Sample SD across the five folds
overall_sd_over_folds = fold_summary_values.std(ddof=1)

# Standard error of the fold mean
overall_se_over_folds = (
    overall_sd_over_folds
    / len(fold_summary_values) ** 0.5
)

print("\n--- Overall Mean ± SD Across Folds ---")
print(
    f"{overall_average_over_folds:.4f} "
    f"± {overall_sd_over_folds:.4f}"
)

print(
    f"Standard error across folds: "
    f"{overall_se_over_folds:.4f}"
)


# -------------------------------------------------------------------
# Pooled out-of-fold Dice
# -------------------------------------------------------------------
#
# Every validation case occurs in one fold. Combining all fold
# predictions therefore gives the pooled out-of-fold performance.
# -------------------------------------------------------------------

pooled_cv_summary = pd.Series({
    "n_cases": df_dice["case"].nunique(),
    "mean_dice": df_dice["dice"].mean(),
    "median_dice": df_dice["dice"].median(),
    "std_dice": df_dice["dice"].std(ddof=1),
})

print("\n--- Pooled Out-of-Fold Dice Across All Cases ---")
print(pooled_cv_summary.round(4).to_string())


print("\n--- Final Five-Fold Summary ---")
print(
    "Unweighted mean ± SD across folds: "
    f"{overall_average_over_folds:.4f} "
    f"± {overall_sd_over_folds:.4f}"
)
print(
    "Pooled mean across all out-of-fold cases: "
    f"{pooled_cv_summary['mean_dice']:.4f}"
)


# -------------------------------------------------------------------
# Plot settings
# -------------------------------------------------------------------

sns.set_theme(style="whitegrid")


# -------------------------------------------------------------------
# Per-case Dice distributions per dataset and fold
# -------------------------------------------------------------------

plt.figure(figsize=(11, 6))

sns.boxplot(
    data=df_dice,
    x="source",
    y="dice",
    hue="model",
    hue_order=MODEL_ORDER,
    palette=MODEL_PALETTE,
)

plt.title("Dice Score by Dataset and Fold")
plt.xlabel("Dataset")
plt.ylabel("Per-case Dice")
plt.ylim(0, 1)

plt.legend(
    title="Fold",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------
# Mean Dice per dataset and fold with 95% CI
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

plt.title("Mean Dice Score by Dataset and Fold")
plt.xlabel("Dataset")
plt.ylabel("Mean Per-case Dice")
plt.ylim(0, 1.12)

plt.legend(
    title="Fold",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------
# Mean Dice across folds per dataset with fold SD
# -------------------------------------------------------------------

dataset_plot = (
    dataset_fold_summary
    .reset_index()
)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=dataset_plot,
    x="source",
    y="mean_over_folds",
    color=AVERAGE_COLOR,
    errorbar=None,
)

ax.errorbar(
    x=range(len(dataset_plot)),
    y=dataset_plot["mean_over_folds"],
    yerr=dataset_plot["sd_over_folds"],
    fmt="none",
    ecolor="black",
    capsize=5,
    linewidth=1.2,
)

for container in ax.containers:
    if hasattr(container, "datavalues"):
        ax.bar_label(
            container,
            fmt="%.3f",
            padding=4,
            fontsize=9,
        )

plt.title("Mean Dice Across Folds per Dataset")
plt.xlabel("Dataset")
plt.ylabel("Mean Dice Across Folds")
plt.ylim(0, 1.05)

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------
# Overall average Dice per fold plus average across folds
# -------------------------------------------------------------------

summary_plot = (
    summary_file_dice
    .reset_index()
)

average_row = pd.DataFrame({
    "model": ["average"],
    "summary_average_dice": [
        overall_average_over_folds
    ],
})

summary_plot = pd.concat(
    [summary_plot, average_row],
    ignore_index=True,
)

plot_order = MODEL_ORDER + ["average"]

plot_palette = {
    **MODEL_PALETTE,
    "average": AVERAGE_COLOR,
}

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=summary_plot,
    x="model",
    y="summary_average_dice",
    order=plot_order,
    hue="model",
    hue_order=plot_order,
    palette=plot_palette,
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

# Add fold SD only to the average bar
average_position = plot_order.index("average")

ax.errorbar(
    x=average_position,
    y=overall_average_over_folds,
    yerr=overall_sd_over_folds,
    fmt="none",
    ecolor="black",
    capsize=5,
    linewidth=1.2,
)

plt.title("Overall Average Dice per Fold")
plt.xlabel("Fold")
plt.ylabel("Average Dice from nnU-Net Summary")
plt.ylim(0, 1.05)

plt.tight_layout()
plt.show()