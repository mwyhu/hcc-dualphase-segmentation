from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis/dice_per_dataset/final"
)

DATASETS = ["HCC_TACE", "TCIA_CRLM", "WORC_CRLM", "LiTS"]

MODELS = [
    "TotalSegmentator_liver_lesions",
    "TotalSegmentator_liver_tumor",
    # "COALA",
    "AtlasNet",
    "SinglePhase",
    "DualPhase",
]

MODEL_CODES = {
    "TotalSegmentator_liver_lesions": "TSLL",
    "TotalSegmentator_liver_tumor": "TSLT",
    # "COALA": "COALA",
    "AtlasNet": "AtlasNet",
    "SinglePhase": "SP",
    "DualPhase": "DP",
}

DATASET_CODES = {
    "HCC_TACE": "HCC",
    "TCIA_CRLM": "TCIA",
    "WORC_CRLM": "WORC",
    "LiTS": "LiTS",
}


# -> ROOT / "SP_HCC_summary.json"
DICE_FILES = {
    model: {
        dataset: ROOT / (
            f"{MODEL_CODES[model]}_{DATASET_CODES[dataset]}_summary.json"
        )
        for dataset in DATASETS
    }
    for model in MODELS
}


# -------------------------------------------------------------------
# Read nnU-Net summary JSON files
# -------------------------------------------------------------------

def case_id_from_path(path):
    """Get the case ID from a prediction filename."""
    name = Path(path).name
    return name.removesuffix(".nii.gz").removesuffix(".nii")


def load_results(file_map):
    rows = []

    missing_files = [
        (model, dataset, path)
        for model, dataset_files in file_map.items()
        for dataset, path in dataset_files.items()
        if not Path(path).is_file()
    ]

    if missing_files:
        details = "\n".join(
            f"  {model} / {dataset}: {path}"
            for model, dataset, path in missing_files
        )
        raise FileNotFoundError(
            f"Missing {len(missing_files)} summary file(s):\n{details}"
        )

    for model, dataset_files in file_map.items():
        for dataset, json_path in dataset_files.items():
            with Path(json_path).open() as file:
                results = json.load(file)

            if "metric_per_case" not in results:
                raise KeyError(
                    f"'metric_per_case' is missing from {json_path}"
                )

            for case_result in results["metric_per_case"]:
                metrics = case_result["metrics"]["1"]

                tp = float(metrics["TP"])
                fp = float(metrics["FP"])
                fn = float(metrics["FN"])

                # Voxel-level precision for this case
                precision = (
                    tp / (tp + fp)
                    if tp + fp > 0
                    else np.nan
                )

                # Voxel-level recall; undefined when ground truth is empty.
                recall = tp / (tp + fn) if tp + fn > 0 else np.nan

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "case": case_id_from_path(
                        case_result["prediction_file"]
                    ),
                    "dice": float(metrics["Dice"]),
                    "precision": precision,
                    "recall": recall,
                    "fn": fn,
                    "tp": tp,
                    "fp": fp,
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No per-case results were found.")

    duplicates = df.duplicated(
        subset=["dataset", "model", "case"],
        keep=False,
    )
    if duplicates.any():
        examples = df.loc[
            duplicates, ["dataset", "model", "case"]
        ].head(10)
        raise ValueError(
            "Duplicate dataset/model/case combinations:\n"
            f"{examples.to_string(index=False)}"
        )

    return df


# -------------------------------------------------------------------
# Check cases
# -------------------------------------------------------------------

def check_case_sets(df):
    reference_model = MODELS[0]

    for dataset in DATASETS:
        reference_cases = set(
            df.loc[
                (df["dataset"] == dataset)
                & (df["model"] == reference_model),
                "case",
            ]
        )

        for model in MODELS[1:]:
            model_cases = set(
                df.loc[
                    (df["dataset"] == dataset)
                    & (df["model"] == model),
                    "case",
                ]
            )

            if model_cases != reference_cases:
                missing = sorted(reference_cases - model_cases)
                extra = sorted(model_cases - reference_cases)

                raise ValueError(
                    f"Case mismatch for {dataset}, {model}.\n"
                    f"Missing compared with {reference_model}: "
                    f"{missing[:10]}"
                    f"{' ...' if len(missing) > 10 else ''}\n"
                    f"Extra compared with {reference_model}: "
                    f"{extra[:10]}"
                    f"{' ...' if len(extra) > 10 else ''}"
                )


# -------------------------------------------------------------------
# Calc Dice, precision and recall summaries
# -------------------------------------------------------------------

def summarise(grouped):
    summary = grouped.agg(
        n_cases=("case", "nunique"),
        n_precision_defined=("precision", "count"),
        n_recall_defined=("recall", "count"),
        mean_recall=("recall", "mean"),
        median_recall=("recall", "median"),
        std_recall=("recall", "std"),
        q1_recall=("recall", lambda x: x.quantile(0.25)),
        q3_recall=("recall", lambda x: x.quantile(0.75)),

        mean_dice=("dice", "mean"),
        median_dice=("dice", "median"),
        std_dice=("dice", "std"),
        q1_dice=("dice", lambda x: x.quantile(0.25)),
        q3_dice=("dice", lambda x: x.quantile(0.75)),

        mean_precision=("precision", "mean"),
        median_precision=("precision", "median"),
        std_precision=("precision", "std"),
        q1_precision=(
            "precision",
            lambda x: x.quantile(0.25),
        ),
        q3_precision=(
            "precision",
            lambda x: x.quantile(0.75),
        ),
    )

    summary["iqr_dice"] = (
        summary["q3_dice"] - summary["q1_dice"]
    )

    summary["iqr_precision"] = (
        summary["q3_precision"] - summary["q1_precision"]
    )

    summary["iqr_recall"] = summary["q3_recall"] - summary["q1_recall"]

    return summary[
        [
            "n_cases",
            "n_precision_defined",
            "n_recall_defined",
            "mean_recall",
            "std_recall",
            "median_recall",
            "q1_recall",
            "q3_recall",
            "iqr_recall",
            "mean_dice",
            "std_dice",
            "median_dice",
            "q1_dice",
            "q3_dice",
            "iqr_dice",
            "mean_precision",
            "std_precision",
            "median_precision",
            "q1_precision",
            "q3_precision",
            "iqr_precision",
        ]
    ]


# -------------------------------------------------------------------
# Analysis
# -------------------------------------------------------------------

df = load_results(DICE_FILES)
check_case_sets(df)

overall_table = summarise(
    df.groupby("model", sort=False)
).reindex(MODELS)

# Separate summary for each dataset and model.
per_dataset_table = summarise(
    df.groupby(["dataset", "model"], sort=False)
).reindex(
    pd.MultiIndex.from_product(
        [DATASETS, MODELS],
        names=["dataset", "model"],
    )
)

mean_dice_table = (
    per_dataset_table["mean_dice"]
    .unstack("model")
    .reindex(index=DATASETS, columns=MODELS)
)

median_dice_table = (
    per_dataset_table["median_dice"]
    .unstack("model")
    .reindex(index=DATASETS, columns=MODELS)
)

# -------------------------------------------------------------------
# Mean median for precision
# -------------------------------------------------------------------
mean_dice_sd_table = (
    per_dataset_table.apply(
        lambda row: f"{row['mean_dice']:.4f} ± {row['std_dice']:.4f}",
        axis=1,
    )
    .unstack("model")
    .reindex(index=DATASETS, columns=MODELS)
)

median_dice_iqr_table = (
    per_dataset_table.apply(
        lambda row: f"{row['median_dice']:.4f} [{row['q1_dice']:.4f}, {row['q3_dice']:.4f}]",
        axis=1,
    )
    .unstack("model")
    .reindex(index=DATASETS, columns=MODELS)
)


# Mean voxel-level precision ± SD per dataset and model
mean_precision_sd_table = (
    per_dataset_table.apply(
        lambda row: (
            f"{row['mean_precision']:.4f} ± "
            f"{row['std_precision']:.4f}"
        ),
        axis=1,
    )
    .unstack("model")
    .reindex(index=DATASETS, columns=MODELS)
)

# Median voxel-level precision [Q1, Q3] per dataset and model
median_precision_iqr_table = (
    per_dataset_table.apply(
        lambda row: (
            f"{row['median_precision']:.4f} "
            f"[{row['q1_precision']:.4f}, "
            f"{row['q3_precision']:.4f}]"
        ),
        axis=1,
    )
    .unstack("model")
    .reindex(index=DATASETS, columns=MODELS)
)


# Recall tables per dataset and model
mean_recall_table = per_dataset_table["mean_recall"].unstack("model").reindex(index=DATASETS, columns=MODELS)
median_recall_table = per_dataset_table["median_recall"].unstack("model").reindex(index=DATASETS, columns=MODELS)

mean_recall_sd_table = (
    per_dataset_table.apply(
        lambda row: f"{row['mean_recall']:.4f} ± {row['std_recall']:.4f}",
        axis=1,
    ).unstack("model").reindex(index=DATASETS, columns=MODELS)
)

median_recall_iqr_table = (
    per_dataset_table.apply(
        lambda row: (
            f"{row['median_recall']:.4f} "
            f"[{row['q1_recall']:.4f}, {row['q3_recall']:.4f}]"
        ),
        axis=1,
    ).unstack("model").reindex(index=DATASETS, columns=MODELS)
)


# -------------------------------------------------------------------
# Results
# -------------------------------------------------------------------

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print("\n--- Overall Dice, Voxel-Level Precision and Recall per Model ---")
print(overall_table.round(4).to_string())

print("\n--- Dice, Voxel-Level Precision and Recall per Dataset and Model ---")
print(per_dataset_table.round(4).to_string())

print("\n--- Mean Dice ± SD per Dataset and Model ---")
print(mean_dice_sd_table.to_string())

print("\n--- Median Dice [Q1, Q3] per Dataset and Model ---")
print(median_dice_iqr_table.to_string())

print("\n--- Mean Voxel-Level Precision ± SD per Dataset and Model ---")
print(mean_precision_sd_table.to_string())

print("\n--- Median Voxel-Level Precision [Q1, Q3] per Dataset and Model ---")
print(median_precision_iqr_table.to_string())

print("\n--- Mean Voxel-Level Recall ± SD per Dataset and Model ---")
print(mean_recall_sd_table.to_string())

print("\n--- Median Voxel-Level Recall [Q1, Q3] per Dataset and Model ---")
print(median_recall_iqr_table.to_string())

# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------

# overall_table.to_csv(ROOT / "overall_model_comparison.csv")
# per_dataset_table.to_csv(ROOT / "per_dataset_model_comparison.csv")
# mean_dice_table.to_csv(ROOT / "mean_dice_by_dataset.csv")
# median_dice_table.to_csv(ROOT / "median_dice_by_dataset.csv")
# mean_dice_sd_table.to_csv(ROOT / "mean_dice_sd_by_dataset.csv")
# median_dice_iqr_table.to_csv(ROOT / "median_dice_iqr_by_dataset.csv")
#
# mean_precision_sd_table.to_csv(
#     ROOT / "mean_precision_sd_by_dataset.csv"
# )
# median_precision_iqr_table.to_csv(
#     ROOT / "median_precision_iqr_by_dataset.csv"
# )
#
# mean_recall_table.to_csv(ROOT / "mean_recall_by_dataset.csv")
# median_recall_table.to_csv(ROOT / "median_recall_by_dataset.csv")
# mean_recall_sd_table.to_csv(ROOT / "mean_recall_sd_by_dataset.csv")
# median_recall_iqr_table.to_csv(ROOT / "median_recall_iqr_by_dataset.csv")
# print(f"\nTables saved in: {ROOT}")



# -------------------------------------------------------------------
# Plots per dataset
# -------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PLOT_DIR = ROOT / "plots_per_dataset"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = [MODEL_CODES[model] for model in MODELS]

MODEL_COLOURS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
]

for dataset in DATASETS:
    dataset_df = df.loc[df["dataset"] == dataset]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        sharey=True,
    )

    for ax, metric, title in zip(
        axes,
        ["dice", "precision", "recall"],
        ["Dice", "Voxel-level precision", "Voxel-level recall"],
    ):
        values = [
            dataset_df.loc[
                dataset_df["model"] == model,
                metric,
            ].dropna().to_numpy()
            for model in MODELS
        ]

        boxplot = ax.boxplot(
            values,
            positions=np.arange(1, len(MODELS) + 1),
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            medianprops={
                "color": "darkorange",
                "linewidth": 2,
            },
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            flierprops={
                "marker": ".",
                "markersize": 3,
                "alpha": 0.4,
                "markeredgecolor": "grey",
            },
        )

        for box, colour in zip(
            boxplot["boxes"],
            MODEL_COLOURS,
        ):
            box.set_facecolor(colour)
            box.set_alpha(0.75)

        ax.set_xticks(np.arange(1, len(MODELS) + 1))
        ax.set_xticklabels(MODEL_LABELS, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylim(-0.03, 1.09)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        # Number of cases with a defined metric
        for position, model_values in enumerate(values, start=1):
            ax.text(
                position,
                1.035,
                f"n={len(model_values)}",
                ha="center",
                va="center",
                fontsize=9,
            )

    axes[0].set_ylabel("Score")

    fig.suptitle(
        f"{dataset}: model comparison",
        fontsize=14,
        fontweight="bold",
    )

    fig.legend(
        handles=[
            Patch(
                facecolor="lightgrey",
                edgecolor="black",
                label="Box: Q1–Q3 (IQR)",
            ),
            plt.Line2D(
                [0],
                [0],
                color="darkorange",
                linewidth=2,
                label="Median",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.94])

    fig.savefig(
        PLOT_DIR / f"{dataset}_dice_precision_recall_boxplots.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

print(f"\nPlots saved in: {PLOT_DIR}")