from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis/dice_per_dataset/final"
)

DATASETS = ["HCC_TACE", "TCIA_CRLM", "WORC_CRLM", "LiTS", "CAIRO5"]

MODELS = [
    "TotalSegmentator_liver_lesions",
    "TotalSegmentator_liver_tumor",
    "COALA",
    "AtlasNet",
    "SinglePhase",
    "DualPhase",
]

MODEL_CODES = {
    "TotalSegmentator_liver_lesions": "TSLL",
    "TotalSegmentator_liver_tumor": "TSLT",
    "COALA": "COALA",
    "AtlasNet": "AtlasNet",
    "SinglePhase": "SP",
    "DualPhase": "DP",
}

DATASET_CODES = {
    "HCC_TACE": "HCC",
    "TCIA_CRLM": "TCIA",
    "WORC_CRLM": "WORC",
    "LiTS": "LiTS",
    "CAIRO5": "CAIRO5",
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
# Boxplots with individual scan points: one figure per metric
# -------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

MODEL_PALETTE = {
    "TotalSegmentator_liver_lesions": "#4C78A8",
    "TotalSegmentator_liver_tumor": "#F58518",
    "AtlasNet": "#54A24B",
    "SinglePhase": "#B279A2",
    "DualPhase": "#E45756",
    "COALA": "#72B7B2",
}

PLOT_DIR = ROOT / "plots_per_metric"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {
    "dice": "Dice",
    "precision": "Voxel-level precision",
    "recall": "Voxel-level recall",
}


def plot_metric_by_dataset(df, metric, ylabel):
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
        plot_df = df.loc[
            df["dataset"] == dataset
        ].dropna(subset=[metric])

        sns.boxplot(
            data=plot_df,
            x="model",
            y=metric,
            order=MODELS,
            hue="model",
            hue_order=MODELS,
            palette=MODEL_PALETTE,
            dodge=False,
            width=0.6,
            showfliers=False,
            ax=ax,
        )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        # Individual scan scores, including outliers.
        for position, model in enumerate(MODELS):
            scores = plot_df.loc[
                plot_df["model"] == model, metric
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

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_CODES[model] for model in MODELS])

        ax.set_title(DATASET_CODES[dataset])
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="x", visible=False)

    for ax in list(axes.flat)[len(DATASETS):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Per-case {ylabel.lower()}",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(
        PLOT_DIR / f"{metric}_boxplots_per_dataset.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


for metric, ylabel in METRICS.items():
    plot_metric_by_dataset(df, metric, ylabel)

print(f"\nPlots saved in: {PLOT_DIR}")


# -------------------------------------------------------------------
# Dice, precision and recall  per dataset
# -------------------------------------------------------------------
for dataset in DATASETS:
    dataset_df = df.loc[df["dataset"] == dataset].copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        sharey=True,
    )

    rng = np.random.default_rng(42)

    for ax, (metric, ylabel) in zip(axes, METRICS.items()):
        plot_df = dataset_df.dropna(subset=[metric])

        sns.boxplot(
            data=plot_df,
            x="model",
            y=metric,
            order=MODELS,
            hue="model",
            hue_order=MODELS,
            palette=MODEL_PALETTE,
            dodge=False,
            width=0.6,
            showfliers=False,
            ax=ax,
        )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        # Individual scan scores, including outliers.
        for position, model in enumerate(MODELS):
            scores = plot_df.loc[
                plot_df["model"] == model, metric
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

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(
            [MODEL_CODES[model] for model in MODELS],
            rotation=30,
            ha="right",
        )

        ax.set_title(ylabel)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Score")

    fig.suptitle(
        f"{DATASET_CODES[dataset]}: model comparison",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(
        PLOT_DIR / f"{dataset}_dice_precision_recall_boxplots.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)