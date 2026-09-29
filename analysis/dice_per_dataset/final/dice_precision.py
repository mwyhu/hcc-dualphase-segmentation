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
    "TCIA_CRLM": "TCIA_CRLM",
    "WORC_CRLM": "WORC_CRLM",
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

                # Voxel-level precision for this case
                precision = (
                    tp / (tp + fp)
                    if tp + fp > 0
                    else np.nan
                )

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "case": case_id_from_path(
                        case_result["prediction_file"]
                    ),
                    "dice": float(metrics["Dice"]),
                    "precision": precision,
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
# Calc Dice and precision summaries
# -------------------------------------------------------------------

def summarise(grouped):
    summary = grouped.agg(
        n_cases=("case", "nunique"),
        n_precision_defined=("precision", "count"),

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

    return summary[
        [
            "n_cases",
            "n_precision_defined",
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
# Results
# -------------------------------------------------------------------

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print("\n--- Overall Dice and Voxel-Level Precision per Model ---")
print(overall_table.round(4).to_string())

print("\n--- Dice and Voxel-Level Precision per Dataset and Model ---")
print(per_dataset_table.round(4).to_string())

print("\n--- Mean Dice per Dataset and Model ---")
print(mean_dice_table.round(4).to_string())

print("\n--- Median Dice per Dataset and Model ---")
print(median_dice_table.round(4).to_string())


# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------

# overall_table.to_csv(ROOT / "overall_model_comparison.csv")
# per_dataset_table.to_csv(ROOT / "per_dataset_model_comparison.csv")
# mean_dice_table.to_csv(ROOT / "mean_dice_by_dataset.csv")
# median_dice_table.to_csv(ROOT / "median_dice_by_dataset.csv")
#
# print(f"\nTables saved in: {ROOT}")