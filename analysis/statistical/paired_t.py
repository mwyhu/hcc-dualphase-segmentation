from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel


BASE_DIR = Path(
    "/Users/michellehu/Desktop/"
    "hcc-dualphase-segmentation/analysis/statistical"
)

OUTPUT_DIR = BASE_DIR / "scheduler_comparison_ttest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_THRESHOLD = 0.20
METRICS = ["dice", "detectability", "precision"]


FILES = {
    "Polynomial": {
        "dice": {
            0: BASE_DIR / "summary_fold1-4/Poly_DP/DP_1e3_fold0_summary.json",
            1: BASE_DIR / "summary_fold1-4/Poly_DP/DP_1e3_fold1_summary.json",
            2: BASE_DIR / "summary_fold1-4/Poly_DP/DP_1e3_fold2_summary.json",
            3: BASE_DIR / "summary_fold1-4/Poly_DP/DP_1e3_fold3_summary.json",
            4: BASE_DIR / "summary_fold1-4/Poly_DP/DP_1e3_fold4_summary.json",
        },
        "detectability": {
            0: BASE_DIR / "detectability_fold_1-4/Poly_DP/poly_DP_fold0.csv",
            1: BASE_DIR / "detectability_fold_1-4/Poly_DP/poly_DP_fold1.csv",
            2: BASE_DIR / "detectability_fold_1-4/Poly_DP/poly_DP_fold2.csv",
            3: BASE_DIR / "detectability_fold_1-4/Poly_DP/poly_DP_fold3.csv",
            4: BASE_DIR / "detectability_fold_1-4/Poly_DP/poly_DP_fold4.csv",
        },
    },
    "Cosine": {
        "dice": {
            0: BASE_DIR / "summary_fold1-4/CA_DP/DP_CA_1e3_fold0_summary.json",
            1: BASE_DIR / "summary_fold1-4/CA_DP/DP_CA_1e3_fold1_summary.json",
            2: BASE_DIR / "summary_fold1-4/CA_DP/DP_CA_1e3_fold2_summary.json",
            3: BASE_DIR / "summary_fold1-4/CA_DP/DP_CA_1e3_fold3_summary.json",
            4: BASE_DIR / "summary_fold1-4/CA_DP/DP_CA_1e3_fold4_summary.json",
        },
        "detectability": {
            0: BASE_DIR / "detectability_fold_1-4/CA_DP/CA_DP_fold0.csv",
            1: BASE_DIR / "detectability_fold_1-4/CA_DP/CA_DP_fold1.csv",
            2: BASE_DIR / "detectability_fold_1-4/CA_DP/CA_DP_fold2.csv",
            3: BASE_DIR / "detectability_fold_1-4/CA_DP/CA_DP_fold3.csv",
            4: BASE_DIR / "detectability_fold_1-4/CA_DP/CA_DP_fold4.csv",
        },
    },
}


# Loading and validation
def remove_nifti_extension(filename):
    """
    Return the filename without .nii or .nii.gz
    """
    filename = Path(filename).name

    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]

    return Path(filename).stem


def get_dataset(case_id):
    """
    Determine the source dataset from the case identifier
    """
    prefixes = ["Liver_Lesions", "MCT_LTDiag", "WAW_TACE", "MSD"]

    for prefix in prefixes:
        if case_id.startswith(prefix):
            return prefix

    return "Unknown"


def check_files_exist():
    """
    Check all configured input paths before running the analysis
    """
    missing = []

    for scheduler, scheduler_files in FILES.items():
        for file_type, fold_files in scheduler_files.items():
            for fold, path in fold_files.items():
                if not path.exists():
                    missing.append(
                        f"{scheduler}, {file_type}, fold {fold}: {path}"
                    )

    if missing:
        raise FileNotFoundError(
            "The following input files were not found:\n"
            + "\n".join(missing)
        )


def load_dice_file(path, fold, scheduler):
    """
    Load per-case Dice scores from an nnU-Net summary JSON
    """
    with path.open() as file:
        data = json.load(file)

    records = []

    for entry in data["metric_per_case"]:
        case_id = remove_nifti_extension(entry["reference_file"])

        records.append(
            {
                "case_id": case_id,
                "dataset": get_dataset(case_id),
                "fold": fold,
                "scheduler": scheduler,
                "dice": float(entry["metrics"]["1"]["Dice"]),
            }
        )

    result = pd.DataFrame(records)

    if result["case_id"].duplicated().any():
        raise ValueError(f"Duplicate Dice cases found in {path}")

    return result


def load_detection_file(path, fold, scheduler):
    """
    Load per-case lesion metrics at IoU threshold
    """
    data = pd.read_csv(path)

    required = {
        "case",
        "iou_threshold",
        "precision",
        "detectability",
    }
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    data = data[
        np.isclose(
            data["iou_threshold"].astype(float),
            IOU_THRESHOLD,
        )
    ].copy()

    if data.empty:
        raise ValueError(f"No IoU {IOU_THRESHOLD} rows found in {path}")

    data["case_id"] = data["case"].apply(remove_nifti_extension)
    data["precision"] = pd.to_numeric(data["precision"], errors="coerce")
    data["detectability"] = pd.to_numeric(
        data["detectability"],
        errors="coerce",
    )

    if data["case_id"].duplicated().any():
        raise ValueError(f"Duplicate cases found in {path}")

    result = data[
        ["case_id", "precision", "detectability"]
    ].copy()
    result["fold"] = fold
    result["scheduler"] = scheduler

    return result


def load_scheduler(scheduler):
    """
    Load and combine all validation folds per scheduler
    """
    fold_results = []

    for fold in range(5):
        dice = load_dice_file(
            FILES[scheduler]["dice"][fold],
            fold,
            scheduler,
        )
        detection = load_detection_file(
            FILES[scheduler]["detectability"][fold],
            fold,
            scheduler,
        )

        if set(dice["case_id"]) != set(detection["case_id"]):
            raise ValueError(
                f"Dice and detection cases do not match for "
                f"{scheduler}, fold {fold}."
            )

        merged = dice.merge(
            detection,
            on=["case_id", "fold", "scheduler"],
            how="inner",
            validate="one_to_one",
        )
        fold_results.append(merged)

    result = pd.concat(fold_results, ignore_index=True)

    if result["case_id"].duplicated().any():
        raise ValueError(
            f"At least one {scheduler} case occurs in multiple folds."
        )

    return result


def pair_schedulers(polynomial, cosine):
    """
    Pair results by case ID, dataset, and fold
    """
    keys = ["case_id", "dataset", "fold"]

    if set(map(tuple, polynomial[keys].to_numpy())) != set(
        map(tuple, cosine[keys].to_numpy())
    ):
        raise ValueError(
            "Polynomial and cosine cases or folds do not match."
        )

    return polynomial[keys + METRICS].merge(
        cosine[keys + METRICS],
        on=keys,
        how="inner",
        suffixes=("_poly", "_cosine"),
        validate="one_to_one",
    )


# Statistical analysis
def paired_mean_confidence_interval(differences, confidence=0.95):
    """
    Parametric CI for the mean paired difference
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = len(differences)

    if n < 2:
        return np.nan, np.nan

    mean_difference = differences.mean()
    standard_error = differences.std(ddof=1) / np.sqrt(n)
    alpha = 1.0 - confidence
    critical_value = t.ppf(1.0 - alpha / 2.0, df=n - 1)
    margin = critical_value * standard_error

    return mean_difference - margin, mean_difference + margin


def holm_correction(p_values):
    """
    Holm's family-wise multiple-testing correction
    """
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid_indices = np.where(np.isfinite(p_values))[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = p_values[valid_indices]
    order = np.argsort(valid_p)
    ordered_p = valid_p[order]
    ordered_adjusted = np.empty(len(ordered_p))
    running_maximum = 0.0

    for rank, p_value in enumerate(ordered_p):
        corrected = (len(ordered_p) - rank) * p_value
        running_maximum = max(running_maximum, corrected)
        ordered_adjusted[rank] = min(running_maximum, 1.0)

    reverse_order = np.empty(len(order), dtype=int)
    reverse_order[order] = np.arange(len(order))
    adjusted[valid_indices] = ordered_adjusted[reverse_order]

    return adjusted


def create_paired_ttest_results(paired):
    """
    Run paired t-tests for Dice, detectability, and precision
    """
    rows = []

    for metric in METRICS:
        poly_column = f"{metric}_poly"
        cosine_column = f"{metric}_cosine"

        metric_data = paired[
            ["case_id", "dataset", "fold", poly_column, cosine_column]
        ].dropna(subset=[poly_column, cosine_column])

        poly_values = metric_data[poly_column].to_numpy(dtype=float)
        cosine_values = metric_data[cosine_column].to_numpy(dtype=float)
        differences = cosine_values - poly_values

        test = ttest_rel(cosine_values, poly_values)
        ci_lower, ci_upper = paired_mean_confidence_interval(differences)

        rows.append(
            {
                "metric": metric,
                "iou_threshold": (
                    np.nan if metric == "dice" else IOU_THRESHOLD
                ),
                "n_paired_cases": len(metric_data),
                "polynomial_mean": poly_values.mean(),
                "polynomial_sd": poly_values.std(ddof=1),
                "cosine_mean": cosine_values.mean(),
                "cosine_sd": cosine_values.std(ddof=1),
                "mean_difference_cosine_minus_polynomial": differences.mean(),
                "sd_paired_difference": differences.std(ddof=1),
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "t_statistic": test.statistic,
                "paired_t_test_p_value": test.pvalue,
                "cosine_better_n": int(np.sum(differences > 0)),
                "polynomial_better_n": int(np.sum(differences < 0)),
                "equal_n": int(np.sum(differences == 0)),
            }
        )

    results = pd.DataFrame(rows)
    results["holm_adjusted_p_value"] = holm_correction(
        results["paired_t_test_p_value"].to_numpy()
    )
    results["significant_after_holm"] = (
        results["holm_adjusted_p_value"] < 0.05
    )

    return results



# Summaries
def create_fold_summary(all_data):
    """
    Calculate means within folds and mean +- SD across folds
    """
    fold_rows = []

    for (scheduler, fold), group in all_data.groupby(
        ["scheduler", "fold"]
    ):
        for metric in METRICS:
            values = group[metric].dropna()
            fold_rows.append(
                {
                    "scheduler": scheduler,
                    "fold": fold,
                    "metric": metric,
                    "n_valid": len(values),
                    "mean": values.mean(),
                    "sd_within_fold": values.std(ddof=1),
                }
            )

    fold_summary = pd.DataFrame(fold_rows)
    across_folds = (
        fold_summary
        .groupby(["scheduler", "metric"], as_index=False)
        .agg(
            n_folds=("mean", "count"),
            mean_of_fold_means=("mean", "mean"),
            sd_across_fold_means=("mean", "std"),
            minimum_fold_mean=("mean", "min"),
            maximum_fold_mean=("mean", "max"),
        )
    )

    return fold_summary, across_folds


def create_dataset_summary(all_data):
    """
    Calc descriptive results per source dataset
    """
    rows = []

    for (scheduler, dataset), group in all_data.groupby(
        ["scheduler", "dataset"]
    ):
        for metric in METRICS:
            values = group[metric].dropna()
            rows.append(
                {
                    "scheduler": scheduler,
                    "dataset": dataset,
                    "metric": metric,
                    "n_valid": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                }
            )

    return pd.DataFrame(rows)



def main():
    check_files_exist()

    polynomial = load_scheduler("Polynomial")
    cosine = load_scheduler("Cosine")
    paired = pair_schedulers(polynomial, cosine)
    all_data = pd.concat([polynomial, cosine], ignore_index=True)

    for metric in METRICS:
        paired[f"{metric}_difference"] = (
            paired[f"{metric}_cosine"]
            - paired[f"{metric}_poly"]
        )

    statistical_results = create_paired_ttest_results(paired)
    fold_summary, across_folds = create_fold_summary(all_data)
    dataset_summary = create_dataset_summary(all_data)

    statistical_results.to_csv(
        OUTPUT_DIR / "paired_ttest_results.csv",
        index=False,
    )
    paired.to_csv(
        OUTPUT_DIR / "paired_case_level_results.csv",
        index=False,
    )
    fold_summary.to_csv(
        OUTPUT_DIR / "fold_level_summary.csv",
        index=False,
    )
    across_folds.to_csv(
        OUTPUT_DIR / "summary_across_folds.csv",
        index=False,
    )
    dataset_summary.to_csv(
        OUTPUT_DIR / "dataset_specific_summary.csv",
        index=False,
    )

    display_columns = [
        "metric",
        "n_paired_cases",
        "polynomial_mean",
        "cosine_mean",
        "mean_difference_cosine_minus_polynomial",
        "ci_95_lower",
        "ci_95_upper",
        "t_statistic",
        "paired_t_test_p_value",
        "holm_adjusted_p_value",
    ]

    print("\nPAIRED OUT-OF-FOLD T-TEST COMPARISON")
    print(f"Lesion matching threshold: IoU >= {IOU_THRESHOLD}")
    print(
        statistical_results[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nMEAN ± SD ACROSS THE FIVE FOLD MEANS")
    print(across_folds.round(4).to_string(index=False))
    print(f"\nResults saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()