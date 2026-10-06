from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

BASE_DIR = Path(
    "/Users/michellehu/Desktop/"
    "hcc-dualphase-segmentation/analysis/statistical/internal_testset/"
)

OUTPUT_DIR = BASE_DIR / "single_vs_dual_phase_ttest"

IOU_THRESHOLD = 0.20
TUMOUR_LABEL = "1"

METRICS = ["dice", "precision", "recall", "detectability"]

FILES = {
    "SinglePhase": {
        "summary": BASE_DIR / "SP_summary.json",
        "detectability": BASE_DIR / "Detect_SP_internal.csv",
    },
    "DualPhase": {
        "summary": BASE_DIR / "DP_summary.json",
        "detectability": BASE_DIR / "Detect_DP_internal.csv",
    },
}


# -------------------------------------------------------------------
# Loading and validation
# -------------------------------------------------------------------

def remove_nifti_extension(filename):
    """Return the filename without .nii or .nii.gz."""
    filename = str(filename).replace("\\", "/").rsplit("/", 1)[-1]

    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]

    return filename


def get_dataset(case_id):
    """Determine the source dataset from the case identifier."""
    prefixes = [
        "Liver_Lesions",
        "MCT_LTDiag",
        "WAW_TACE",
        "HCC_TACE",
        "MSD",
    ]

    for prefix in prefixes:
        if case_id.startswith(prefix):
            return prefix

    return "Unknown"


def check_files_exist():
    """Check all configured input paths before running the analysis."""
    missing = []

    for phase, phase_files in FILES.items():
        for file_type, path in phase_files.items():
            if not path.is_file():
                missing.append(f"{phase}, {file_type}: {path}")

    if missing:
        raise FileNotFoundError(
            "The following input files were not found:\n"
            + "\n".join(missing)
        )


def validate_metric_values(data, metrics, path):
    """Allow missing values, but reject infinite or out-of-range scores."""
    for metric in metrics:
        values = data[metric].to_numpy(dtype=float)

        invalid = (
            np.isinf(values)
            | (np.isfinite(values) & ((values < 0) | (values > 1)))
        )

        if invalid.any():
            cases = data.loc[invalid, "case_id"].tolist()
            raise ValueError(
                f"Invalid {metric} values in {path}. "
                f"Expected scores between 0 and 1 or NaN. Cases: {cases}"
            )


def load_summary_file(path, phase):
    """Load per-case Dice, voxel precision and voxel recall."""
    with path.open() as file:
        data = json.load(file)

    records = []

    for entry in data["metric_per_case"]:
        case_id = remove_nifti_extension(entry["reference_file"])
        metrics = entry["metrics"][TUMOUR_LABEL]

        tp = float(metrics["TP"])
        fp = float(metrics["FP"])
        fn = float(metrics["FN"])

        counts = np.array([tp, fp, fn], dtype=float)
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError(
                f"Invalid TP, FP or FN counts for {case_id} in {path}"
            )

        precision_denominator = tp + fp
        recall_denominator = tp + fn

        dice_value = metrics["Dice"]

        records.append(
            {
                "case_id": case_id,
                "dataset": get_dataset(case_id),
                "phase": phase,
                "dice": (
                    float(dice_value)
                    if dice_value is not None else np.nan
                ),
                "precision": (
                    tp / precision_denominator
                    if precision_denominator > 0 else np.nan
                ),
                "recall": (
                    tp / recall_denominator
                    if recall_denominator > 0 else np.nan
                ),
            }
        )

    result = pd.DataFrame(
        records,
        columns=[
            "case_id",
            "dataset",
            "phase",
            "dice",
            "precision",
            "recall",
        ],
    )

    if result.empty:
        raise ValueError(f"No per-case results found in {path}")

    if result["case_id"].duplicated().any():
        duplicates = result.loc[
            result["case_id"].duplicated(keep=False), "case_id"
        ].tolist()
        raise ValueError(
            f"Duplicate summary cases found in {path}: {duplicates}"
        )

    validate_metric_values(
        result,
        ["dice", "precision", "recall"],
        path,
    )

    return result


def load_detection_file(path, phase):
    """Load lesion detectability at the configured IoU threshold."""
    data = pd.read_csv(path)

    required = {"case", "iou_threshold", "detectability"}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    thresholds = pd.to_numeric(
        data["iou_threshold"],
        errors="raise",
    )

    # Select results calculated using the 0.2 lesion-matching threshold.
    data = data[
        np.isclose(
            thresholds,
            IOU_THRESHOLD,
            rtol=0,
            atol=1e-8,
        )
    ].copy()

    if data.empty:
        raise ValueError(
            f"No IoU {IOU_THRESHOLD} rows found in {path}"
        )

    if data["case"].isna().any():
        raise ValueError(f"Missing case identifiers in {path}")

    data["case_id"] = data["case"].apply(remove_nifti_extension)
    data["detectability"] = pd.to_numeric(
        data["detectability"],
        errors="raise",
    )

    if data["case_id"].duplicated().any():
        duplicates = data.loc[
            data["case_id"].duplicated(keep=False), "case_id"
        ].tolist()
        raise ValueError(
            f"Duplicate detection cases found in {path}: {duplicates}"
        )

    result = data[["case_id", "detectability"]].copy()
    result["phase"] = phase

    validate_metric_values(result, ["detectability"], path)

    return result


def load_phase(phase):
    """Combine summary metrics and lesion detectability for one model."""
    summary = load_summary_file(
        FILES[phase]["summary"],
        phase,
    )
    detection = load_detection_file(
        FILES[phase]["detectability"],
        phase,
    )

    summary_cases = set(summary["case_id"])
    detection_cases = set(detection["case_id"])

    if summary_cases != detection_cases:
        raise ValueError(
            f"Summary and detection cases do not match for {phase}.\n"
            f"Only in summary: "
            f"{sorted(summary_cases - detection_cases)}\n"
            f"Only in detection: "
            f"{sorted(detection_cases - summary_cases)}"
        )

    return summary.merge(
        detection,
        on=["case_id", "phase"],
        how="inner",
        validate="one_to_one",
    )


def pair_phases(single, dual):
    """Pair single- and dual-phase results by case ID and dataset."""
    keys = ["case_id", "dataset"]

    single_keys = set(map(tuple, single[keys].to_numpy()))
    dual_keys = set(map(tuple, dual[keys].to_numpy()))

    if single_keys != dual_keys:
        raise ValueError(
            "Single- and dual-phase cases or datasets do not match.\n"
            f"Only in single phase: {sorted(single_keys - dual_keys)}\n"
            f"Only in dual phase: {sorted(dual_keys - single_keys)}"
        )

    return single[keys + METRICS].merge(
        dual[keys + METRICS],
        on=keys,
        how="inner",
        suffixes=("_single", "_dual"),
        validate="one_to_one",
    )


# -------------------------------------------------------------------
# Statistical analysis
# -------------------------------------------------------------------

def paired_mean_confidence_interval(differences, confidence=0.95):
    """Parametric confidence interval for the mean paired difference."""
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
    """Holm correction across the configured family of metric tests."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid_indices = np.flatnonzero(np.isfinite(p_values))

    if len(valid_indices) == 0:
        return adjusted

    order = valid_indices[np.argsort(p_values[valid_indices])]
    running_maximum = 0.0

    # Include all configured tests in the family size.
    family_size = len(p_values)

    for rank, index in enumerate(order):
        corrected = (family_size - rank) * p_values[index]
        running_maximum = max(running_maximum, corrected)
        adjusted[index] = min(running_maximum, 1.0)

    return adjusted


def safe_mean(values):
    return float(np.mean(values)) if len(values) else np.nan


def safe_sd(values):
    return float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan


def create_paired_ttest_results(paired):
    """Run two-sided paired t-tests for the four configured metrics."""
    rows = []

    for metric in METRICS:
        single_column = f"{metric}_single"
        dual_column = f"{metric}_dual"

        metric_data = (
            paired[
                ["case_id", "dataset", single_column, dual_column]
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[single_column, dual_column])
        )

        single_values = metric_data[single_column].to_numpy(dtype=float)
        dual_values = metric_data[dual_column].to_numpy(dtype=float)
        differences = dual_values - single_values
        n = len(differences)

        t_statistic = np.nan
        p_value = np.nan

        if n >= 2:
            if np.all(differences == 0):
                # Every pair is identical.
                t_statistic = 0.0
                p_value = 1.0
            elif np.all(differences == differences[0]):
                # A constant nonzero difference has zero standard error.
                t_statistic = np.copysign(np.inf, differences[0])
                p_value = 0.0
            else:
                test = ttest_rel(dual_values, single_values)
                t_statistic = float(test.statistic)
                p_value = float(test.pvalue)

        ci_lower, ci_upper = paired_mean_confidence_interval(differences)

        rows.append(
            {
                "metric": metric,
                "iou_threshold": (
                    IOU_THRESHOLD
                    if metric == "detectability" else np.nan
                ),
                "n_paired_cases": n,
                "n_excluded_pairs": len(paired) - n,
                "single_phase_mean": safe_mean(single_values),
                "single_phase_sd": safe_sd(single_values),
                "dual_phase_mean": safe_mean(dual_values),
                "dual_phase_sd": safe_sd(dual_values),
                "mean_difference_dual_minus_single": safe_mean(differences),
                "sd_paired_difference": safe_sd(differences),
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "t_statistic": t_statistic,
                "paired_t_test_p_value": p_value,
                "dual_better_n": int(np.sum(differences > 0)),
                "single_better_n": int(np.sum(differences < 0)),
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


# -------------------------------------------------------------------
# Descriptive summaries
# -------------------------------------------------------------------

def summarise_values(group, metric):
    """Summarise valid per-case scores, including median and quartiles."""
    values = group[metric].replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "metric": metric,
        "iou_threshold": (
            IOU_THRESHOLD if metric == "detectability" else np.nan
        ),
        "n_cases": len(group),
        "n_valid": len(values),
        "n_missing": len(group) - len(values),
        "mean": values.mean(),
        "sd": values.std(ddof=1),
        "median": values.median(),
        "q1": values.quantile(0.25),
        "q3": values.quantile(0.75),
    }


def create_overall_summary(all_data):
    """Calculate descriptive statistics for each model."""
    rows = []

    for phase, group in all_data.groupby("phase", sort=False):
        for metric in METRICS:
            rows.append(
                {
                    "phase": phase,
                    **summarise_values(group, metric),
                }
            )

    return pd.DataFrame(rows)


def create_dataset_summary(all_data):
    """Calculate descriptive statistics per model and source dataset."""
    rows = []

    for (phase, dataset), group in all_data.groupby(
        ["phase", "dataset"],
        sort=False,
    ):
        for metric in METRICS:
            rows.append(
                {
                    "phase": phase,
                    "dataset": dataset,
                    **summarise_values(group, metric),
                }
            )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    check_files_exist()

    single = load_phase("SinglePhase")
    dual = load_phase("DualPhase")

    paired = pair_phases(single, dual)
    all_data = pd.concat([single, dual], ignore_index=True)

    for metric in METRICS:
        paired[f"{metric}_difference"] = (
            paired[f"{metric}_dual"]
            - paired[f"{metric}_single"]
        )

    statistical_results = create_paired_ttest_results(paired)
    overall_summary = create_overall_summary(all_data)
    dataset_summary = create_dataset_summary(all_data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    statistical_results.to_csv(
        OUTPUT_DIR / "paired_ttest_results.csv",
        index=False,
    )
    paired.to_csv(
        OUTPUT_DIR / "paired_case_level_results.csv",
        index=False,
    )
    all_data.to_csv(
        OUTPUT_DIR / "case_level_metrics.csv",
        index=False,
    )
    overall_summary.to_csv(
        OUTPUT_DIR / "overall_summary.csv",
        index=False,
    )
    dataset_summary.to_csv(
        OUTPUT_DIR / "dataset_specific_summary.csv",
        index=False,
    )

    display_columns = [
        "metric",
        "n_paired_cases",
        "n_excluded_pairs",
        "single_phase_mean",
        "dual_phase_mean",
        "mean_difference_dual_minus_single",
        "ci_95_lower",
        "ci_95_upper",
        "t_statistic",
        "paired_t_test_p_value",
        "holm_adjusted_p_value",
        "significant_after_holm",
    ]

    print("\nPAIRED SINGLE- VS DUAL-PHASE T-TEST COMPARISON")
    print("Dice, precision and recall: voxel metrics from summary JSON")
    print(f"Detectability: lesion matching at IoU >= {IOU_THRESHOLD}")
    print("Difference: dual phase minus single phase")
    print("Holm correction across the four metric tests")
    print("Confidence intervals: unadjusted 95% paired-difference CIs")
    print(
        statistical_results[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nOVERALL DESCRIPTIVE RESULTS")
    print(overall_summary.round(4).to_string(index=False))

    print("\nDESCRIPTIVE RESULTS PER DATASET")
    print(dataset_summary.round(4).to_string(index=False))

    print(f"\nResults saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()