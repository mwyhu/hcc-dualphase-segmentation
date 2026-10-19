from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

BASE_DIR = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis"
)

OUTPUT_DIR = Path(
    "/Users/michellehu/Desktop/"
    "hcc-dualphase-segmentation/benchmark/emperical_ceiling/bootstrap"
)

IOU_THRESHOLD = 0.20
TUMOUR_LABEL = "1"

METRICS = ["dice", "precision", "recall", "detectability"]

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42

FILES = {
    "SP": {
        "summary": BASE_DIR / "dice_per_dataset/ceiling/SP_on_ceiling_testset.json",
        "detectability": BASE_DIR / "detectability/ceiling/Detect_SP_on_ceiling_testset.csv",
    },
    "SP ceiling": {
        "summary": BASE_DIR / "dice_per_dataset/ceiling/SPCeiling_on_ceiling_testset.json",
        "detectability": (
            BASE_DIR / "detectability/ceiling/Detect_SPceiling_on_ceiling_testset.csv"
        ),
    },
    "DP": {
        "summary": BASE_DIR / "dice_per_dataset/ceiling/DP_on_ceiling_testset.json",
        "detectability": BASE_DIR / "detectability/ceiling/Detect_DP_on_ceiling_testset.csv",
    },
    "DP ceiling": {
        "summary": BASE_DIR / "dice_per_dataset/ceiling/DPCeiling_on_ceiling_testset.json",
        "detectability": (
            BASE_DIR / "detectability/ceiling/Detect_DPceiling_on_ceiling_testset.csv"
        ),
    },
}

# Each comparison uses ceiling minus original.
COMPARISONS = [
    ("SP ceiling versus SP", "SP", "SP ceiling"),
    ("DP ceiling versus DP", "DP", "DP ceiling"),
]


# -------------------------------------------------------------------
# Loading and validation
# -------------------------------------------------------------------

def remove_nifti_extension(filename):
    """
    Return the filename without .nii or .nii.gz
    """
    filename = str(filename).replace("\\", "/").rsplit("/", 1)[-1]

    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]

    return filename


def get_dataset(case_id):
    """
    Determine the source dataset from the case id
    """
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
    """
    Check all configured input paths before running the analysis
    """
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
    """
    Allow missing values, reject infinite or out-of-range scores
    """
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
    """
    Load per-case Dice, voxel precision and voxel recall
    """
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
    """
    Load lesion detectability at the configured IoU threshold
    """
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
    """
    Combine summary metrics and lesion detectability for one model
    """
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
    """
    Pair single- and dual-phase results by case ID and dataset
    """
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
    """
    Parametric confidence interval for the mean paired difference
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


def paired_bootstrap_test(differences, n_resamples=N_BOOTSTRAP, confidence=0.95, seed=BOOTSTRAP_SEED,):
    """
    Bootstrap the mean paired difference (DP - SP).
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    n = len(differences)

    if n < 2:
        return np.nan, np.nan, np.nan

    observed_mean = float(differences.mean())

    if np.all(differences == differences[0]):
        if differences[0] == 0:
            return 0.0, 0.0, 1.0

        # A constant nonzero difference provides no observed variation.
        # Report the degenerate interval, but leave the p-value undefined.
        return observed_mean, observed_mean, np.nan

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_resamples)

    # Generate resamples in batches to limit memory use.
    batch_size = 1000

    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)

        indices = rng.integers(
            0,
            n,
            size=(stop - start, n),
        )

        bootstrap_means[start:stop] = (
            differences[indices].mean(axis=1)
        )

    # Unadjusted percentile confidence interval
    alpha = 1.0 - confidence

    ci_lower, ci_upper = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    # Equivalent to resampling differences centred at zero
    null_means = bootstrap_means - observed_mean

    extreme_count = np.count_nonzero(
        np.abs(null_means) >= abs(observed_mean)
    )

    # Monte Carlo correction avoids a reported p-value of zero
    p_value = (extreme_count + 1) / (n_resamples + 1)

    return float(ci_lower), float(ci_upper), float(p_value)


def holm_correction(p_values):
    """
    Holm correction across the configured family of metric tests
    """
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)

    valid_indices = np.flatnonzero(np.isfinite(p_values))

    if len(valid_indices) == 0:
        return adjusted

    order = valid_indices[np.argsort(p_values[valid_indices])]
    running_maximum = 0.0

    # Include all configured tests in the family size
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
    """
    Run paired t-tests and paired bootstrap tests for each metric
    """
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
                t_statistic = 0.0
                p_value = 1.0

            elif np.all(differences == differences[0]):
                t_statistic = np.copysign(np.inf, differences[0])
                p_value = 0.0

            else:
                test = ttest_rel(
                    dual_values,
                    single_values,
                    alternative="two-sided",
                )
                t_statistic = float(test.statistic)
                p_value = float(test.pvalue)

        ci_lower, ci_upper = paired_mean_confidence_interval(
            differences
        )

        bootstrap_lower, bootstrap_upper, bootstrap_p = (
            paired_bootstrap_test(differences)
        )

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
                "mean_difference_dual_minus_single": safe_mean(
                    differences
                ),
                "sd_paired_difference": safe_sd(differences),
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "t_statistic": t_statistic,
                "paired_t_test_p_value": p_value,
                "bootstrap_ci_95_lower": bootstrap_lower,
                "bootstrap_ci_95_upper": bootstrap_upper,
                "bootstrap_p_value": bootstrap_p,
                "bootstrap_n_resamples": N_BOOTSTRAP,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "dual_better_n": int(np.sum(differences > 0)),
                "single_better_n": int(np.sum(differences < 0)),
                "equal_n": int(np.sum(differences == 0)),
            }
        )

    results = pd.DataFrame(rows)

    # Separate correction across the four paired t-tests
    results["holm_adjusted_p_value"] = holm_correction(
        results["paired_t_test_p_value"].to_numpy()
    )

    results["significant_after_holm"] = (
        results["holm_adjusted_p_value"] < 0.05
    )

    # Separate correction across the four bootstrap tests
    results["bootstrap_holm_adjusted_p_value"] = holm_correction(
        results["bootstrap_p_value"].to_numpy()
    )

    results["bootstrap_significant_after_holm"] = (
        results["bootstrap_holm_adjusted_p_value"] < 0.05
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
    """
    Calculate descriptive statistics for each model
    """
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

    model_data = {
        model: load_model(model)
        for model in FILES
    }

    # Check that all four models use the same test cases.
    reference_model = next(iter(model_data))
    reference_cases = set(model_data[reference_model]["case_id"])

    for model, data in model_data.items():
        model_cases = set(data["case_id"])

        if model_cases != reference_cases:
            raise ValueError(
                f"{model}: cases differ from {reference_model}.\n"
                f"Only in {model}: "
                f"{sorted(model_cases - reference_cases)}\n"
                f"Only in {reference_model}: "
                f"{sorted(reference_cases - model_cases)}"
            )

    all_data = pd.concat(
        list(model_data.values()),
        ignore_index=True,
    )

    statistical_frames = []
    paired_frames = []

    for comparison, original_model, ceiling_model in COMPARISONS:
        paired = pair_models(
            model_data[original_model],
            model_data[ceiling_model],
        )

        paired["comparison"] = comparison
        paired["original_model"] = original_model
        paired["ceiling_model"] = ceiling_model

        for metric in METRICS:
            paired[f"{metric}_difference"] = (
                paired[f"{metric}_ceiling"]
                - paired[f"{metric}_original"]
            )

        statistical_frames.append(
            create_paired_ttest_results(
                paired,
                comparison,
                original_model,
                ceiling_model,
            )
        )

        paired_frames.append(paired)

    statistical_results = pd.concat(
        statistical_frames,
        ignore_index=True,
    )

    paired_results = pd.concat(
        paired_frames,
        ignore_index=True,
    )

    # Holm correction across eight t-tests:
    # two comparisons × four metrics.
    statistical_results["holm_adjusted_p_value"] = holm_correction(
        statistical_results["paired_t_test_p_value"].to_numpy()
    )

    statistical_results["significant_after_holm"] = (
        statistical_results["holm_adjusted_p_value"] < 0.05
    )

    # Separate Holm correction across eight bootstrap tests.
    statistical_results["bootstrap_holm_adjusted_p_value"] = (
        holm_correction(
            statistical_results["bootstrap_p_value"].to_numpy()
        )
    )

    statistical_results["bootstrap_significant_after_holm"] = (
        statistical_results["bootstrap_holm_adjusted_p_value"] < 0.05
    )

    overall_summary = create_overall_summary(all_data)
    dataset_summary = create_dataset_summary(all_data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    statistical_results.to_csv(
        OUTPUT_DIR / "paired_ttest_results.csv",
        index=False,
    )

    paired_results.to_csv(
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

    ttest_display_columns = [
        "comparison",
        "metric",
        "n_paired_cases",
        "n_excluded_pairs",
        "original_mean",
        "ceiling_mean",
        "mean_difference_ceiling_minus_original",
        "ci_95_lower",
        "ci_95_upper",
        "t_statistic",
        "paired_t_test_p_value",
        "holm_adjusted_p_value",
        "significant_after_holm",
    ]

    bootstrap_display_columns = [
        "comparison",
        "metric",
        "n_paired_cases",
        "n_excluded_pairs",
        "original_mean",
        "ceiling_mean",
        "mean_difference_ceiling_minus_original",
        "bootstrap_ci_95_lower",
        "bootstrap_ci_95_upper",
        "bootstrap_p_value",
        "bootstrap_holm_adjusted_p_value",
        "bootstrap_significant_after_holm",
    ]

    print("\nCEILING TEST SET: PAIRED T-TEST COMPARISONS")
    print("Comparisons: SP ceiling versus SP; DP ceiling versus DP")
    print("Dice, precision and recall: voxel metrics from summary JSON")
    print(f"Detectability: lesion matching at IoU >= {IOU_THRESHOLD}")
    print("Difference: ceiling minus original")
    print("Means use the same valid paired cases for each comparison")
    print("Holm correction across eight tests: two comparisons × four metrics")
    print("Confidence intervals: unadjusted 95% parametric CIs")

    print(
        statistical_results[ttest_display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nCEILING TEST SET: PAIRED BOOTSTRAP COMPARISONS")
    print(f"Bootstrap resamples: {N_BOOTSTRAP:,}")
    print(f"Random seed: {BOOTSTRAP_SEED}")
    print("Statistic: mean paired difference, ceiling minus original")
    print("Resampling unit: independent case, preserving model pairing")
    print("Test: approximate two-sided null-centred bootstrap")
    print("Holm correction across eight bootstrap tests")
    print("Confidence intervals: unadjusted 95% percentile bootstrap CIs")

    print(
        statistical_results[bootstrap_display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nOVERALL DESCRIPTIVE RESULTS")
    print("Each model uses its own valid cases for each metric.")
    print("Overall summaries weight cases equally.")
    print(overall_summary.round(4).to_string(index=False))

    print("\nDESCRIPTIVE RESULTS PER DATASET")
    print(dataset_summary.round(4).to_string(index=False))

    print(f"\nResults saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()