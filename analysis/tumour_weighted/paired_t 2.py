from pathlib import Path
import csv
import json
import math

import numpy as np
from scipy.stats import t


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

BASE_DIR = Path(
    "/Users/michellehu/Desktop/"
    "hcc-dualphase-segmentation/analysis/tumour_weighted"
)

OUTPUT_DIR = BASE_DIR / "crlm_sp_vs_dp_results"

DATASETS = ["WORC", "TCIA", "CAIRO5"]
MODELS = ["SP", "DP"]
METRICS = ["dice", "precision", "recall", "detectability"]

IOU_THRESHOLD = 0.20
TUMOUR_LABEL = "1"
ALPHA = 0.05

TEST_METHOD = (
    "Equal-dataset-weighted paired comparison "
    "with Satterthwaite degrees of freedom"
)


# -------------------------------------------------------------------
# Load scores
# -------------------------------------------------------------------

def remove_nifti_extension(value):
    """Return the case filename without .nii or .nii.gz."""
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return name


def check_files_exist():
    missing = []

    for model in MODELS:
        for dataset in DATASETS:
            paths = [
                BASE_DIR
                / "summary_files"
                / f"{model}_{dataset}_summary.json",
                BASE_DIR
                / "detectability_files"
                / f"Detect_{model}_{dataset}.csv",
            ]

            for path in paths:
                if not path.is_file():
                    missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(missing)
        )


def load_scores():
    """
    Load voxel metrics from JSON and lesion detectability from CSV.

    CSV precision is lesion-level precision and is not used here.
    Scores are indexed by dataset, model, case ID and metric.
    """
    scores = {}

    def add_score(dataset, model, case, metric, value):
        case_id = remove_nifti_extension(case)

        if not case_id:
            raise ValueError("Empty case identifier")

        key = (dataset, model, case_id, metric)

        if key in scores:
            raise ValueError(f"Duplicate score: {key}")

        if value is not None:
            value = float(value)

            if math.isnan(value):
                value = None
            elif not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"Invalid score for {key}: {value}"
                )

        scores[key] = value

    for model in MODELS:
        for dataset in DATASETS:
            summary_path = (
                BASE_DIR
                / "summary_files"
                / f"{model}_{dataset}_summary.json"
            )

            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)

            entries = summary.get("metric_per_case")

            if not isinstance(entries, list) or not entries:
                raise ValueError(
                    f"No per-case metrics found in {summary_path}"
                )

            for entry in entries:
                case = (
                    entry.get("reference_file")
                    or entry.get("prediction_file")
                )

                if not case:
                    raise ValueError(
                        f"Missing case identifier in {summary_path}"
                    )

                label_metrics = entry.get("metrics", {})

                if TUMOUR_LABEL not in label_metrics:
                    raise ValueError(
                        f"Label {TUMOUR_LABEL} not found for "
                        f"{case} in {summary_path}"
                    )

                values = label_metrics[TUMOUR_LABEL]

                tp = float(values["TP"])
                fp = float(values["FP"])
                fn = float(values["FN"])

                counts = np.array([tp, fp, fn], dtype=float)

                if (
                    not np.isfinite(counts).all()
                    or (counts < 0).any()
                ):
                    raise ValueError(
                        f"Invalid TP/FP/FN for {case} "
                        f"in {summary_path}"
                    )

                dice = values["Dice"]
                precision = (
                    tp / (tp + fp) if tp + fp > 0 else None
                )
                recall = (
                    tp / (tp + fn) if tp + fn > 0 else None
                )

                add_score(dataset, model, case, "dice", dice)
                add_score(
                    dataset, model, case, "precision", precision
                )
                add_score(dataset, model, case, "recall", recall)

            detection_path = (
                BASE_DIR
                / "detectability_files"
                / f"Detect_{model}_{dataset}.csv"
            )

            selected_rows = 0

            with detection_path.open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                reader = csv.DictReader(handle)

                required = {
                    "case",
                    "iou_threshold",
                    "detectability",
                }

                if not required <= set(reader.fieldnames or []):
                    raise ValueError(
                        f"{detection_path} requires columns: "
                        f"{sorted(required)}"
                    )

                for row in reader:
                    threshold = float(row["iou_threshold"])

                    if not math.isclose(
                        threshold,
                        IOU_THRESHOLD,
                        rel_tol=0,
                        abs_tol=1e-8,
                    ):
                        continue

                    raw = (row["detectability"] or "").strip()

                    value = (
                        None
                        if raw.lower() in (
                            "", "nan", "na", "none", "null"
                        )
                        else float(raw)
                    )

                    add_score(
                        dataset,
                        model,
                        row["case"],
                        "detectability",
                        value,
                    )

                    selected_rows += 1

            if selected_rows == 0:
                raise ValueError(
                    f"No IoU {IOU_THRESHOLD} rows "
                    f"in {detection_path}"
                )

    return scores


# -------------------------------------------------------------------
# Statistical calculations
# -------------------------------------------------------------------

def holm_correction(p_values):
    """Holm adjustment across the four configured metric tests."""
    p_values = np.asarray(p_values, dtype=float)

    adjusted = np.full(len(p_values), np.nan)
    valid_indices = np.flatnonzero(np.isfinite(p_values))

    order = valid_indices[
        np.argsort(p_values[valid_indices])
    ]

    running_maximum = 0.0
    family_size = len(p_values)

    for rank, index in enumerate(order):
        corrected = (family_size - rank) * p_values[index]
        running_maximum = max(running_maximum, corrected)
        adjusted[index] = min(running_maximum, 1.0)

    return adjusted


def descriptive_statistics(values):
    """Describe the supplied case scores."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        raise ValueError("Cannot summarise an empty array")

    q1, q3 = np.quantile(values, [0.25, 0.75])

    return {
        "n_valid": len(values),
        "mean": float(values.mean()),
        "sd": (
            float(values.std(ddof=1))
            if len(values) > 1
            else np.nan
        ),
        "median": float(np.median(values)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def equal_dataset_paired_test(differences_by_dataset):
    """
    Test the equally weighted average of dataset mean differences.

    Pairing:
        Each difference is DP - SP for the same case.

    Weighting:
        Each dataset contributes 1 / K.

    Variance:
        sum((1 / K)**2 * sample_variance_d / n_d)

    Inference:
        Approximate two-sided t-test with Satterthwaite
        degrees of freedom and an unadjusted 95% CI.

    Assumes independent cases within and between datasets.
    The trained SP and DP models are treated as fixed.
    """
    k = len(DATASETS)
    weight = 1.0 / k

    dataset_means = []
    variance_components = []
    sample_sizes = []

    for dataset in DATASETS:
        differences = np.asarray(
            differences_by_dataset[dataset],
            dtype=float,
        )

        n = len(differences)

        if n < 2:
            raise ValueError(
                f"{dataset}: at least two valid paired cases "
                "are required"
            )

        if not np.isfinite(differences).all():
            raise ValueError(
                f"{dataset}: nonfinite paired differences"
            )

        dataset_means.append(float(differences.mean()))
        sample_sizes.append(n)

        variance = float(differences.var(ddof=1))

        variance_components.append(
            weight**2 * variance / n
        )

    mean_difference = float(np.mean(dataset_means))

    variance_components = np.asarray(
        variance_components,
        dtype=float,
    )
    sample_sizes = np.asarray(
        sample_sizes,
        dtype=float,
    )

    total_variance = float(variance_components.sum())
    standard_error = float(np.sqrt(total_variance))

    if total_variance == 0:
        all_identical = all(
            np.all(
                np.asarray(differences_by_dataset[dataset]) == 0
            )
            for dataset in DATASETS
        )

        return {
            "mean_difference": mean_difference,
            "standard_error": standard_error,
            "degrees_of_freedom": np.nan,
            "ci_lower": 0.0 if all_identical else np.nan,
            "ci_upper": 0.0 if all_identical else np.nan,
            "t_statistic": 0.0 if all_identical else np.nan,
            "p_value": 1.0 if all_identical else np.nan,
            "test_status": (
                "identical_scores"
                if all_identical
                else "zero_within_dataset_variance"
            ),
        }

    df_denominator = float(np.sum(
        variance_components**2 / (sample_sizes - 1)
    ))

    degrees_of_freedom = (
        total_variance**2 / df_denominator
    )

    t_statistic = mean_difference / standard_error

    p_value = float(
        2 * t.sf(
            abs(t_statistic),
            df=degrees_of_freedom,
        )
    )

    margin = float(
        t.ppf(0.975, df=degrees_of_freedom)
        * standard_error
    )

    return {
        "mean_difference": mean_difference,
        "standard_error": standard_error,
        "degrees_of_freedom": float(degrees_of_freedom),
        "ci_lower": mean_difference - margin,
        "ci_upper": mean_difference + margin,
        "t_statistic": float(t_statistic),
        "p_value": p_value,
        "test_status": "ok",
    }


# -------------------------------------------------------------------
# Create results
# -------------------------------------------------------------------

def create_results(scores):
    statistical_rows = []
    paired_rows = []
    weight_rows = []
    descriptive_rows = []
    dataset_comparison_rows = []

    k = len(DATASETS)

    for metric in METRICS:
        case_sets = {
            model: {
                (dataset, case_id)
                for (
                    dataset,
                    stored_model,
                    case_id,
                    stored_metric,
                ) in scores
                if stored_model == model
                and stored_metric == metric
            }
            for model in MODELS
        }

        # Including dataset in the key prevents incorrect pairing
        # of identical filenames from different datasets.
        if case_sets["SP"] != case_sets["DP"]:
            raise ValueError(
                f"{metric}: SP and DP case lists do not match.\n"
                f"Only SP: "
                f"{sorted(case_sets['SP'] - case_sets['DP'])}\n"
                f"Only DP: "
                f"{sorted(case_sets['DP'] - case_sets['SP'])}"
            )

        all_keys = sorted(case_sets["SP"])

        # Exclude a pair if either model's metric is undefined.
        included_keys = [
            (dataset, case_id)
            for dataset, case_id in all_keys
            if scores[
                (dataset, "SP", case_id, metric)
            ] is not None
            and scores[
                (dataset, "DP", case_id, metric)
            ] is not None
        ]

        n = len(included_keys)

        single_values = np.array([
            scores[(dataset, "SP", case_id, metric)]
            for dataset, case_id in included_keys
        ], dtype=float)

        dual_values = np.array([
            scores[(dataset, "DP", case_id, metric)]
            for dataset, case_id in included_keys
        ], dtype=float)

        differences = dual_values - single_values

        single_by_dataset = {}
        dual_by_dataset = {}
        differences_by_dataset = {}

        for dataset in DATASETS:
            indices = np.array([
                index
                for index, (stored_dataset, _) in enumerate(
                    included_keys
                )
                if stored_dataset == dataset
            ], dtype=int)

            dataset_n = len(indices)

            if dataset_n < 2:
                raise ValueError(
                    f"{metric}/{dataset}: fewer than two "
                    "valid paired cases"
                )

            single = single_values[indices]
            dual = dual_values[indices]
            dataset_differences = differences[indices]

            single_by_dataset[dataset] = single
            dual_by_dataset[dataset] = dual
            differences_by_dataset[dataset] = (
                dataset_differences
            )

            dataset_total = sum(
                stored_dataset == dataset
                for stored_dataset, _ in all_keys
            )

            weight_rows.append({
                "dataset": dataset,
                "metric": metric,
                "n_paired_cases": dataset_n,
                "n_excluded_pairs": dataset_total - dataset_n,
                "weight": 1 / k,
                "case_weight": 1 / (k * dataset_n),
            })

            dataset_comparison_rows.append({
                "dataset": dataset,
                "metric": metric,
                "iou_threshold": (
                    IOU_THRESHOLD
                    if metric == "detectability"
                    else ""
                ),
                "n_paired_cases": dataset_n,
                "n_excluded_pairs": dataset_total - dataset_n,
                "single_phase_mean": float(single.mean()),
                "single_phase_sd": float(single.std(ddof=1)),
                "dual_phase_mean": float(dual.mean()),
                "dual_phase_sd": float(dual.std(ddof=1)),
                "mean_difference_dual_minus_single": float(
                    dataset_differences.mean()
                ),
                "sd_paired_difference": float(
                    dataset_differences.std(ddof=1)
                ),
                "dataset_weight": 1 / k,
            })

            for model, values in (
                ("SP", single),
                ("DP", dual),
            ):
                descriptive_rows.append({
                    "tumour_type": "CRLM",
                    "scope": "within_dataset",
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    **descriptive_statistics(values),
                })

        # Mean of the three dataset means.
        single_macro_mean = float(np.mean([
            single_by_dataset[dataset].mean()
            for dataset in DATASETS
        ]))

        dual_macro_mean = float(np.mean([
            dual_by_dataset[dataset].mean()
            for dataset in DATASETS
        ]))

        test = equal_dataset_paired_test(
            differences_by_dataset
        )

        statistical_rows.append({
            "tumour_type": "CRLM",
            "metric": metric,
            "iou_threshold": (
                IOU_THRESHOLD
                if metric == "detectability"
                else ""
            ),
            "weighting": "equal_dataset",
            "test_method": TEST_METHOD,
            "n_datasets": k,
            "n_paired_cases": n,
            "n_excluded_pairs": len(all_keys) - n,

            # Equal-dataset-weighted means.
            "single_phase_mean": single_macro_mean,
            "dual_phase_mean": dual_macro_mean,
            "mean_difference_dual_minus_single": (
                test["mean_difference"]
            ),

            # These SDs describe pooled cases, not uncertainty
            # around the equal-dataset-weighted means.
            "pooled_case_single_phase_sd": float(
                single_values.std(ddof=1)
            ),
            "pooled_case_dual_phase_sd": float(
                dual_values.std(ddof=1)
            ),
            "pooled_case_sd_paired_difference": float(
                differences.std(ddof=1)
            ),

            "standard_error_mean_difference": (
                test["standard_error"]
            ),
            "degrees_of_freedom": (
                test["degrees_of_freedom"]
            ),
            "ci_95_lower": test["ci_lower"],
            "ci_95_upper": test["ci_upper"],
            "t_statistic": test["t_statistic"],

            # Name retained for compatibility.
            # The test_method column specifies the actual method.
            "paired_t_test_p_value": test["p_value"],
            "test_status": test["test_status"],

            # Raw case counts, without weighting.
            "dual_better_n": int(np.sum(differences > 0)),
            "single_better_n": int(np.sum(differences < 0)),
            "equal_n": int(np.sum(differences == 0)),
        })

        for (
            (dataset, case_id),
            single,
            dual,
            difference,
        ) in zip(
            included_keys,
            single_values,
            dual_values,
            differences,
        ):
            dataset_n = len(
                differences_by_dataset[dataset]
            )

            paired_rows.append({
                "dataset": dataset,
                "case_id": case_id,
                "metric": metric,
                "single_phase": float(single),
                "dual_phase": float(dual),
                "difference_dual_minus_single": float(
                    difference
                ),
                "case_weight": 1 / (k * dataset_n),
            })

        # Separate pooled case summaries.
        # Their mean is case weighted, unlike the macro-means
        # in paired_ttest_results.csv.
        for model, values in (
            ("SP", single_values),
            ("DP", dual_values),
        ):
            descriptive_rows.append({
                "tumour_type": "CRLM",
                "scope": "pooled_cases",
                "dataset": "ALL",
                "model": model,
                "metric": metric,
                **descriptive_statistics(values),
            })

    adjusted_p_values = holm_correction([
        row["paired_t_test_p_value"]
        for row in statistical_rows
    ])

    for row, adjusted_p in zip(
        statistical_rows,
        adjusted_p_values,
    ):
        row["holm_adjusted_p_value"] = float(adjusted_p)
        row["significant_after_holm"] = (
            bool(adjusted_p < ALPHA)
            if np.isfinite(adjusted_p)
            else ""
        )

    return (
        statistical_rows,
        paired_rows,
        weight_rows,
        descriptive_rows,
        dataset_comparison_rows,
    )


# -------------------------------------------------------------------
# Save and print
# -------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    check_files_exist()
    scores = load_scores()

    (
        statistical_results,
        paired_results,
        dataset_weights,
        descriptive_results,
        dataset_comparisons,
    ) = create_results(scores)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "paired_ttest_results.csv": statistical_results,
        "paired_case_level_results.csv": paired_results,
        "dataset_weights.csv": dataset_weights,
        "paired_descriptive_summary.csv": descriptive_results,
        "per_dataset_paired_comparison.csv": dataset_comparisons,
    }

    for filename, rows in outputs.items():
        write_csv(OUTPUT_DIR / filename, rows)

    print("\nCOMBINED CRLM: SP VERSUS DP")
    print("Datasets: WORC, TCIA and CAIRO5")
    print("Weighting: each dataset contributes one third")
    print("Difference: DP minus SP")
    print(f"Detectability: CSV rows at IoU {IOU_THRESHOLD}")
    print(f"Test: {TEST_METHOD}")
    print("Holm correction across four metric tests")
    print(
        "Confidence intervals: unadjusted 95% CIs "
        "for the equal-dataset-weighted mean difference"
    )

    for row in statistical_results:
        print(
            f"\n{row['metric'].upper()} "
            f"(n={row['n_paired_cases']}, "
            f"excluded={row['n_excluded_pairs']})"
        )
        print(
            f"SP macro-mean: "
            f"{row['single_phase_mean']:.4f}"
        )
        print(
            f"DP macro-mean: "
            f"{row['dual_phase_mean']:.4f}"
        )
        print(
            f"Mean difference: "
            f"{row['mean_difference_dual_minus_single']:+.4f}"
        )
        print(
            f"95% CI: [{row['ci_95_lower']:.4f}, "
            f"{row['ci_95_upper']:.4f}]"
        )
        print(
            f"Standard error: "
            f"{row['standard_error_mean_difference']:.6f}"
        )
        print(
            f"Degrees of freedom: "
            f"{row['degrees_of_freedom']:.2f}"
        )
        print(
            f"Raw p: "
            f"{row['paired_t_test_p_value']:.6g}"
        )
        print(
            f"Holm-adjusted p: "
            f"{row['holm_adjusted_p_value']:.6g}"
        )
        print(
            f"Significant after Holm: "
            f"{row['significant_after_holm']}"
        )
        print(f"Test status: {row['test_status']}")

    print(f"\nResults saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()