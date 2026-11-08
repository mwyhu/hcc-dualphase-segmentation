from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

DEFAULT_BASE = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis/statistical/external_validation"
)

MODELS = ["SP", "DP", "TSLL", "TSLT", "AtlasNet", "COALA"]
DATASETS = ["HCC", "LiTS", "TCIA", "WORC", "CAIRO5"]
METRICS = ["dice", "precision", "recall", "detectability"]

COMPARISONS = [
    (MODELS[row], MODELS[column])
    for row in range(len(MODELS))
    for column in range(row)
]

SCOPES = [
    (dataset, [dataset])
    for dataset in DATASETS
]

IOU_THRESHOLD = 0.20
TUMOUR_LABEL = "1"
ALPHA = 0.05


# -------------------------------------------------------------------
# Load and validate
# -------------------------------------------------------------------

def case_id(value):
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return name


def validate_frame(frame, path, metrics):
    if frame.empty:
        raise ValueError(f"No cases found in {path}")

    if frame["case_id"].isna().any():
        raise ValueError(f"Missing case identifiers in {path}")

    if frame["case_id"].eq("").any():
        raise ValueError(f"Empty case identifiers in {path}")

    if frame["case_id"].duplicated().any():
        raise ValueError(f"Duplicate case identifiers in {path}")

    for metric in metrics:
        values = frame[metric].to_numpy(dtype=float)

        invalid = np.isinf(values) | (
            np.isfinite(values)
            & ((values < 0) | (values > 1))
        )

        if invalid.any():
            raise ValueError(
                f"Invalid {metric} scores in {path}"
            )


def load_summary(path):
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    rows = []

    for entry in data["metric_per_case"]:
        metrics = entry["metrics"][TUMOUR_LABEL]

        filename = (
            entry.get("reference_file")
            or entry.get("prediction_file")
        )

        if not filename:
            raise ValueError(
                f"Missing case identifier in {path}"
            )

        tp, fp, fn = [
            float(metrics[key])
            for key in ["TP", "FP", "FN"]
        ]

        counts = np.array([tp, fp, fn])

        if (
            not np.isfinite(counts).all()
            or (counts < 0).any()
        ):
            raise ValueError(
                f"Invalid voxel counts in {path}"
            )

        rows.append({
            "case_id": case_id(filename),
            "dice": (
                float(metrics["Dice"])
                if metrics["Dice"] is not None
                else np.nan
            ),
            "precision": (
                tp / (tp + fp)
                if tp + fp > 0
                else np.nan
            ),
            "recall": (
                tp / (tp + fn)
                if tp + fn > 0
                else np.nan
            ),
        })

    frame = pd.DataFrame(rows)
    validate_frame(frame, path, METRICS[:3])

    return frame


def load_detection(path):
    frame = pd.read_csv(path)

    required = {
        "case",
        "iou_threshold",
        "detectability",
    }

    if not required.issubset(frame.columns):
        raise ValueError(
            f"Missing columns in {path}: "
            f"{required - set(frame.columns)}"
        )

    thresholds = pd.to_numeric(
        frame["iou_threshold"],
        errors="raise",
    )

    frame = frame.loc[
        np.isclose(
            thresholds,
            IOU_THRESHOLD,
            rtol=0,
            atol=1e-8,
        )
    ].copy()

    if frame["case"].isna().any():
        raise ValueError(
            f"Missing case identifiers in {path}"
        )

    frame["case_id"] = frame["case"].map(case_id)

    frame["detectability"] = pd.to_numeric(
        frame["detectability"],
        errors="raise",
    )

    frame = frame[["case_id", "detectability"]]

    validate_frame(frame, path, ["detectability"])

    return frame


def load_all(base):
    paths = []

    for model in MODELS:
        detection_code = (
            "AN" if model == "AtlasNet" else model
        )

        for dataset in DATASETS:
            paths.append((
                model,
                dataset,
                base
                / "summary_files"
                / f"{model}_{dataset}_summary.json",
                base
                / "detectability_files"
                / f"Detect_{detection_code}_{dataset}.csv",
            ))

    missing = [
        str(path)
        for _, _, summary, detection in paths
        for path in [summary, detection]
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(missing)
        )

    frames = []

    for model, dataset, summary_path, detection_path in paths:
        summary = load_summary(summary_path)
        detection = load_detection(detection_path)

        if set(summary["case_id"]) != set(detection["case_id"]):
            raise ValueError(
                f"Summary/detection case mismatch: "
                f"{model}, {dataset}"
            )

        frame = summary.merge(
            detection,
            on="case_id",
            validate="one_to_one",
        )

        frame["model"] = model
        frame["dataset"] = dataset

        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)

    expected = None

    for model in MODELS:
        keys = set(map(
            tuple,
            data.loc[
                data["model"] == model,
                ["dataset", "case_id"],
            ].to_numpy(),
        ))

        if expected is not None and keys != expected:
            raise ValueError(
                f"Case coverage differs for {model}. "
                "Resolve before comparing."
            )

        expected = keys

    return data


# -------------------------------------------------------------------
# Statistical helpers
# -------------------------------------------------------------------

def holm(p_values):
    """Holm correction, retaining the configured family size."""
    p_values = np.asarray(p_values, dtype=float)

    adjusted = np.full(len(p_values), np.nan)

    valid = np.flatnonzero(np.isfinite(p_values))
    order = valid[np.argsort(p_values[valid])]

    running_maximum = 0.0

    for rank, index in enumerate(order):
        running_maximum = max(
            running_maximum,
            (len(p_values) - rank) * p_values[index],
        )

        adjusted[index] = min(running_maximum, 1.0)

    return adjusted


def mean(values):
    return float(np.mean(values)) if len(values) else np.nan


def sd(values):
    return (
        float(np.std(values, ddof=1))
        if len(values) >= 2
        else np.nan
    )


def median(values):
    return (
        float(np.median(values))
        if len(values)
        else np.nan
    )


def wilcoxon_test(differences):
    """Two-sided Wilcoxon test on paired differences."""
    test_differences = np.round(
        np.asarray(differences, dtype=float),
        decimals=12,
    )

    statistic = np.nan
    p_value = np.nan

    if len(test_differences) >= 2:
        if np.all(test_differences == 0):
            statistic = 0.0
            p_value = 1.0

        else:
            test = wilcoxon(
                test_differences,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            statistic = float(test.statistic)
            p_value = float(test.pvalue)

    return (
        statistic,
        p_value,
        int(np.count_nonzero(test_differences)),
    )


# -------------------------------------------------------------------
# Friedman and pairwise analysis
# -------------------------------------------------------------------

def analyse_scope(data, scope, datasets):
    subset = data.loc[
        data["dataset"].isin(datasets)
    ]

    omnibus_rows = []
    rank_rows = []
    pairwise_rows = []
    case_tables = []

    for metric in METRICS:
        wide = subset.pivot(
            index=["dataset", "case_id"],
            columns="model",
            values=metric,
        ).reindex(columns=MODELS)

        wide = wide.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Friedman requires valid scores across all six models.
        complete = wide.dropna(subset=MODELS)

        values = complete.to_numpy(dtype=float)
        n_complete = len(complete)

        statistic = np.nan
        p_value = np.nan

        if n_complete >= 2:
            if np.all(values == values[:, :1]):
                statistic = 0.0
                p_value = 1.0

            else:
                test = friedmanchisquare(*[
                    complete[model].to_numpy()
                    for model in MODELS
                ])

                statistic = float(test.statistic)
                p_value = float(test.pvalue)

        omnibus_rows.append({
            "scope": scope,
            "datasets_included": ";".join(datasets),
            "metric": metric,
            "n_complete_cases": n_complete,
            "n_excluded_cases": len(wide) - n_complete,
            "n_models": len(MODELS),
            "friedman_statistic": statistic,
            "degrees_of_freedom": len(MODELS) - 1,
            "friedman_p_value": p_value,
            "kendalls_w": (
                statistic
                / (n_complete * (len(MODELS) - 1))
                if n_complete and np.isfinite(statistic)
                else np.nan
            ),
        })

        mean_ranks = (
            complete.rank(
                axis=1,
                ascending=False,
                method="average",
            )
            .mean()
        )

        for model in MODELS:
            rank_rows.append({
                "scope": scope,
                "metric": metric,
                "model": model,
                "n_complete_cases": n_complete,
                "mean_rank": mean_ranks[model],
            })

        for model_a, model_b in COMPARISONS:
            is_sp_dp = {model_a, model_b} == {"SP", "DP"}

            if is_sp_dp:
                # SP–DP uses every case with valid SP and DP
                # scores, regardless of the other models.
                paired = wide[
                    [model_a, model_b]
                ].dropna(subset=[model_a, model_b])

                case_selection = "valid_SP_DP_pairs"

            else:
                # Other comparisons retain the complete cases
                # used for the six-model Friedman benchmark.
                paired = complete[[model_a, model_b]]

                case_selection = "complete_across_all_models"

            n_paired = len(paired)

            va = paired[model_a].to_numpy(dtype=float)
            vb = paired[model_b].to_numpy(dtype=float)

            differences = va - vb

            (
                pair_statistic,
                pair_p,
                n_nonzero,
            ) = wilcoxon_test(differences)

            pairwise_rows.append({
                "scope": scope,
                "datasets_included": ";".join(datasets),
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "case_selection": case_selection,
                "iou_threshold": (
                    IOU_THRESHOLD
                    if metric == "detectability"
                    else np.nan
                ),
                "n_paired_cases": n_paired,
                "n_excluded_cases": len(wide) - n_paired,
                "n_nonzero_differences": n_nonzero,
                "model_a_mean": mean(va),
                "model_b_mean": mean(vb),
                "model_a_sd": sd(va),
                "model_b_sd": sd(vb),
                "model_a_median": median(va),
                "model_b_median": median(vb),
                "mean_difference_a_minus_b": mean(differences),
                "median_paired_difference_a_minus_b": median(
                    differences
                ),
                "wilcoxon_statistic": pair_statistic,
                "wilcoxon_p_value": pair_p,
            })

            cases = (
                paired[[model_a, model_b]]
                .reset_index()
                .rename(columns={
                    model_a: "score_a",
                    model_b: "score_b",
                })
            )

            cases["scope"] = scope
            cases["metric"] = metric
            cases["model_a"] = model_a
            cases["model_b"] = model_b
            cases["case_selection"] = case_selection
            cases["difference_a_minus_b"] = differences

            case_tables.append(cases)

    # Friedman: Holm correction across four metrics per dataset.
    omnibus = pd.DataFrame(omnibus_rows)

    omnibus["holm_adjusted_p_value"] = holm(
        omnibus["friedman_p_value"]
    )

    omnibus["significant_after_holm"] = (
        omnibus["holm_adjusted_p_value"] < ALPHA
    )

    omnibus["significance"] = (
        omnibus["holm_adjusted_p_value"].map(
            lambda p: (
                "Not testable"
                if not np.isfinite(p)
                else (
                    "Significant"
                    if p < ALPHA
                    else "Not significant"
                )
            )
        )
    )

    pairwise = pd.DataFrame(pairwise_rows)

    # Benchmark: Holm across 15 model pairs per dataset × metric.
    pairwise["holm_adjusted_p_value"] = (
        pairwise.groupby("metric")["wilcoxon_p_value"]
        .transform(
            lambda values: holm(values.to_numpy())
        )
    )

    pairwise["holm_family"] = (
        "15 model pairs within dataset and metric"
    )

    lookup = omnibus.set_index("metric")[
        "holm_adjusted_p_value"
    ]

    pairwise["friedman_holm_adjusted_p_value"] = (
        pairwise["metric"].map(lookup)
    )

    pairwise["friedman_significant_after_holm"] = (
        pairwise["friedman_holm_adjusted_p_value"] < ALPHA
    )

    pairwise["pairwise_significant_after_holm"] = (
        pairwise["holm_adjusted_p_value"] < ALPHA
    )

    # Preserve the Friedman gate for benchmark reporting.
    pairwise["significant_after_holm"] = (
        pairwise["friedman_significant_after_holm"]
        & pairwise["pairwise_significant_after_holm"]
    )

    def significance_label(row):
        if (
            not np.isfinite(row["wilcoxon_p_value"])
            or not np.isfinite(
                row["friedman_holm_adjusted_p_value"]
            )
        ):
            return "Not testable"

        if not row["friedman_significant_after_holm"]:
            return "Not interpreted: Friedman not significant"

        return (
            "Significant"
            if row["significant_after_holm"]
            else "Not significant"
        )

    pairwise["significance"] = pairwise.apply(
        significance_label,
        axis=1,
    )

    return (
        omnibus,
        pairwise,
        pd.DataFrame(rank_rows),
        pd.concat(case_tables, ignore_index=True),
    )


# -------------------------------------------------------------------
# Dedicated SP–DP results for the SP–DP plot
# -------------------------------------------------------------------

def create_sp_dp_results(pairwise):
    """
    Extract DP minus SP using valid SP–DP pairs.

    Apply Holm correction across four metrics within this dataset.
    This dedicated comparison does not use the Friedman gate.
    """
    selected = pairwise.loc[
        (pairwise["model_a"] == "DP")
        & (pairwise["model_b"] == "SP")
    ].copy()

    if (
        len(selected) != len(METRICS)
        or selected["metric"].duplicated().any()
        or set(selected["metric"]) != set(METRICS)
    ):
        raise ValueError(
            "Expected one DP–SP result for each metric"
        )

    selected = (
        selected.set_index("metric")
        .loc[METRICS]
        .reset_index()
    )

    adjusted = holm(
        selected["wilcoxon_p_value"].to_numpy()
    )

    result = selected[[
        "scope",
        "datasets_included",
        "metric",
        "case_selection",
        "iou_threshold",
        "n_paired_cases",
        "n_excluded_cases",
        "n_nonzero_differences",
        "model_b_mean",
        "model_a_mean",
        "model_b_sd",
        "model_a_sd",
        "mean_difference_a_minus_b",
        "median_paired_difference_a_minus_b",
        "wilcoxon_statistic",
        "wilcoxon_p_value",
    ]].rename(columns={
        "model_b_mean": "single_phase_mean",
        "model_a_mean": "dual_phase_mean",
        "model_b_sd": "single_phase_sd",
        "model_a_sd": "dual_phase_sd",
        "mean_difference_a_minus_b": (
            "mean_difference_dual_minus_single"
        ),
        "median_paired_difference_a_minus_b": (
            "median_paired_difference_dual_minus_single"
        ),
    })

    result["holm_adjusted_p_value"] = adjusted

    result["significant_after_holm"] = (
        result["holm_adjusted_p_value"] < ALPHA
    )

    result["significance"] = (
        result["holm_adjusted_p_value"].map(
            lambda p: (
                "Not testable"
                if not np.isfinite(p)
                else (
                    "Significant"
                    if p < ALPHA
                    else "Not significant"
                )
            )
        )
    )

    result["holm_family"] = (
        "4 metrics within dataset for DP minus SP"
    )

    result["test_method"] = (
        "Two-sided Wilcoxon signed-rank test"
    )

    return result


# -------------------------------------------------------------------
# Descriptive summaries
# -------------------------------------------------------------------

def descriptive_summary(data):
    """
    Summarise each model's available scores independently.

    These sample sizes may differ from the paired analyses.
    """
    rows = []

    for scope, datasets in SCOPES:
        for model in MODELS:
            frame = data.loc[
                (data["model"] == model)
                & data["dataset"].isin(datasets)
            ]

            for metric in METRICS:
                values = frame[metric].dropna()

                rows.append({
                    "scope": scope,
                    "model": model,
                    "metric": metric,
                    "n_cases": len(frame),
                    "n_valid": len(values),
                    "n_missing": len(frame) - len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                })

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Six-model external benchmark and dedicated "
            "SP–DP comparisons."
        )
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE,
    )

    args = parser.parse_args()

    data = load_all(args.base_dir)

    output = args.base_dir / "friedman_posthoc_results"
    output.mkdir(parents=True, exist_ok=True)

    omnibus_tables = []
    pairwise_tables = []
    rank_tables = []
    case_tables = []
    sp_dp_tables = []

    for scope, datasets in SCOPES:
        omnibus, pairwise, ranks, cases = analyse_scope(
            data,
            scope,
            datasets,
        )

        sp_dp = create_sp_dp_results(pairwise)

        omnibus.to_csv(
            output / f"omnibus_friedman_{scope}.csv",
            index=False,
        )

        pairwise.to_csv(
            output / f"posthoc_wilcoxon_{scope}.csv",
            index=False,
        )

        sp_dp.to_csv(
            output / f"SP_DP_wilcoxon_{scope}.csv",
            index=False,
        )

        omnibus_tables.append(omnibus)
        pairwise_tables.append(pairwise)
        rank_tables.append(ranks)
        case_tables.append(cases)
        sp_dp_tables.append(sp_dp)

        print(f"\nFRIEDMAN: {scope}")
        print(
            "Case selection: valid scores across all six models."
        )
        print(
            "Holm correction across four Friedman metric tests."
        )

        print(omnibus[[
            "metric",
            "n_complete_cases",
            "n_excluded_cases",
            "friedman_statistic",
            "kendalls_w",
            "holm_adjusted_p_value",
            "significance",
        ]].to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        ))

        print(f"\nBENCHMARK PAIRWISE WILCOXON: {scope}")
        print("Differences are model A minus model B.")
        print(
            "Holm correction across 15 pairs per metric."
        )
        print(
            "SP–DP uses valid SP–DP pairs; other comparisons "
            "use the six-model complete cases."
        )

        print(pairwise[[
            "model_a",
            "model_b",
            "metric",
            "case_selection",
            "n_paired_cases",
            "mean_difference_a_minus_b",
            "holm_adjusted_p_value",
            "significance",
        ]].to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        ))

        print(f"\nDEDICATED DP MINUS SP: {scope}")
        print(
            "Valid SP–DP pairs; Holm correction across "
            "four metrics; no Friedman significance gate."
        )

        print(sp_dp[[
            "metric",
            "n_paired_cases",
            "n_excluded_cases",
            "single_phase_mean",
            "dual_phase_mean",
            "mean_difference_dual_minus_single",
            "wilcoxon_p_value",
            "holm_adjusted_p_value",
            "significance",
        ]].to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        ))

    for tables, filename in [
        (omnibus_tables, "all_omnibus_results.csv"),
        (
            pairwise_tables,
            "all_posthoc_wilcoxon_results.csv",
        ),
        (rank_tables, "model_mean_ranks.csv"),
        (case_tables, "paired_case_level_results.csv"),
        (sp_dp_tables, "all_SP_DP_wilcoxon_results.csv"),
    ]:
        pd.concat(
            tables,
            ignore_index=True,
        ).to_csv(
            output / filename,
            index=False,
        )

    data.to_csv(
        output / "case_level_metrics.csv",
        index=False,
    )

    descriptive_summary(data).to_csv(
        output / "descriptive_summary.csv",
        index=False,
    )

    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()