from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

BASE_DIR = Path(
    "/Users/michellehu/Desktop/"
    "hcc-dualphase-segmentation/analysis/statistical/internal_testset/"
)

OUTPUT_DIR = BASE_DIR / "normality_checks"
METRICS = ["dice", "precision", "recall", "detectability"]
ALPHA = 0.05
IOU_THRESHOLD = 0.20
TUMOUR_LABEL = "1"

FILES = {
    "SinglePhase": {
        "summary": BASE_DIR / "SP_on_SP_testset.json",
        "detectability": BASE_DIR / "Detect_SP_on_SP_testset.csv",
    },
    "DualPhase": {
        "summary": BASE_DIR / "DP_on_DP_testset.json",
        "detectability": BASE_DIR / "Detect_DP_on_DP_testset.csv",
    },
}


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
    Determine the source dataset from the case identifier
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
    Allow missing values, but reject infinite or out-of-range scores
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




def assess(values):
    """
    Shapiro-Wilk on finite paired differences
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    w = p = np.nan

    if n < 3:
        interpretation = "Insufficient pairs (minimum 3)"
    elif np.ptp(values) == 0:
        interpretation = "Constant differences; normality test not applicable"
    else:
        result = stats.shapiro(values)
        w, p = float(result.statistic), float(result.pvalue)
        interpretation = (
            "Evidence against normality" if p < ALPHA
            else "No evidence against normality (not proof of normality)"
        )
        if n > 5000:
            interpretation += "; Shapiro p-value may be inaccurate for n > 5000"

    return {
        "n_paired_cases": n,
        "mean_difference": values.mean() if n else np.nan,
        "sd_difference": values.std(ddof=1) if n >= 2 else np.nan,
        "median_difference": np.median(values) if n else np.nan,
        "skewness": float(stats.skew(values, bias=False))
            if n >= 3 and np.ptp(values) > 0 else np.nan,
        "n_zero_differences": int(np.sum(values == 0)),
        "n_unique_differences": len(np.unique(values)),
        "shapiro_w": w,
        "shapiro_p_value": p,
        "alpha": ALPHA,
        "interpretation": interpretation,
    }


def plot_diagnostics(values, metric, scope, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if len(values):
        axes[0].hist(values, bins="auto", color="#8172B2", edgecolor="white")
        axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    else:
        axes[0].text(0.5, 0.5, "No complete pairs", ha="center", transform=axes[0].transAxes)
    axes[0].set(title="Histogram of paired differences", xlabel="DP − SP", ylabel="Cases")

    if len(values) >= 3 and np.ptp(values) > 0:
        stats.probplot(values, dist="norm", plot=axes[1])
        axes[1].set(title="Normal Q–Q plot", xlabel="Theoretical normal quantiles", ylabel="Ordered differences")
    else:
        axes[1].text(0.5, 0.5, "Q–Q plot unavailable:\ntoo few or constant differences",
                     ha="center", transform=axes[1].transAxes)

    fig.suptitle(f"{scope}: {metric} (n = {len(values)})")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--base-dir", type=Path, default=BASE_DIR,
                        help="Directory containing the four original input files.")

    parser.add_argument("--output", type=Path, default=None)

    parser.add_argument("--by-dataset", action="store_true",
                        help="Also diagnose each dataset separately, if comparing within datasets.")

    args = parser.parse_args()

    global FILES

    FILES = {
        "SinglePhase": {
            "summary": args.base_dir / "SP_on_SP_testset.json",
            "detectability": args.base_dir / "Detect_SP_on_SP_testset.csv",
        },
        "DualPhase": {
            "summary": args.base_dir / "DP_on_DP_testset.json",
            "detectability": args.base_dir / "Detect_DP_on_DP_testset.csv",
        },
    }

    if args.output is None:
        args.output = args.base_dir / "normality_checks"
    check_files_exist()
    data = pair_phases(load_phase("SinglePhase"), load_phase("DualPhase"))

    if args.by_dataset and ("dataset" not in data or data["dataset"].isna().any()):
        raise ValueError("--by-dataset requires a dataset value for every case.")

    args.output.mkdir(parents=True, exist_ok=True)

    scopes = [("Overall", data)]

    if args.by_dataset:
        scopes.extend((str(name), group) for name, group in data.groupby("dataset", sort=True))
    rows = []

    for scope_index, (scope, group) in enumerate(scopes):
        for metric in METRICS:
            single = pd.to_numeric(group[f"{metric}_single"], errors="raise").to_numpy(dtype=float)
            dual = pd.to_numeric(group[f"{metric}_dual"], errors="raise").to_numpy(dtype=float)
            valid = np.isfinite(single) & np.isfinite(dual)
            for scores in (single, dual):
                if np.any(np.isfinite(scores) & ((scores < 0) | (scores > 1))):
                    raise ValueError(f"{metric}: expected scores in [0, 1].")
            differences = dual[valid] - single[valid]
            rows.append({"scope": scope, "metric": metric,
                         "n_excluded_pairs": int((~valid).sum()), **assess(differences)})
            plot_diagnostics(differences, metric, scope,
                             args.output / f"{scope_index:02d}_{metric}_normality.png")

    results = pd.DataFrame(rows)

    results.to_csv(args.output / "normality_results.csv", index=False)

    for metric in METRICS:
        data[f"{metric}_difference"] = data[f"{metric}_dual"] - data[f"{metric}_single"]

    data.to_csv(args.output / "paired_case_level_results.csv", index=False)

    columns = ["scope", "metric", "n_paired_cases", "n_excluded_pairs", "shapiro_w", "shapiro_p_value", "interpretation"]

    print("\nNORMALITY OF PAIRED DIFFERENCES: DP - SP")
    print(results[columns].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nInterpretation:")
    print("p < 0.05: evidence against a normal distribution.")
    print("p >= 0.05: insufficient evidence against normality; does not prove normality.")
    print("Check Q-Q plots for tail departures and outliers, and histograms for skewness and ties.")
    print(f"\nSaved CSV and plots to: {args.output}")


if __name__ == "__main__":
    main()
