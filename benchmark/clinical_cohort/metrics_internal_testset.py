from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

BASE_DIR = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "benchmark/clinical_cohort"
)

DICE_DIR = BASE_DIR / "dice"
DETECT_DIR = BASE_DIR / "detect_files"
PLOT_DIR = BASE_DIR / "plots"

MODEL_ORDER = ["SP", "SP ceiling", "DP", "DP ceiling"]

MODEL_PALETTE = {
    "SP": "#033943",
    "SP ceiling": "#5E9E98",
    "DP": "#f0781a",
    "DP ceiling": "#E8A524",
}

METRICS = {
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Sensitivity",
    "detectability": "Detectability",
}

REPORT_THRESHOLD = 0.20
TUMOUR_LABEL = "1"

# Add one entry per dataset, with all four models.
# Filenames below retain your current filenames.
DATASETS = {
    "CAESAR": {
        "prefixes": ["CAESAR"],
        "dice_files": {
            "SP": DICE_DIR / "SP_CAIRO5_summary.json",
            "SP ceiling": DICE_DIR / "ceiling_SP_CAIRO5.json",
            "DP": DICE_DIR / "DP_CAIRO5_summary.json",
            "DP ceiling": DICE_DIR / "ceiling_DP_CAIRO5.json",
        },
        "detect_files": {
            "SP": DETECT_DIR / "Detect_SP_CAIRO5.csv",
            "SP ceiling": DETECT_DIR / "Detect_ceiling_SP_CAIRO5.csv",
            "DP": DETECT_DIR / "Detect_DP_CAIRO5.csv",
            "DP ceiling": DETECT_DIR / "Detect_ceiling_DP_CAIRO5.csv",
        },
    },

    # Copy and uncomment this entry for another dataset.
    #
    # "Dataset name": {
    #     "prefixes": ["PATIENT_PREFIX"],
    #     "dice_files": {
    #         "SP": DICE_DIR / "SP_DATASET_summary.json",
    #         "SP ceiling": DICE_DIR / "ceiling_SP_DATASET.json",
    #         "DP": DICE_DIR / "DP_DATASET_summary.json",
    #         "DP ceiling": DICE_DIR / "ceiling_DP_DATASET.json",
    #     },
    #     "detect_files": {
    #         "SP": DETECT_DIR / "Detect_SP_DATASET.csv",
    #         "SP ceiling": DETECT_DIR / "Detect_ceiling_SP_DATASET.csv",
    #         "DP": DETECT_DIR / "Detect_DP_DATASET.csv",
    #         "DP ceiling": DETECT_DIR / "Detect_ceiling_DP_DATASET.csv",
    #     },
    # },
}

# Dataset plots follow the order of entries in DATASETS.
sns.set_theme(style="whitegrid")


# -------------------------------------------------------------------
# Case IDs and dataset identification
# -------------------------------------------------------------------

def case_id(value):
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    return name.removesuffix(".nii.gz").removesuffix(".nii")


def validate_settings():
    if not DATASETS:
        raise ValueError("No datasets configured")

    prefix_owners = {}

    for dataset, config in DATASETS.items():
        prefixes = config.get("prefixes", [])

        if not prefixes:
            raise ValueError(
                f"{dataset}: configure at least one patient prefix"
            )

        for prefix in prefixes:
            if not isinstance(prefix, str) or not prefix.strip():
                raise ValueError(
                    f"{dataset}: patient prefixes must be non-empty strings"
                )

            normalised = prefix.upper()

            if normalised in prefix_owners:
                raise ValueError(
                    f"Duplicate patient prefix {prefix}: "
                    f"{prefix_owners[normalised]} and {dataset}"
                )

            prefix_owners[normalised] = dataset

        for file_type in ("dice_files", "detect_files"):
            configured = set(config.get(file_type, {}))
            expected = set(MODEL_ORDER)

            if configured != expected:
                raise ValueError(
                    f"{dataset}, {file_type}: model names do not match.\n"
                    f"Missing: {sorted(expected - configured)}\n"
                    f"Unexpected: {sorted(configured - expected)}"
                )


def get_dataset(case_name):
    name = case_id(case_name).upper()

    prefixes = [
        (prefix.upper(), dataset)
        for dataset, config in DATASETS.items()
        for prefix in config["prefixes"]
    ]

    # Longer prefixes take priority.
    for prefix, dataset in sorted(
        prefixes,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if name.startswith(prefix):
            return dataset

    raise ValueError(
        f"Unrecognised dataset for patient: {case_name}. "
        "Add the patient prefix to DATASETS."
    )


def get_dataset_order(df):
    available = set(df["source"])

    return [
        dataset
        for dataset in DATASETS
        if dataset in available
    ]


def validate_case_dataset(case_name, expected_dataset):
    identified_dataset = get_dataset(case_name)

    if identified_dataset != expected_dataset:
        raise ValueError(
            f"{case_name}: patient belongs to {identified_dataset}, "
            f"but the input file is configured for {expected_dataset}"
        )


# -------------------------------------------------------------------
# Score validation
# -------------------------------------------------------------------

def validate_scores(df, metrics):
    for metric in metrics:
        df[metric] = pd.to_numeric(
            df[metric],
            errors="raise",
        )

        values = df[metric].dropna()

        if not values.between(0, 1).all():
            raise ValueError(
                f"{metric}: scores must be between 0 and 1"
            )

    if df.duplicated(["source", "model", "case"]).any():
        raise ValueError(
            "Duplicate dataset/model/case scores found"
        )

    return df


# -------------------------------------------------------------------
# Load Dice, voxel precision and voxel recall
# -------------------------------------------------------------------

def load_voxel_results():
    rows = []

    for dataset, config in DATASETS.items():
        for model in MODEL_ORDER:
            path = config["dice_files"][model]

            with path.open(encoding="utf-8") as handle:
                results = json.load(handle)

            entries = results["metric_per_case"]

            if not entries:
                raise ValueError(
                    f"{dataset}, {model}: no voxel results in {path}"
                )

            for entry in entries:
                filename = (
                    entry.get("reference_file")
                    or entry.get("prediction_file")
                )

                if not filename:
                    raise ValueError(
                        f"Missing case filename in {path}"
                    )

                patient = case_id(filename)
                validate_case_dataset(patient, dataset)

                metrics = entry["metrics"][TUMOUR_LABEL]

                tp = float(metrics["TP"])
                fp = float(metrics["FP"])
                fn = float(metrics["FN"])

                counts = np.array([tp, fp, fn])

                if (
                    not np.isfinite(counts).all()
                    or (counts < 0).any()
                ):
                    raise ValueError(
                        f"Invalid TP/FP/FN for {filename}"
                    )

                rows.append({
                    "source": dataset,
                    "model": model,
                    "case": patient,
                    "dice": metrics["Dice"],
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

    return validate_scores(
        pd.DataFrame(rows),
        ["dice", "precision", "recall"],
    )


# -------------------------------------------------------------------
# Load lesion detectability
# -------------------------------------------------------------------

def load_detectability_results():
    frames = []

    for dataset, config in DATASETS.items():
        for model in MODEL_ORDER:
            path = config["detect_files"][model]
            df = pd.read_csv(path)

            required = {
                "case",
                "detectability",
                "iou_threshold",
            }

            missing = required - set(df.columns)

            if missing:
                raise ValueError(
                    f"{path.name}: missing columns {sorted(missing)}"
                )

            df["iou_threshold"] = pd.to_numeric(
                df["iou_threshold"],
                errors="raise",
            )

            df = df.loc[
                np.isclose(
                    df["iou_threshold"],
                    REPORT_THRESHOLD,
                    rtol=0,
                    atol=1e-8,
                )
            ].copy()

            if df.empty:
                raise ValueError(
                    f"{path.name}: no rows at IoU "
                    f"{REPORT_THRESHOLD:.2f}"
                )

            if df["case"].isna().any():
                raise ValueError(
                    f"{path.name}: missing case IDs"
                )

            df["case"] = df["case"].map(case_id)

            for patient in df["case"]:
                validate_case_dataset(patient, dataset)

            df["source"] = dataset
            df["model"] = model

            # Plotted precision uses voxel counts from the JSON.
            frames.append(
                df[["source", "model", "case", "detectability"]]
            )

    return validate_scores(
        pd.concat(frames, ignore_index=True),
        ["detectability"],
    )


# -------------------------------------------------------------------
# Combine sources
# -------------------------------------------------------------------

def load_combined_results():
    voxel_df = load_voxel_results()
    detect_df = load_detectability_results()

    for dataset in DATASETS:
        for model in MODEL_ORDER:
            voxel_cases = set(
                voxel_df.loc[
                    (voxel_df["source"] == dataset)
                    & (voxel_df["model"] == model),
                    "case",
                ]
            )

            detect_cases = set(
                detect_df.loc[
                    (detect_df["source"] == dataset)
                    & (detect_df["model"] == model),
                    "case",
                ]
            )

            if voxel_cases != detect_cases:
                raise ValueError(
                    f"{dataset}, {model}: JSON and detectability "
                    "case IDs do not match.\n"
                    f"Only JSON: "
                    f"{sorted(voxel_cases - detect_cases)[:10]}\n"
                    f"Only CSV: "
                    f"{sorted(detect_cases - voxel_cases)[:10]}"
                )

        # Compare models within each dataset.
        reference_model = MODEL_ORDER[0]

        reference_cases = set(
            voxel_df.loc[
                (voxel_df["source"] == dataset)
                & (voxel_df["model"] == reference_model),
                "case",
            ]
        )

        for model in MODEL_ORDER[1:]:
            model_cases = set(
                voxel_df.loc[
                    (voxel_df["source"] == dataset)
                    & (voxel_df["model"] == model),
                    "case",
                ]
            )

            if model_cases != reference_cases:
                raise ValueError(
                    f"{dataset}, {model}: test case IDs differ "
                    "between models.\n"
                    f"Only {reference_model}: "
                    f"{sorted(reference_cases - model_cases)[:10]}\n"
                    f"Only {model}: "
                    f"{sorted(model_cases - reference_cases)[:10]}"
                )

    return voxel_df.merge(
        detect_df,
        on=["source", "model", "case"],
        how="inner",
        validate="one_to_one",
    )


# -------------------------------------------------------------------
# Summarise metric values
# -------------------------------------------------------------------

def summarise_values(values):
    values = values.dropna()

    if values.empty:
        return {
            "N": 0,
            "Mean ± SD": "NA",
            "Median [Q1, Q3]": "NA",
            "IQR": "NA",
        }

    mean = values.mean()
    sd = values.std(ddof=1)
    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    mean_sd = (
        f"{mean:.4f} ± {sd:.4f}"
        if len(values) > 1
        else f"{mean:.4f} ± NA"
    )

    return {
        "N": len(values),
        "Mean ± SD": mean_sd,
        "Median [Q1, Q3]": (
            f"{median:.4f} [{q1:.4f}, {q3:.4f}]"
        ),
        "IQR": f"{q3 - q1:.4f}",
    }


# -------------------------------------------------------------------
# Print and save overall descriptive table
# -------------------------------------------------------------------

def print_metric_table(df):
    rows = []

    for metric, label in METRICS.items():
        for model in MODEL_ORDER:
            values = df.loc[
                df["model"] == model,
                metric,
            ]

            rows.append({
                "Metric": label,
                "Model": model,
                **summarise_values(values),
            })

    table = pd.DataFrame(rows)

    print(
        "\n--- Clinical cohorts combined: primary versus "
        "empirical ceiling models ---"
    )
    print(
        f"Detectability: IoU ≥ {REPORT_THRESHOLD:.2f}. "
        "Undefined scores are excluded per metric. "
        "Overall summaries weight patients equally."
    )
    print(table.to_string(index=False))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    path = PLOT_DIR / "ceiling_metric_table.csv"
    table.to_csv(path, index=False)

    print(f"Saved: {path}")


# -------------------------------------------------------------------
# Print and save descriptive tables per dataset
# -------------------------------------------------------------------

def print_dataset_metric_tables(df):
    rows = []

    for dataset in get_dataset_order(df):
        for metric, label in METRICS.items():
            for model in MODEL_ORDER:
                values = df.loc[
                    (df["source"] == dataset)
                    & (df["model"] == model),
                    metric,
                ]

                rows.append({
                    "Dataset": dataset,
                    "Metric": label,
                    "Model": model,
                    **summarise_values(values),
                })

    table = pd.DataFrame(rows)

    print(
        "\n--- Clinical cohorts: metrics per dataset and model ---"
    )
    print(
        f"Detectability: IoU ≥ {REPORT_THRESHOLD:.2f}. "
        "Undefined scores are excluded per metric."
    )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for metric, label in METRICS.items():
        metric_table = table.loc[
            table["Metric"] == label
        ].drop(columns="Metric")

        print(f"\n--- {label} ---")
        print(metric_table.to_string(index=False))

        metric_path = PLOT_DIR / f"SP_DP_{metric}_per_dataset.csv"
        metric_table.to_csv(metric_path, index=False)

        print(f"Saved: {metric_path}")

    path = PLOT_DIR / "SP_DP_metrics_per_dataset.csv"
    table.to_csv(path, index=False)

    print(f"Saved: {path}")


# -------------------------------------------------------------------
# Plot all four metrics across all patients
# -------------------------------------------------------------------

def plot_combined_metrics(df):
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5.5),
        sharey=True,
    )

    rng = np.random.default_rng(42)

    for ax, (metric, label) in zip(axes, METRICS.items()):
        subset = df.dropna(subset=[metric])

        if not subset.empty:
            sns.boxplot(
                data=subset,
                x="model",
                y=metric,
                order=MODEL_ORDER,
                hue="model",
                hue_order=MODEL_ORDER,
                palette=MODEL_PALETTE,
                dodge=False,
                width=0.55,
                showfliers=False,
                ax=ax,
            )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        for position, model in enumerate(MODEL_ORDER):
            values = subset.loc[
                subset["model"] == model,
                metric,
            ].to_numpy(dtype=float)

            ax.scatter(
                position
                + rng.uniform(-0.16, 0.16, size=len(values)),
                values,
                color="black",
                alpha=0.25,
                s=9,
                linewidths=0,
                zorder=3,
            )

        ax.set_title(label, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="x", visible=False)

        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(
            ["SP", "SP\nceiling", "DP", "DP\nceiling"]
        )

    axes[0].set_ylabel("Per-case score")

    fig.suptitle(
        "Clinical cohorts combined: "
        "Primary versus Empirical Ceiling models",
        fontsize=16,
    )

    fig.text(
        0.5,
        0.02,
        "Voxel level: Dice, Precision, Sensitivity.\n"
        f"Lesion level: Detectability (IoU ≥ {REPORT_THRESHOLD:.2f}).",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.94])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        path = PLOT_DIR / f"SP_DP_ceiling_all_metrics.{extension}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Saved: {path}")

    plt.show()
    plt.close(fig)


# -------------------------------------------------------------------
# Plot all four models per dataset
# -------------------------------------------------------------------

def plot_metrics_by_dataset(df):
    dataset_order = get_dataset_order(df)

    figure_width = max(18, 5 * len(dataset_order))

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(figure_width, 6),
        sharey=True,
    )

    rng = np.random.default_rng(42)

    for ax, (metric, label) in zip(axes, METRICS.items()):
        subset = df.dropna(subset=[metric])

        if not subset.empty:
            sns.boxplot(
                data=subset,
                x="source",
                y=metric,
                order=dataset_order,
                hue="model",
                hue_order=MODEL_ORDER,
                palette=MODEL_PALETTE,
                dodge=True,
                width=0.7,
                showfliers=False,
                ax=ax,
            )

        for dataset_position, dataset in enumerate(dataset_order):
            for model_position, model in enumerate(MODEL_ORDER):
                values = subset.loc[
                    (subset["source"] == dataset)
                    & (subset["model"] == model),
                    metric,
                ].to_numpy(dtype=float)

                offset = (
                    (model_position + 0.5) / len(MODEL_ORDER) - 0.5
                ) * 0.7

                ax.scatter(
                    dataset_position
                    + offset
                    + rng.uniform(
                        -0.06,
                        0.06,
                        size=len(values),
                    ),
                    values,
                    color="black",
                    alpha=0.25,
                    s=9,
                    linewidths=0,
                    zorder=3,
                )

        ax.set_title(label, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="x", visible=False)

        ax.set_xticks(range(len(dataset_order)))
        ax.set_xticklabels(
            dataset_order,
            rotation=30 if len(dataset_order) > 1 else 0,
            ha="right" if len(dataset_order) > 1 else "center",
        )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    axes[0].set_ylabel("Per-case score")

    fig.suptitle(
        "Clinical cohorts: Primary versus "
        "Empirical Ceiling models by dataset",
        fontsize=16,
    )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=MODEL_PALETTE[model],
            edgecolor="black",
        )
        for model in MODEL_ORDER
    ]

    fig.legend(
        handles,
        MODEL_ORDER,
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=len(MODEL_ORDER),
        frameon=False,
    )

    fig.text(
        0.5,
        0.015,
        "Voxel level: Dice, Precision, Sensitivity.\n"
        f"Lesion level: Detectability (IoU ≥ {REPORT_THRESHOLD:.2f}).",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.17, 1, 0.94])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        path = PLOT_DIR / f"SP_DP_all_metrics_by_dataset.{extension}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Saved: {path}")

    plt.show()
    plt.close(fig)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    validate_settings()

    df = load_combined_results()

    print_metric_table(df)
    plot_combined_metrics(df)

    print_dataset_metric_tables(df)
    plot_metrics_by_dataset(df)


if __name__ == "__main__":
    main()