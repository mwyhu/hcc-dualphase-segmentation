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
    "configuration_experiment/postprocessing"
)

DICE_DIR = BASE_DIR / "dice"
DETECT_DIR = BASE_DIR / "detect_files"
PLOT_DIR = BASE_DIR / "plots"

DICE_FILES = {
    "Raw Prediction": DICE_DIR / "2000_ep_summary.json",
    "Post-processed Prediction": DICE_DIR / "2000_ep_summary_pp.json",
}

DETECT_FILES = {
    "Raw Prediction": DETECT_DIR / "DP_2000_epochs.csv",
    "Post-processed Prediction": DETECT_DIR / "DP_2000_epochs_pp.csv",
}

MODEL_ORDER = [
    "Raw Prediction",
    "Post-processed Prediction",
]

TICK_ORDER = [
    "Raw\nPrediction",
    "Post-processed\nPrediction",
]

MODEL_PALETTE = {
    "Raw Prediction": "#f0781a",
    "Post-processed Prediction": "#AE7B95",
}

DATASET_ORDER = [
    "WAW_TACE",
    "MCT_LTDiag",
    "Liver_Lesions",
    "MSD08"
]

DATASET_PREFIXES = {
    "WAW_TACE": "WAW_TACE",
    "MCT_LTDiag": "MCT_LTDiag",
    "Liver_Lesions": "Liver_Lesions",
    "MSD08":"MSD08"
}

METRICS = {
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Sensitivity",
    "detectability": "Detectability",
}

REPORT_THRESHOLD = 0.20
TUMOUR_LABEL = "1"

sns.set_theme(style="whitegrid")


def case_id(value):
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    return name.removesuffix(".nii.gz").removesuffix(".nii")


def get_dataset(case_name):
    name = case_id(case_name).strip()

    for prefix, dataset in DATASET_PREFIXES.items():
        if name == prefix or name.startswith(prefix + "_"):
            return dataset

    raise ValueError(
        f"Unrecognised dataset prefix: {name!r}. "
        f"Expected prefixes: {list(DATASET_PREFIXES)}"
    )

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

    if df.duplicated(["model", "case"]).any():
        raise ValueError("Duplicate model/case scores found")

    return df


# -------------------------------------------------------------------
# Load Dice, voxel precision and voxel sensitivity
# -------------------------------------------------------------------

def load_voxel_results():
    rows = []

    for model, path in DICE_FILES.items():
        with path.open(encoding="utf-8") as handle:
            results = json.load(handle)

        for entry in results["metric_per_case"]:
            filename = (
                entry.get("reference_file")
                or entry.get("prediction_file")
            )

            if not filename:
                raise ValueError(
                    f"Missing case filename in {path}"
                )

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
                "model": model,
                "case": case_id(filename),
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

    for model, path in DETECT_FILES.items():
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
        df["model"] = model

        # CSV precision is lesion-level precision.
        # The plotted precision uses voxel precision from the JSON.
        frames.append(
            df[["model", "case", "detectability"]]
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

    for model in MODEL_ORDER:
        voxel_cases = set(
            voxel_df.loc[voxel_df["model"] == model, "case"]
        )
        detect_cases = set(
            detect_df.loc[detect_df["model"] == model, "case"]
        )

        if voxel_cases != detect_cases:
            raise ValueError(
                f"{model}: JSON and detectability case IDs "
                "do not match.\n"
                f"Only JSON: {sorted(voxel_cases - detect_cases)[:10]}\n"
                f"Only CSV: {sorted(detect_cases - voxel_cases)[:10]}"
            )

    # Raw and post-processed predictions must use the same cases.
    reference_cases = set(
        voxel_df.loc[
            voxel_df["model"] == MODEL_ORDER[0],
            "case",
        ]
    )

    for model in MODEL_ORDER[1:]:
        model_cases = set(
            voxel_df.loc[voxel_df["model"] == model, "case"]
        )

        if model_cases != reference_cases:
            raise ValueError(
                f"{model}: validation case IDs differ between models"
            )

    combined = voxel_df.merge(
        detect_df,
        on=["model", "case"],
        how="inner",
        validate="one_to_one",
    )

    combined["dataset"] = combined["case"].map(get_dataset)

    return combined


# -------------------------------------------------------------------
# Plot all four metrics, overall or per dataset
# -------------------------------------------------------------------

def plot_combined_metrics(df, dataset=None):
    scope = dataset if dataset is not None else "All datasets"
    suffix = f"_{dataset}" if dataset is not None else ""

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
                position + rng.uniform(
                    -0.16,
                    0.16,
                    size=len(values),
                ),
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
        ax.set_xticklabels(TICK_ORDER)

    axes[0].set_ylabel("Per-case score")

    fig.suptitle(
        "Configuration experiment: dual-phase validation, fold 0\n"
        f"Raw versus post-processed prediction | {scope}",
        fontsize=16,
    )

    fig.text(
        0.5,
        0.02,
        "Voxel level: Dice, Precision, Sensitivity. "
        f"Lesion level: Detectability (IoU ≥ {REPORT_THRESHOLD:.2f}).",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.94])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f"pp_config_all_metrics{suffix}.png"

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
# Print and save descriptive table
# -------------------------------------------------------------------

def print_metric_table(df, dataset=None):
    scope = dataset if dataset is not None else "All datasets"
    suffix = f"_{dataset}" if dataset is not None else ""
    rows = []

    for metric, label in METRICS.items():
        for model in MODEL_ORDER:
            values = df.loc[
                df["model"] == model,
                metric,
            ].dropna()

            if values.empty:
                mean_sd = "NA"
                median_iqr = "NA"
            else:
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
                median_iqr = (
                    f"{median:.4f} [{q1:.4f}, {q3:.4f}]"
                )

            rows.append({
                "Dataset": scope,
                "Metric": label,
                "Model": model,
                "N": len(values),
                "Mean ± SD": mean_sd,
                "Median [Q1, Q3]": median_iqr,
            })

    table = pd.DataFrame(rows)

    print(
        "\n--- Configuration experiment: post-processing "
        f"| {scope} ---"
    )
    print(
        f"Detectability: IoU ≥ {REPORT_THRESHOLD:.2f}. "
        "Undefined scores are excluded per metric. "
        "Summaries weight cases equally."
    )
    print(table.to_string(index=False))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f"pp_config_metric_table{suffix}.csv"
    table.to_csv(path, index=False)

    print(f"Saved: {path}")

    return table


# -------------------------------------------------------------------
# Plot overall overview
# -------------------------------------------------------------------

def plot_metrics_by_dataset(df):
    datasets = [
        dataset
        for dataset in DATASET_ORDER
        if dataset in df["dataset"].unique()
    ]

    if not datasets:
        return

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        sharey=True,
    )

    for ax, (metric, label) in zip(axes.flat, METRICS.items()):
        subset = df.dropna(subset=[metric])

        if not subset.empty:
            sns.boxplot(
                data=subset,
                x="dataset",
                y=metric,
                order=datasets,
                hue="model",
                hue_order=MODEL_ORDER,
                palette=MODEL_PALETTE,
                dodge=True,
                width=0.7,
                showfliers=False,
                ax=ax,
            )

            sns.stripplot(
                data=subset,
                x="dataset",
                y=metric,
                order=datasets,
                hue="model",
                hue_order=MODEL_ORDER,
                palette={
                    model: "black"
                    for model in MODEL_ORDER
                },
                dodge=True,
                jitter=0.15,
                alpha=0.2,
                size=2.5,
                ax=ax,
            )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        ax.set_title(label, fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("Per-case score")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets)
        ax.grid(axis="x", visible=False)

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
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
    )

    fig.suptitle(
        "Configuration experiment: dual-phase validation, fold 0\n"
        "Raw versus post-processed predictions by dataset",
        fontsize=16,
        y=0.98,
    )

    fig.text(
        0.5,
        0.02,
        "Voxel level: Dice, Precision, Sensitivity. "
        f"Lesion level: Detectability (IoU ≥ {REPORT_THRESHOLD:.2f}).",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.87])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / "pp_config_all_metrics_by_dataset.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Saved: {path}")

    plt.show()
    plt.close(fig)


def main():
    df = load_combined_results()

    # Overall results and existing four-panel figure
    print_metric_table(df)
    plot_combined_metrics(df)

    # All datasets together in one figure.
    plot_metrics_by_dataset(df)

    # Per dataset
    dataset_tables = []

    for dataset in DATASET_ORDER:
        subset = df.loc[df["dataset"] == dataset].copy()

        if subset.empty:
            print(f"No cases found for {dataset}; skipping.")
            continue

        table = print_metric_table(subset, dataset=dataset)
        dataset_tables.append(table)

        plot_combined_metrics(subset, dataset=dataset)

    if dataset_tables:
        combined_table = pd.concat(
            dataset_tables,
            ignore_index=True,
        )

        path = PLOT_DIR / "pp_config_metric_table_by_dataset.csv"
        combined_table.to_csv(path, index=False)

        print(f"Saved: {path}")


if __name__ == "__main__":
    main()