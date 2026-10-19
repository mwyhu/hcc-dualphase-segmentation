from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

VOXEL_DIR = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis/dice_per_dataset/final"
)

DETECTABILITY_DIR = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis/detectability/final"
)

PLOT_DIR = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "benchmark/full_benchmark/plots"
)

DATASETS = ["HCC", "LiTS", "TCIA", "WORC", "CAIRO5"]
MODELS = ["TSLL", "TSLT", "COALA", "AtlasNet", "SP", "DP"]

DATASET_LABELS = {
    "HCC": "HCC-TACE-Seg",
    "LiTS": "LiTS",
    "TCIA": "TCIA-CRLM",
    "WORC": "WORC-CRLM",
    "CAIRO5": "CAIRO5",
}

DETECTABILITY_MODEL_CODES = {
    "TSLL": "TSLL",
    "TSLT": "TSLT",
    "COALA": "COALA",
    "AtlasNet": "AN",
    "SP": "SP",
    "DP": "DP",
}

METRICS = {
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Sensitivity",
    "detectability": "Detectability",
}

MODEL_PALETTE = {
    "TSLL": "#4C78A8",
    "TSLT": "#F58518",
    "COALA": "#72B7B2",
    "AtlasNet": "#54A24B",
    "SP": "#B279A2",
    "DP": "#E45756",
}

REPORT_THRESHOLD = 0.20

sns.set_theme(style="whitegrid")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)


# -------------------------------------------------------------------
# Case IDs
# -------------------------------------------------------------------

def case_id(value):
    """Remove directory and NIfTI extension."""
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]

    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[:-len(suffix)]

    return name


# -------------------------------------------------------------------
# Load voxel metrics from nnU-Net summaries
# -------------------------------------------------------------------

def load_voxel_results():
    rows = []

    for dataset in DATASETS:
        for model in MODELS:
            path = VOXEL_DIR / f"{model}_{dataset}_summary.json"

            if not path.is_file():
                raise FileNotFoundError(f"Missing summary: {path}")

            with path.open() as file:
                results = json.load(file)

            cases = results.get("metric_per_case")

            if not cases:
                raise ValueError(f"No per-case results in {path}")

            for result in cases:
                metrics = result["metrics"]["1"]

                tp = float(metrics["TP"])
                fp = float(metrics["FP"])
                fn = float(metrics["FN"])

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "case": case_id(result["prediction_file"]),
                    "dice": float(metrics["Dice"]),
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

    df = pd.DataFrame(rows)

    keys = ["dataset", "model", "case"]
    duplicates = df.duplicated(keys, keep=False)

    if duplicates.any():
        raise ValueError(
            "Duplicate voxel results:\n"
            + df.loc[duplicates, keys].head(20).to_string(index=False)
        )

    return df


# -------------------------------------------------------------------
# Load lesion detectability
# -------------------------------------------------------------------

def load_detectability_results():
    frames = []

    for dataset in DATASETS:
        for model in MODELS:
            csv_model = DETECTABILITY_MODEL_CODES[model]

            path = DETECTABILITY_DIR / (
                f"Detect_{csv_model}_{dataset}.csv"
            )

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing detectability CSV: {path}"
                )

            df = pd.read_csv(path)

            required = {"case", "detectability", "iou_threshold"}
            missing = required - set(df.columns)

            if missing:
                raise ValueError(
                    f"{path.name}: missing columns {sorted(missing)}"
                )

            thresholds = pd.to_numeric(
                df["iou_threshold"],
                errors="raise",
            )

            selected = df.loc[
                np.isclose(
                    thresholds,
                    REPORT_THRESHOLD,
                    rtol=0,
                    atol=1e-8,
                ),
                ["case", "detectability"],
            ].copy()

            if selected.empty:
                raise ValueError(
                    f"{path.name}: no results at "
                    f"IoU {REPORT_THRESHOLD:.2f}"
                )

            if selected["case"].isna().any():
                raise ValueError(
                    f"{path.name}: missing case IDs"
                )

            selected["case"] = selected["case"].map(case_id)

            selected["detectability"] = pd.to_numeric(
                selected["detectability"],
                errors="raise",
            )

            values = selected["detectability"].dropna()

            if not values.between(0, 1).all():
                raise ValueError(
                    f"{path.name}: detectability outside [0, 1]"
                )

            selected["dataset"] = dataset
            selected["model"] = model

            frames.append(selected)

    return pd.concat(frames, ignore_index=True)


# -------------------------------------------------------------------
# Match voxel metrics and detectability
# -------------------------------------------------------------------

def combine_results(voxel_df, detectability_df):
    keys = ["dataset", "model", "case"]

    combined = voxel_df.merge(
        detectability_df,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = combined.loc[
        combined["_merge"] != "both",
        keys + ["_merge"],
    ]

    if not unmatched.empty:
        raise ValueError(
            "Case mismatch between voxel summaries "
            "and detectability CSVs:\n"
            + unmatched.head(20).to_string(index=False)
        )

    return combined.drop(columns="_merge")


def check_case_sets(df):
    reference_model = MODELS[0]

    for dataset in DATASETS:
        subset = df.loc[df["dataset"] == dataset]

        reference_cases = set(
            subset.loc[
                subset["model"] == reference_model,
                "case",
            ]
        )

        for model in MODELS[1:]:
            model_cases = set(
                subset.loc[
                    subset["model"] == model,
                    "case",
                ]
            )

            if model_cases != reference_cases:
                raise ValueError(
                    f"Case mismatch for {dataset}/{model}.\n"
                    f"Missing: "
                    f"{sorted(reference_cases - model_cases)[:10]}\n"
                    f"Extra: "
                    f"{sorted(model_cases - reference_cases)[:10]}"
                )


# -------------------------------------------------------------------
# Summary tables
# -------------------------------------------------------------------

def create_summary(df):
    rows = []

    for dataset in DATASETS:
        for model in MODELS:
            subset = df.loc[
                (df["dataset"] == dataset)
                & (df["model"] == model)
            ]

            for metric, label in METRICS.items():
                values = subset[metric].dropna()

                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)

                rows.append({
                    "Dataset": DATASET_LABELS[dataset],
                    "Model": model,
                    "Metric": label,
                    "N": len(values),
                    "Mean": values.mean(),
                    "SD": values.std(),
                    "Median": values.median(),
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": q3 - q1,
                })

    summary = pd.DataFrame(rows)

    formatted = summary[
        ["Dataset", "Model", "Metric", "N"]
    ].copy()

    formatted["Mean ± SD"] = summary.apply(
        lambda row: (
            f"{row['Mean']:.4f} ± {row['SD']:.4f}"
        ),
        axis=1,
    )

    formatted["Median [Q1, Q3]"] = summary.apply(
        lambda row: (
            f"{row['Median']:.4f} "
            f"[{row['Q1']:.4f}, {row['Q3']:.4f}]"
        ),
        axis=1,
    )

    formatted["IQR"] = summary["IQR"].map(
        lambda value: f"{value:.4f}"
    )

    for dataset in DATASETS:
        label = DATASET_LABELS[dataset]

        print(f"\n--- {label} ---")
        print(
            formatted.loc[
                formatted["Dataset"] == label
            ].to_string(index=False)
        )

    summary.to_csv(
        PLOT_DIR / "per_dataset_metric_summary.csv",
        index=False,
    )

    formatted.to_csv(
        PLOT_DIR / "per_dataset_metric_summary_formatted.csv",
        index=False,
    )


# -------------------------------------------------------------------
# One figure per dataset: four metrics in one row
# -------------------------------------------------------------------

def plot_dataset(df, dataset):
    dataset_df = df.loc[df["dataset"] == dataset]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(23, 5.5),
        sharey=True,
    )

    rng = np.random.default_rng(42)

    for ax, (metric, label) in zip(axes, METRICS.items()):
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

        # Individual case scores, including outliers.
        for position, model in enumerate(MODELS):
            values = plot_df.loc[
                plot_df["model"] == model,
                metric,
            ].to_numpy(dtype=float)

            ax.scatter(
                position + rng.uniform(
                    -0.2,
                    0.2,
                    size=len(values),
                ),
                values,
                color="black",
                alpha=0.25,
                s=8,
                linewidths=0,
                zorder=3,
            )

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(
            MODELS,
            rotation=30,
            ha="right",
        )

        ax.set_title(label, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Score")

    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: model comparison",
        fontsize=16,
    )

    fig.text(
        0.5,
        0.015,
        "Voxel level: Dice, Precision, Sensitivity. "
        f"Lesion level: Detectability (IoU ≥ {REPORT_THRESHOLD:.2f}). "
        "Undefined scores excluded per metric.",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.94])

    filename = (
        f"{dataset}_dice_precision_sensitivity_detectability"
    )

    for extension in ("png", "pdf"):
        path = PLOT_DIR / f"{filename}.{extension}"

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
# Combined CRLM: equal weight per dataset
# -------------------------------------------------------------------

CRLM_DATASETS = ["TCIA", "WORC", "CAIRO5"]


def weighted_quantile(values, weights, probabilities):
    """Quantiles of the weighted empirical distribution."""
    order = np.argsort(values)

    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]

    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]

    indices = np.searchsorted(
        cumulative,
        probabilities,
        side="left",
    )

    return values[np.clip(indices, 0, len(values) - 1)]


def plot_combined_crlm(df):
    crlm_df = df.loc[df["dataset"].isin(CRLM_DATASETS)]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(23, 5.8),
        sharey=True,
    )

    rng = np.random.default_rng(42)
    summary_rows = []

    for ax, (metric, label) in zip(axes, METRICS.items()):
        boxes = []
        macro_means = []
        model_values = []

        for model in MODELS:
            parts = []
            weight_parts = []
            dataset_means = []

            for dataset in CRLM_DATASETS:
                values = crlm_df.loc[
                    (crlm_df["dataset"] == dataset)
                    & (crlm_df["model"] == model),
                    metric,
                ].to_numpy(dtype=float)

                values = values[np.isfinite(values)]

                if len(values) == 0:
                    raise ValueError(
                        f"No valid scores for {dataset}/{model}/{metric}"
                    )

                parts.append(values)

                # Each dataset contributes one third of the total weight.
                weight_parts.append(
                    np.full(
                        len(values),
                        1 / (len(CRLM_DATASETS) * len(values)),
                    )
                )

                dataset_means.append(values.mean())

            values = np.concatenate(parts)
            weights = np.concatenate(weight_parts)

            q1, median, q3 = weighted_quantile(
                values,
                weights,
                [0.25, 0.50, 0.75],
            )

            iqr = q3 - q1

            boxes.append({
                "q1": float(q1),
                "med": float(median),
                "q3": float(q3),
                "whislo": float(
                    values[values >= q1 - 1.5 * iqr].min()
                ),
                "whishi": float(
                    values[values <= q3 + 1.5 * iqr].max()
                ),
                "fliers": [],
            })

            macro_mean = float(np.mean(dataset_means))

            macro_means.append(macro_mean)
            model_values.append(values)

            summary_rows.append({
                "Group": "CRLM",
                "Model": model,
                "Metric": label,
                "N": len(values),
                "Equal_dataset_mean": macro_mean,
                "Weighted_median": median,
                "Weighted_Q1": q1,
                "Weighted_Q3": q3,
                "Weighted_IQR": iqr,
            })

        positions = np.arange(len(MODELS))

        artists = ax.bxp(
            boxes,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops={
                "color": "#222222",
                "linewidth": 1.5,
            },
        )

        for box, model in zip(artists["boxes"], MODELS):
            box.set_facecolor(MODEL_PALETTE[model])
            box.set_alpha(0.8)

        for position, values in zip(positions, model_values):
            ax.scatter(
                position + rng.uniform(
                    -0.2,
                    0.2,
                    size=len(values),
                ),
                values,
                color="black",
                alpha=0.20,
                s=8,
                linewidths=0,
                zorder=3,
            )

        ax.scatter(
            positions,
            macro_means,
            marker="D",
            color="black",
            edgecolors="white",
            linewidths=0.8,
            s=55,
            zorder=5,
            label="Equal-dataset mean",
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            MODELS,
            rotation=30,
            ha="right",
        )

        ax.set_title(label, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Score")

    fig.suptitle(
        "CRLM: model comparison",
        fontsize=16,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        frameon=False,
    )

    fig.text(
        0.5,
        0.015,
        "Metrics give WORC-CRLM, TCIA-CRLM and CAIRO5 equal weight. "
        "Voxel level: Dice, Precision, Sensitivty. Lesion level: "
        f"Detectability (IoU ≥ {REPORT_THRESHOLD:.2f}).",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.19, 1, 0.94])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    filename = "CRLM_dice_precision_sensitivity_detectability"

    for extension in ("png", "pdf"):
        path = PLOT_DIR / f"{filename}.{extension}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Saved: {path}")

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        PLOT_DIR / "CRLM_equal_dataset_summary.csv",
        index=False,
    )

    print("\n--- Combined CRLM: equal dataset weighting ---")
    print(summary.round(4).to_string(index=False))

    plt.show()
    plt.close(fig)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    voxel_df = load_voxel_results()
    detectability_df = load_detectability_results()

    df = combine_results(
        voxel_df,
        detectability_df,
    )

    check_case_sets(df)
    create_summary(df)

    for dataset in DATASETS:
        plot_dataset(df, dataset)

    plot_combined_crlm(df)


if __name__ == "__main__":
    main()