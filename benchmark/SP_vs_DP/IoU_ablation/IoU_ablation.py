from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

BASE_DIR = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis"
)

DETECT_FILES = {
    model: BASE_DIR
    / "detectability"
    / "internal_testset"
    / f"Detect_{model}_internal.csv"
    for model in ["SP", "DP"]
}

MODEL_ORDER = ["SP", "DP"]

MODEL_PALETTE = {
    "SP": "#033943",
    "DP": "#f0781a",
}

REPORT_THRESHOLDS = [0.15, 0.20, 0.50]

PLOT_DIR = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "benchmark/SP_vs_DP/IoU_ablation/plots"
)

sns.set_theme(style="whitegrid")


def case_id(value):
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    return name.removesuffix(".nii.gz").removesuffix(".nii")


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

        for threshold in REPORT_THRESHOLDS:
            subset = df.loc[
                np.isclose(
                    df["iou_threshold"],
                    threshold,
                    rtol=0,
                    atol=1e-8,
                )
            ].copy()

            if subset.empty:
                raise ValueError(
                    f"{path.name}: no rows at IoU "
                    f"{threshold:.2f}"
                )

            if subset["case"].isna().any():
                raise ValueError(
                    f"{path.name}: missing case IDs "
                    f"at IoU {threshold:.2f}"
                )

            subset["case"] = subset["case"].map(case_id)

            subset["detectability"] = pd.to_numeric(
                subset["detectability"],
                errors="raise",
            )

            values = subset["detectability"].dropna()

            if not values.between(0, 1).all():
                raise ValueError(
                    f"{path.name}: detectability scores must "
                    "be between 0 and 1"
                )

            subset["model"] = model
            subset["iou_threshold"] = threshold

            frames.append(
                subset[[
                    "model",
                    "case",
                    "iou_threshold",
                    "detectability",
                ]]
            )

    results = pd.concat(frames, ignore_index=True)

    if results.duplicated(
        ["model", "case", "iou_threshold"]
    ).any():
        raise ValueError(
            "Duplicate model/case/threshold scores found"
        )

    return results


# -------------------------------------------------------------------
# Pairing across models and thresholds
# -------------------------------------------------------------------

def prepare_paired_results(df):
    expected_cases = set(df["case"])

    for threshold in REPORT_THRESHOLDS:
        for model in MODEL_ORDER:
            cases = set(
                df.loc[
                    (df["iou_threshold"] == threshold)
                    & (df["model"] == model),
                    "case",
                ]
            )

            if cases != expected_cases:
                raise ValueError(
                    f"{model}, IoU {threshold:.2f}: "
                    "case IDs differ across models or thresholds.\n"
                    f"Missing cases: "
                    f"{sorted(expected_cases - cases)[:10]}"
                )

    paired = df.pivot(
        index="case",
        columns=["iou_threshold", "model"],
        values="detectability",
    )

    expected_columns = pd.MultiIndex.from_product(
        [REPORT_THRESHOLDS, MODEL_ORDER],
        names=["iou_threshold", "model"],
    )

    paired = paired.reindex(columns=expected_columns)

    total_cases = len(paired)

    # Exclude a patient from every comparison if any score is undefined
    paired = paired.dropna()

    excluded_cases = total_cases - len(paired)

    if paired.empty:
        raise ValueError(
            "No patients have valid scores for both models "
            "at all three thresholds"
        )

    print(
        f"\nPatients included: {len(paired)}"
        f"\nPatients excluded due to undefined scores: "
        f"{excluded_cases}"
    )

    return paired


def summarise_results(paired):
    rows = []

    for threshold in REPORT_THRESHOLDS:
        sp = paired[(threshold, "SP")]
        dp = paired[(threshold, "DP")]

        rows.append({
            "iou_threshold": threshold,
            "n": len(paired),
            "sp_mean": sp.mean(),
            "sp_sd": sp.std(ddof=1),
            "dp_mean": dp.mean(),
            "dp_sd": dp.std(ddof=1),
            "difference_pp": 100 * (dp - sp).mean(),
        })

    return pd.DataFrame(rows)


def format_mean_sd(mean, sd):
    if pd.isna(sd):
        return f"{mean:.4f} ± NA"

    return f"{mean:.4f} ± {sd:.4f}"


def print_and_save_table(summary):
    rows = []

    for _, row in summary.iterrows():
        rows.append({
            "IoU threshold": f"{row['iou_threshold']:.4f}",
            "SP detectability, mean ± SD": format_mean_sd(
                row["sp_mean"],
                row["sp_sd"],
            ),
            "DP detectability, mean ± SD": format_mean_sd(
                row["dp_mean"],
                row["dp_sd"],
            ),
            "DP − SP (percentage points)": (
                f"{row['difference_pp']:+.4f}"
            ),
        })

    table = pd.DataFrame(rows)

    n = int(summary["n"].iloc[0])

    print("\n--- IoU-threshold sensitivity analysis ---")
    print(table.to_string(index=False))
    print(
        f"\nN = {n} patients, paired across models and thresholds."
        "\nDetectability is summarised across patients."
        "\nPositive DP − SP values favour DP."
    )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    path = PLOT_DIR / (
        "SP_DP_detectability_threshold_comparison.csv"
    )

    table.to_csv(path, index=False)

    print(f"\nSaved: {path}")


# -------------------------------------------------------------------
# Plot mean detectability and per-patient distributions
# -------------------------------------------------------------------

def plot_detectability_thresholds(summary, paired):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True,
    )

    # Mean detectability across thresholds
    for model in MODEL_ORDER:
        axes[0].plot(
            summary["iou_threshold"],
            summary[f"{model.lower()}_mean"],
            label=model,
            color=MODEL_PALETTE[model],
            marker="o",
            markersize=7,
            linewidth=2,
        )

    axes[0].set_title("Mean detectability")
    axes[0].set_xlabel("IoU threshold")
    axes[0].set_ylabel("Detectability")

    axes[0].set_xticks(REPORT_THRESHOLDS)
    axes[0].set_xticklabels(
        [f"{threshold:.2f}" for threshold in REPORT_THRESHOLDS]
    )

    axes[0].margins(x=0.08)

    # paired scores; one row per patient/model/threshold
    frames = []

    for threshold in REPORT_THRESHOLDS:
        for model in MODEL_ORDER:
            frames.append(
                pd.DataFrame({
                    "case": paired.index,
                    "iou_threshold": threshold,
                    "model": model,
                    "detectability": (
                        paired[(threshold, model)].to_numpy()
                    ),
                })
            )

    plot_df = pd.concat(frames, ignore_index=True)

    # Per-patient score distributions.
    sns.boxplot(
        data=plot_df,
        x="iou_threshold",
        y="detectability",
        order=REPORT_THRESHOLDS,
        hue="model",
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        width=0.65,
        whis=1.5,
        showfliers=True,
        fliersize=3,
        ax=axes[1],
    )

    axes[1].set_title("Per-patient distribution")
    axes[1].set_xlabel("IoU threshold")
    axes[1].set_ylabel("")

    axes[1].set_xticks(range(len(REPORT_THRESHOLDS)))
    axes[1].set_xticklabels(
        [f"{threshold:.2f}" for threshold in REPORT_THRESHOLDS]
    )

    handles, labels = axes[0].get_legend_handles_labels()

    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "Internal test set: IoU-threshold sensitivity analysis",
        fontsize=14,
    )

    fig.legend(
        handles,
        labels,
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=len(MODEL_ORDER),
        frameon=False,
    )

    fig.text(
        0.5,
        0.02,
        f"The same {len(paired)} patients contribute to both panels "
        "at every threshold.\n"
        "Boxes show the median and interquartile range. ",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.23, 1, 0.94])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        path = PLOT_DIR / (
            f"SP_DP_detectability_threshold_comparison.{extension}"
        )

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
    df = load_detectability_results()
    paired = prepare_paired_results(df)
    summary = summarise_results(paired)

    print_and_save_table(summary)
    plot_detectability_thresholds(summary, paired)


if __name__ == "__main__":
    main()