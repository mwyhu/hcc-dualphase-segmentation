from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

DEFAULT_RESULTS = Path(
    "/Users/michellehu/Desktop/hcc-dualphase-segmentation/"
    "analysis/statistical/external_validation/excl_cairo/"
    "friedman_pairwise_valid_pairs_results"
)

MODELS = ["COALA", "AtlasNet", "TSLT", "TSLL", "SP", "DP"]
DISPLAY_MODELS = ["COALA", "AN", "TSLT", "TSLL", "SP", "DP"]
MODEL_TO_DISPLAY = dict(zip(MODELS, DISPLAY_MODELS))

DATASETS = ["HCC", "LiTS", "TCIA", "WORC"]

DATASET_LABELS = {
    "HCC": "HCC-TACE-Seg",
    "LiTS": "LiTS",
    "TCIA": "TCIA-CRLM",
    "WORC": "WORC-CRLM",
}

METRICS = ["dice", "precision", "recall", "detectability"]

METRIC_LABELS = {
    "dice": "Dice",
    "precision": "Precision",
    "recall": "Sensitivity",
    "detectability": "Detectability",
}

ALPHA = 0.05


# -------------------------------------------------------------------
# Read and validate results
# -------------------------------------------------------------------

def resolve_dataset_csv(results_dir: Path, dataset: str) -> Path:
    """Locate the pairwise post-hoc Wilcoxon CSV for a given dataset."""
    path = results_dir / f"posthoc_wilcoxon_{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected post-hoc Wilcoxon file not found: {path}"
        )
    return path


def load_results(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    print(f"Reading: {path.resolve()}")

    required = {
        "model_a",
        "model_b",
        "metric",
        "mean_difference_a_minus_b",
        "holm_adjusted_p_value",
        "significant_after_holm",
        "n_paired_cases",
        "case_selection",
    }

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    aliases = {model.casefold(): model for model in MODELS}
    aliases.update({
        "an": "AtlasNet",
        "atlasnet": "AtlasNet",
        "singlephase": "SP",
        "dualphase": "DP",
    })

    for column in ["model_a", "model_b"]:
        cleaned = data[column].astype(str).str.strip()
        data[column] = cleaned.map(lambda name: aliases.get(name.casefold(), name))

    data["metric"] = data["metric"].astype(str).str.strip().str.lower()

    for column in [
        "mean_difference_a_minus_b",
        "holm_adjusted_p_value",
        "n_paired_cases",
    ]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    return data


# -------------------------------------------------------------------
# Build p-value matrix and ranks from pairwise CSV
# -------------------------------------------------------------------

def build_cd_inputs(data: pd.DataFrame, metric: str, ranks_df: pd.DataFrame | None = None, dataset: str | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """
    Constructs:
      1. avg_ranks: Series indexed by DISPLAY_MODELS (lower = better)
      2. p_matrix: Symmetric DataFrame of holm_adjusted_p_value indexed by DISPLAY_MODELS
    """
    sub = data.loc[data["metric"] == metric].copy()
    if sub.empty:
        raise ValueError(f"No rows found for metric {metric!r}")

    n = len(MODELS)
    p_mat = np.ones((n, n), dtype=float)
    diff_mat = np.zeros((n, n), dtype=float)

    for row in sub.itertuples(index=False):
        i = MODELS.index(row.model_a)
        j = MODELS.index(row.model_b)

        p_val = float(row.holm_adjusted_p_value) if np.isfinite(row.holm_adjusted_p_value) else 1.0
        diff = float(row.mean_difference_a_minus_b) if np.isfinite(row.mean_difference_a_minus_b) else 0.0

        p_mat[i, j] = p_val
        p_mat[j, i] = p_val

        # diff is (model_a - model_b)
        diff_mat[i, j] = diff
        diff_mat[j, i] = -diff

    p_matrix = pd.DataFrame(p_mat, index=DISPLAY_MODELS, columns=DISPLAY_MODELS)

    # Use exact mean ranks if supplied
    if ranks_df is not None:
        r_sub = ranks_df.loc[ranks_df["metric"] == metric]
        if dataset is not None and "dataset" in r_sub.columns:
            r_sub = r_sub.loc[r_sub["dataset"] == dataset]

        avg_ranks = (
            r_sub.set_index("model")["mean_rank"]
            .reindex(MODELS)
            .rename(index=MODEL_TO_DISPLAY)
        )
        if avg_ranks.isna().any():
            raise ValueError(f"Missing mean_rank values for {dataset} / {metric}")
        return avg_ranks, p_matrix

    # Derive positions from mean pairwise advantage
    # Higher average (model_i - model_j) -> better model -> lower rank number (closer to 1)
    mean_advantage = pd.Series(diff_mat.mean(axis=1), index=DISPLAY_MODELS)
    avg_ranks = mean_advantage.rank(ascending=False, method="average")

    return avg_ranks, p_matrix


# -------------------------------------------------------------------
# Plot 4x4 Grid of CD Diagrams (Datasets x Metrics)
# -------------------------------------------------------------------

def plot_cd_grid(
    results_dir: Path,
    output_path: Path,
    ranks_df: pd.DataFrame | None = None,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=len(METRICS),
        figsize=(20, 13),
    )

    for row_idx, dataset in enumerate(DATASETS):
        csv_path = resolve_dataset_csv(results_dir, dataset)
        data = load_results(csv_path)

        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            avg_ranks, p_matrix = build_cd_inputs(
                data=data,
                metric=metric,
                ranks_df=ranks_df,
                dataset=dataset,
            )

            label_fmt = "{label} ({rank:.2f})" if ranks_df is not None else "{label}"

            sp.critical_difference_diagram(
                avg_ranks,
                p_matrix,
                ax=ax,
                alpha=ALPHA,
                label_fmt_left=label_fmt,
                label_fmt_right=label_fmt,
                text_h_margin=0.06,
                marker_props={"marker": "o", "s": 35, "color": "#2B5C8F"},
                elbow_props={"color": "#2B5C8F", "linewidth": 1.2},
                crossbar_props={"color": "#D95F02", "linewidth": 2.8},
            )

            if row_idx == 0:
                ax.set_title(
                    METRIC_LABELS[metric],
                    fontsize=13,
                    fontweight="bold",
                    pad=14,
                )

            if col_idx == 0:
                ax.text(
                    -0.12,
                    0.5,
                    DATASET_LABELS[dataset],
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90,
                )

    plt.suptitle(
        "Critical difference diagrams: model performance comparison using average ranks",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.text(0.5, -0.05, "Average ranks across 4 datasets (lower is better).\n"
                         "Horizontal bars connect models with no statistically significant difference\n"
                         "(pairwise Wilcoxon signed-rank test with Holm correction, p < 0.05 across 15 model pairs within each dataset and metric.)", ha="center", fontsize=8.5, style="italic")
    plt.tight_layout(rect=[0.03, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CD diagram grid to: {output_path.resolve()}")


# -------------------------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 4x4 Critical Difference diagrams from pairwise Wilcoxon-Holm CSVs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Directory containing dataset pairwise CSVs (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS / "cd_diagrams_4x4.png",
        help="Output path for the 4x4 CD diagram figure.",
    )
    parser.add_argument(
        "--ranks-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns [dataset, metric, model, mean_rank] for exact Friedman ranks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks_df = None

    if args.ranks_csv is not None:
        ranks_df = pd.read_csv(args.ranks_csv)
        ranks_df.columns = ranks_df.columns.str.strip()
        ranks_df["metric"] = ranks_df["metric"].astype(str).str.strip().str.lower()

    plot_cd_grid(
        results_dir=args.results_dir,
        output_path=args.output,
        ranks_df=ranks_df,
    )


if __name__ == "__main__":
    main()
