from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

DATASET_LABELS = {
    "WAW_TACE_grid_fixed": "WAW-TACE",
    "WAW_TACE": "WAW-TACE",
    "HCC_TACE": "HCC-TACE-Seg",
    "LiTS": "LiTS",
    "Liver_Lesions": "Liver Lesions",
    "MCT_LTDiag_grid_fixed": "MCT-LTDiag",
    "MCT_LTDiag": "MCT-LTDiag",
    "TCIA_CRLM": "TCIA-CRLM",
    "WORC_CRLM": "WORC-CRLM",
    "MSD": "MSD",
    "CAIRO5": "CAIRO5",
}

PHASE_COLORS = {
    "ART": "#f07814",
    "PV": "#003741",
}

PHASE_PATTERNS = [
    "ART_only",
    "PV_only",
    "ART+PV",
    "no_valid_phase",
]

PATTERN_LABELS = [
    "ART only",
    "PV only",
    "ART + PV",
    "No valid phase",
]

PATTERN_COLORS = [
    "#f07814",
    "#003741",
    "#77ada1",
    "#c6c6c6",
]


# -------------------------------------------------------------------
# Read and prepare CSV files
# -------------------------------------------------------------------

def parse_boolean(series):
    values = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    invalid = (
        ~values.isin(["true", "false", "1", "0"])
        & values.notna()
    )

    if invalid.any():
        raise ValueError(
            f"Invalid boolean values in {series.name}: "
            f"{values[invalid].unique()}"
        )

    return values.isin(["true", "1"])


def load_exports(root, selected=None):
    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError(root)

    if (root / "cases.csv").is_file():
        folders = [root]
    else:
        folders = sorted(
            folder
            for folder in root.iterdir()
            if folder.is_dir()
            and (folder / "cases.csv").is_file()
        )

    if not folders:
        raise ValueError(
            f"No dataset folders containing cases.csv found in {root}"
        )

    tables = {
        "cases": [],
        "volumes": [],
        "lesions": [],
    }

    for folder in folders:
        for name in tables:
            path = folder / f"{name}.csv"

            if not path.is_file():
                raise FileNotFoundError(path)

            dataframe = pd.read_csv(
                path,
                dtype={
                    "dataset": str,
                    "case_id": str,
                },
            )

            if not {"dataset", "case_id"} <= set(dataframe.columns):
                raise ValueError(
                    f"Missing dataset/case_id columns: {path}"
                )

            tables[name].append(dataframe)

    tables = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in tables.items()
    }

    available = list(
        dict.fromkeys(tables["cases"]["dataset"])
    )

    order = selected or available

    if len(order) != len(set(order)):
        raise ValueError("--datasets contains repeated names.")

    missing = set(order) - set(available)

    if missing:
        raise ValueError(
            f"Datasets not found: {sorted(missing)}. "
            f"Available dataset identifiers: {available}"
        )

    tables = {
        name: dataframe[
            dataframe["dataset"].isin(order)
        ].copy()
        for name, dataframe in tables.items()
    }

    cases = tables["cases"]

    if cases.duplicated(["dataset", "case_id"]).any():
        raise ValueError(
            "Duplicate dataset/case_id rows found. "
            "Include only one export per patient and dataset."
        )

    # Missing/unreadable masks must not contribute zero lesions.
    if "seg_available" in cases:
        label_column = "seg_available"
    elif "label_available" in cases:
        label_column = "label_available"
    else:
        raise ValueError(
            "cases.csv needs seg_available or label_available."
        )

    labelled = cases[
        parse_boolean(cases[label_column])
    ].copy()

    labelled["n_lesions"] = pd.to_numeric(
        labelled["n_lesions"],
        errors="coerce",
    )

    tables["labelled_cases"] = labelled

    # Exclude zero-filled and unreadable CT volumes.
    volumes = tables["volumes"]

    usable = volumes[
        parse_boolean(volumes["read_ok"])
        & ~parse_boolean(volumes["zero_filled"])
    ].copy()

    # Compatibility with earlier extraction script.
    if "n_slices" not in usable and "slices" in usable:
        usable = usable.rename(
            columns={"slices": "n_slices"}
        )

    tables["usable_volumes"] = usable

    return tables, order


# -------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------

def set_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def save_figure(figure, output, name, dpi):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for extension in ["png", "pdf"]:
        path = output / f"{name}.{extension}"

        figure.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(f"Saved: {path}")


def numeric_values(dataframe, column, positive=False):
    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).to_numpy(dtype=float)

    values = values[np.isfinite(values)]

    if positive:
        values = values[values > 0]

    return values


def numeric_panel(
    ax,
    dataframe,
    order,
    column,
    ylabel,
    title,
    grouped=False,
    log=False,
):
    rng = np.random.default_rng(42)
    sample_sizes = []

    for index, dataset in enumerate(order):
        phases = ["ART", "PV"] if grouped else [None]

        for phase in phases:
            subset = dataframe[
                dataframe["dataset"].eq(dataset)
            ]

            if phase is not None:
                subset = subset[
                    subset["phase"].eq(phase)
                ]

            values = numeric_values(
                subset,
                column,
                positive=log,
            )

            sample_sizes.append({
                "dataset": dataset,
                "panel": title,
                "phase": phase,
                "n": len(values),
                "unit_of_observation": (
                    "lesion"
                    if column == "volume_ml"
                    else "volume"
                    if grouped
                    else "patient"
                ),
            })

            if grouped:
                offset = -0.18 if phase == "ART" else 0.18
                position = index + offset
                width = 0.29
                color = PHASE_COLORS[phase]
            else:
                position = index
                width = 0.53
                color = "#77ada1"

            if len(values) == 0:
                continue

            boxplot = ax.boxplot(
                [values],
                positions=[position],
                widths=width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                medianprops={
                    "color": "#222222",
                    "linewidth": 1.4,
                },
                whiskerprops={"color": "#555555"},
                capprops={"color": "#555555"},
            )

            boxplot["boxes"][0].set(
                facecolor=color,
                alpha=0.65,
                edgecolor="#444444",
            )

            jitter = rng.uniform(
                -width * 0.29,
                width * 0.29,
                len(values),
            )

            # All observations are shown; outliers are not hidden.
            ax.scatter(
                position + jitter,
                values,
                s=7,
                color="#333333",
                alpha=0.20,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )

    ax.set_xticks(range(len(order)))

    if grouped:
        labels = [
            DATASET_LABELS.get(dataset, dataset)
            for dataset in order
        ]

        ax.legend(
            handles=[
                Patch(
                    facecolor=PHASE_COLORS[phase],
                    label=phase,
                )
                for phase in ["ART", "PV"]
            ],
            loc="best",
            frameon=False,
        )

    else:
        counts = {
            item["dataset"]: item["n"]
            for item in sample_sizes
        }

        labels = [
            (
                f"{DATASET_LABELS.get(dataset, dataset)}"
                f"\n(n={counts[dataset]})"
            )
            for dataset in order
        ]

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_xlim(-0.65, len(order) - 0.35)

    has_values = any(
        item["n"] > 0 for item in sample_sizes
    )

    if log and has_values:
        ax.set_yscale("log")
    elif column == "n_lesions":
        ax.set_ylim(bottom=-0.5)
    elif column == "n_slices":
        ax.set_ylim(bottom=0)

    if not has_values:
        ax.text(
            0.5,
            0.5,
            "No available measurements",
            transform=ax.transAxes,
            ha="center",
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)

    return sample_sizes


def phase_panel(ax, cases, order, percent=False):
    cases = cases.copy()

    cases["phase_pattern"] = cases["phase_pattern"].replace({
        "unknown_or_missing": "no_valid_phase",
    })

    if not cases["phase_pattern"].isin(PHASE_PATTERNS).all():
        raise ValueError(
            "Unexpected phase_pattern values in cases.csv."
        )

    totals = np.array([
        cases["dataset"].eq(dataset).sum()
        for dataset in order
    ])

    bottom = np.zeros(len(order))
    sample_sizes = []

    for pattern, label, color in zip(
        PHASE_PATTERNS,
        PATTERN_LABELS,
        PATTERN_COLORS,
    ):
        counts = np.array([
            (
                cases["dataset"].eq(dataset)
                & cases["phase_pattern"].eq(pattern)
            ).sum()
            for dataset in order
        ])

        heights = (
            counts / totals * 100
            if percent else counts
        )

        bars = ax.bar(
            range(len(order)),
            heights,
            bottom=bottom,
            color=color,
            label=label,
            width=0.7,
        )

        for bar, count in zip(bars, counts):
            if count == 0:
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if pattern == "PV_only"
                    else "#222222"
                ),
            )

        for dataset, count in zip(order, counts):
            sample_sizes.append({
                "dataset": dataset,
                "panel": "A. Phase availability",
                "phase": pattern,
                "n": int(count),
                "unit_of_observation": "patient",
            })

        bottom += heights

    labels = [
        (
            f"{DATASET_LABELS.get(dataset, dataset)}"
            f"\n(n={total})"
        )
        for dataset, total in zip(order, totals)
    ]

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Patients (%)"
        if percent else "Number of patients"
    )

    ax.set_title(
        "A. Phase availability",
        loc="left",
    )

    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper left",
    )

    ax.set_ylim(
        0,
        122 if percent else max(totals.max() * 1.3, 1),
    )

    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)

    return sample_sizes


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot dataset characteristics from CSV exports."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_characteristics"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help=(
            "Dataset identifiers from the CSVs, "
            "in the desired plotting order."
        ),
    )

    parser.add_argument(
        "--phase-percent",
        action="store_true",
        help=(
            "Use percentage heights for phase bars. "
            "Segment labels still show patient counts."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    output = (
        args.output_dir
        if args.output_dir
        else args.input_dir / "characteristics_figures"
    )

    data, order = load_exports(
        args.input_dir,
        args.datasets,
    )

    set_style()

    panels = [
        (
            "phase_availability",
            lambda ax: phase_panel(
                ax,
                data["cases"],
                order,
                percent=args.phase_percent,
            ),
        ),
        (
            "lesions_per_patient",
            lambda ax: numeric_panel(
                ax,
                data["labelled_cases"],
                order,
                "n_lesions",
                "Number of lesions",
                "B. Lesions per patient",
            ),
        ),
        (
            "lesion_volume",
            lambda ax: numeric_panel(
                ax,
                data["lesions"],
                order,
                "volume_ml",
                "Lesion volume (mL; log scale)",
                "C. Individual lesion volume",
                log=True,
            ),
        ),
        (
            "slices_by_phase",
            lambda ax: numeric_panel(
                ax,
                data["usable_volumes"],
                order,
                "n_slices",
                "Number of slices per CT volume",
                "D. Slices by phase",
                grouped=True,
            ),
        ),
    ]

    # Combined four-panel figure.
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(max(15, len(order) * 1.6), 11),
        layout="constrained",
    )

    sample_sizes = []

    for ax, (_, draw_panel) in zip(axes.flat, panels):
        sample_sizes.extend(draw_panel(ax))

    figure.suptitle(
        "Dataset characteristics",
        fontsize=17,
    )

    save_figure(
        figure,
        output,
        "characteristics_overview",
        args.dpi,
    )

    plt.close(figure)

    # Separate characteristics_figures.
    for name, draw_panel in panels:
        figure, ax = plt.subplots(
            figsize=(max(9, len(order) * 1.2), 6),
            layout="constrained",
        )

        draw_panel(ax)

        save_figure(
            figure,
            output,
            name,
            args.dpi,
        )

        plt.close(figure)

    pd.DataFrame(sample_sizes).to_csv(
        output / "plotted_sample_sizes.csv",
        index=False,
    )

    lesion_volumes = pd.to_numeric(
        data["lesions"]["volume_ml"],
        errors="coerce",
    )

    if (lesion_volumes <= 0).any():
        warnings.warn(
            "Nonpositive lesion volumes were excluded "
            "from the logarithmic plot."
        )

    print("Finished.")


if __name__ == "__main__":
    main()