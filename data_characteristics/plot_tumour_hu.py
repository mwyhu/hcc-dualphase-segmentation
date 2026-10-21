from pathlib import Path
import argparse

import pandas as pd

from plot_characteristics import (
    load_exports,
    set_style,
    numeric_panel,
    save_figure,
)

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot phase-specific mean tumour HU per patient."
        )
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
        help="Dataset identifiers in the desired plotting order.",
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

    # Already excludes unreadable and zero-filled volumes.
    volumes = data["usable_volumes"].copy()

    volumes["tumour_mean_hu"] = pd.to_numeric(
        volumes["tumour_mean_hu"],
        errors="coerce",
    )

    # One observation per patient and phase.
    patient_hu = (
        volumes.groupby(
            ["dataset", "case_id", "phase"],
            as_index=False,
        )["tumour_mean_hu"]
        .mean()
    )

    set_style()

    figure, ax = plt.subplots(
        figsize=(max(10, len(order) * 1.25), 6),
        layout="constrained",
    )

    sample_sizes = numeric_panel(
        ax,
        patient_hu,
        order,
        "tumour_mean_hu",
        "Mean tumour attenuation (HU)",
        "Tumour attenuation by contrast phase",
        grouped=True,
    )

    for item in sample_sizes:
        item["unit_of_observation"] = "patient_phase"

    save_figure(
        figure,
        output,
        "tumour_hu_by_phase",
        args.dpi,
    )

    pd.DataFrame(sample_sizes).to_csv(
        output / "tumour_hu_sample_sizes.csv",
        index=False,
    )

    ax.set_ylim(-100, 250)

    save_figure(
        figure,
        output,
        "tumour_hu_by_phase_zoomed",
        args.dpi,
    )

    plt.close(figure)

    print("Finished.")


if __name__ == "__main__":
    main()