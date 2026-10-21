from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Dataset names and grouping
# -------------------------------------------------------------------

# Identifiers are matched against the "dataset" column in summary.csv.
DATASET_ALIASES = {
    "WAW_TACE": "WAW-TACE",
    "WAW_TACE_grid_fixed": "WAW-TACE",
    "MSD": "MSD",
    "MCT_LTDiag": "MCT-LTDiag",
    "MCT_LTDiag_grid_fixed": "MCT-LTDiag",
    "Liver_Lesions": "Liver Lesions",
    "HCC_TACE": "HCC-TACE-Seg",
    "LiTS": "LiTS",
    "TCIA_CRLM": "TCIA-CRLM",
    "WORC_CRLM": "WORC-CRLM",
    "CAIRO5": "CAIRO5",
}

GROUPS = {
    "original": [
        "WAW-TACE",
        "MSD",
        "MCT-LTDiag",
        "Liver Lesions",
    ],
    "external": [
        "HCC-TACE-Seg",
        "LiTS",
        "TCIA-CRLM",
        "WORC-CRLM",
        "CAIRO5",
    ],
}


# metric, display label, formatting, decimal places
MAIN_ROWS = [
    (
        "patients",
        "Patients, n",
        "count",
        0,
    ),
    (
        "patients_with_annotated_tumours",
        "With annotated tumours, n (%)",
        "percentage",
        0,
    ),
    (
        "patients_ART_only",
        "ART only, n (%)",
        "percentage",
        0,
    ),
    (
        "patients_PV_only",
        "PV only, n (%)",
        "percentage",
        0,
    ),
    (
        "patients_ART+PV",
        "ART and PV, n (%)",
        "percentage",
        0,
    ),
    (
        "total_annotated_lesions",
        "Total annotated lesions, n",
        "count",
        0,
    ),
    (
        "lesions_per_tumour_positive_patient",
        "Lesions per tumour-positive patient",
        "continuous",
        1,
    ),
    (
        "individual_lesion_volume_lesion_weighted",
        "Individual lesion volume (mL)",
        "continuous",
        2,
    ),
    (
        "total_tumour_volume_per_tumour_positive_patient",
        "Total tumour volume per tumour-positive patient (mL)",
        "continuous",
        2,
    ),
    (
        "ART_spacing_z_mm",
        "ART slice spacing (mm)",
        "continuous",
        2,
    ),
    (
        "PV_spacing_z_mm",
        "PV slice spacing (mm)",
        "continuous",
        2,
    ),
]

APPENDIX_ROWS = [
    (
        "patients_without_annotated_tumours",
        "Without annotated tumours, n (%)",
        "percentage",
        0,
    ),
    (
        "patients_with_unknown_tumour_status",
        "Unknown tumour status, n (%)",
        "percentage",
        0,
    ),
    (
        "patients_no_valid_phase",
        "No valid CT phase, n (%)",
        "percentage",
        0,
    ),
    (
        "ART_n_slices",
        "ART slices per CT volume",
        "continuous",
        0,
    ),
    (
        "PV_n_slices",
        "PV slices per CT volume",
        "continuous",
        0,
    ),
    (
        "ART_spacing_x_mm",
        "ART voxel spacing, x (mm)",
        "continuous",
        2,
    ),
    (
        "ART_spacing_y_mm",
        "ART voxel spacing, y (mm)",
        "continuous",
        2,
    ),
    (
        "PV_spacing_x_mm",
        "PV voxel spacing, x (mm)",
        "continuous",
        2,
    ),
    (
        "PV_spacing_y_mm",
        "PV voxel spacing, y (mm)",
        "continuous",
        2,
    ),
    (
        "ART_tumour_mean_hu",
        "ART mean tumour attenuation (HU)",
        "continuous",
        1,
    ),
    (
        "PV_tumour_mean_hu",
        "PV mean tumour attenuation (HU)",
        "continuous",
        1,
    ),
]


# -------------------------------------------------------------------
# Read summary files
# -------------------------------------------------------------------

def load_summaries(root):
    if not root.is_dir():
        raise FileNotFoundError(root)

    if (root / "summary.csv").is_file():
        paths = [root / "summary.csv"]
    else:
        paths = sorted(root.glob("*/summary.csv"))

    if not paths:
        raise ValueError(
            f"No summary.csv files found under {root}"
        )

    frames = []

    for path in paths:
        frame = pd.read_csv(
            path,
            dtype={"dataset": str},
        )

        if not {"dataset", "metric"} <= set(frame.columns):
            raise ValueError(
                f"Missing dataset/metric columns: {path}"
            )

        frame["source_file"] = str(path)
        frames.append(frame)

    summary = pd.concat(frames, ignore_index=True)

    # Convert raw dataset identifiers to thesis display names.
    summary["display_dataset"] = summary["dataset"].map(
        DATASET_ALIASES
    )

    # Also accept identifiers already using the display names.
    known_names = {
        name
        for names in GROUPS.values()
        for name in names
    }

    already_named = summary["dataset"].isin(known_names)

    summary.loc[
        already_named, "display_dataset"
    ] = summary.loc[already_named, "dataset"]

    unknown = summary.loc[
        summary["display_dataset"].isna(),
        "dataset",
    ].unique()

    if len(unknown):
        print(
            "Ignoring unrecognised dataset identifiers: "
            + ", ".join(map(str, unknown))
        )
        print(
            "Add them to DATASET_ALIASES if they should be included."
        )

    summary = summary[
        summary["display_dataset"].notna()
    ].copy()

    # Prevent accidentally combining test exports and full exports.
    duplicates = summary.duplicated(
        ["display_dataset", "metric"],
        keep=False,
    )

    if duplicates.any():
        details = summary.loc[
            duplicates,
            ["dataset", "metric", "source_file"],
        ]

        raise ValueError(
            "Multiple summaries found for the same dataset/metric. "
            "Keep one complete raw-data export per dataset.\n"
            + details.to_string(index=False)
        )

    return summary


# -------------------------------------------------------------------
# Formatting
# -------------------------------------------------------------------

def format_number(value, decimals):
    if pd.isna(value):
        return "—"

    return f"{float(value):,.{decimals}f}"


def get_row(summary, dataset, metric):
    selected = summary[
        summary["display_dataset"].eq(dataset)
        & summary["metric"].eq(metric)
    ]

    if selected.empty:
        return None

    return selected.iloc[0]


def format_cell(row, kind, decimals):
    if row is None:
        return "—"

    if kind == "count":
        return format_number(row.get("value"), 0)

    if kind == "percentage":
        value = row.get("value", np.nan)
        percent = row.get("percent", np.nan)

        if pd.isna(value):
            return "—"

        if pd.isna(percent):
            return format_number(value, 0)

        return (
            f"{format_number(value, 0)} "
            f"({format_number(percent, 1)}%)"
        )

    n = row.get("n", np.nan)

    if pd.isna(n) or n == 0:
        return "—"

    return (
        f"{format_number(row.get('median'), decimals)} "
        f"[{format_number(row.get('q1'), decimals)}, "
        f"{format_number(row.get('q3'), decimals)}]"
    )


def build_table(summary, datasets, row_settings):
    rows = []

    for metric, label, kind, decimals in row_settings:
        output_row = {"Characteristic": label}

        for dataset in datasets:
            source = get_row(summary, dataset, metric)

            output_row[dataset] = format_cell(
                source,
                kind,
                decimals,
            )

        rows.append(output_row)

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Notes and available sample sizes
# -------------------------------------------------------------------

def make_notes(summary, datasets, row_settings):
    notes = [
        (
            "Values are n (%) or median [Q1, Q3]. "
            "Each case represents one patient."
        ),
        (
            "Tumour presence refers to the reference annotations. "
            "Tumour-positive and tumour-negative percentages use "
            "patients with readable segmentations as the denominator; "
            "phase percentages use all cases."
        ),
        (
            "Lesions are connected components in the reference masks. "
            "Individual lesion-volume summaries give each lesion equal "
            "weight. Total tumour volume gives each tumour-positive "
            "patient equal weight."
        ),
        (
            "Zero-filled channels are treated as missing phases. "
            "Slice spacing is the z-axis voxel spacing and is not "
            "necessarily the acquisition slice thickness."
        ),
        (
            "Tumour HU values summarise the per-patient mean over "
            "annotated tumour voxels. Missing or undefined measurements "
            "are shown as —."
        ),
        (
            "These tables describe the raw datasets supplied to the "
            "extraction script, before any subsequent cohort exclusions."
        ),
    ]

    availability_rows = []

    for metric, label, kind, _ in row_settings:
        if kind != "continuous":
            continue

        entries = []

        for dataset in datasets:
            source = get_row(summary, dataset, metric)

            n = (
                source.get("n", np.nan)
                if source is not None else np.nan
            )

            if pd.isna(n):
                entries.append(f"{dataset}: unavailable")
            else:
                entries.append(f"{dataset}: {int(n)}")

            availability_rows.append({
                "dataset": dataset,
                "characteristic": label,
                "metric": metric,
                "n": n,
            })

        notes.append(
            f"Available n for {label}: "
            + "; ".join(entries)
            + "."
        )

    # Explicitly report incomplete segmentation/phase coverage.
    for dataset in datasets:
        for metric, description in [
            (
                "patients_with_unknown_tumour_status",
                "patients with unknown tumour status",
            ),
            (
                "patients_no_valid_phase",
                "patients without a valid CT phase",
            ),
        ]:
            source = get_row(summary, dataset, metric)

            if source is not None:
                value = source.get("value", np.nan)

                if pd.notna(value) and value > 0:
                    notes.append(
                        f"{dataset}: {int(value)} {description}."
                    )

    return notes, pd.DataFrame(availability_rows)


# -------------------------------------------------------------------
# Export CSV, Markdown
# -------------------------------------------------------------------
def save_table(table, notes, output, stem, title):
    csv_path = output / f"{stem}.csv"
    md_path = output / f"{stem}.md"

    table.to_csv(csv_path, index=False)

    def escape_markdown(value):
        return (
            str(value)
            .replace("|", r"\|")
            .replace("\n", " ")
        )

    lines = [
        f"**{title}**",
        "",
        "| " + " | ".join(
            escape_markdown(column)
            for column in table.columns
        ) + " |",
        "| " + " | ".join(
            ["---"] * len(table.columns)
        ) + " |",
    ]

    for row in table.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(
                escape_markdown(value)
                for value in row
            ) + " |"
        )

    lines.append("")

    for note in notes:
        lines.extend([note, ""])

    md_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create concise thesis characteristics tables "
            "for original and external source datasets."
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

    args = parser.parse_args()

    output = (
        args.output_dir
        if args.output_dir
        else args.input_dir / "tables"
    )

    summary = load_summaries(args.input_dir)

    if summary.empty:
        raise ValueError(
            "No recognised datasets found. "
            "Check DATASET_ALIASES."
        )

    output.mkdir(parents=True, exist_ok=True)

    available = set(summary["display_dataset"])

    for group, expected_datasets in GROUPS.items():
        datasets = [
            dataset
            for dataset in expected_datasets
            if dataset in available
        ]

        missing = [
            dataset
            for dataset in expected_datasets
            if dataset not in available
        ]

        if missing:
            print(
                f"Missing datasets for {group}: "
                + ", ".join(missing)
            )

        if not datasets:
            continue

        title = (
            "Characteristics of the original source datasets"
            if group == "original"
            else "Characteristics of the external source datasets"
        )

        for section, row_settings in [
            ("main", MAIN_ROWS),
            ("appendix", APPENDIX_ROWS),
        ]:
            table = build_table(
                summary,
                datasets,
                row_settings,
            )

            notes, sample_sizes = make_notes(
                summary,
                datasets,
                row_settings,
            )

            section_title = (
                title
                if section == "main"
                else title + ": supplementary characteristics"
            )

            stem = f"characteristics_{group}_{section}"

            save_table(
                table,
                notes,
                output,
                stem,
                section_title,
            )

            sample_sizes.to_csv(
                output / f"{stem}_sample_sizes.csv",
                index=False,
            )

            if section == "main":
                print(
                    "\n"
                    + section_title
                    + "\n"
                    + table.to_string(index=False)
                    + "\n"
                )

    print(f"Finished. Tables are in: {output}")


if __name__ == "__main__":
    main()