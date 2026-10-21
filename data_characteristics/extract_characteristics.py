from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


def discover_files(folder, pattern):
    """
    Match filenames using a template containing {case_id}

    e.g.,
        {case_id}.nii.gz
        {case_id}_0000.nii.gz
        {case_id}_ART.nii.gz
        {case_id}/segmentation.nii.gz
    """
    if pattern.count("{case_id}") != 1:
        raise ValueError(
            "Each filename pattern must contain {case_id} exactly once."
        )

    expression = re.escape(pattern).replace(
        re.escape("{case_id}"),
        r"(?P<case_id>.+)",
    )

    regex = re.compile(expression)
    glob_pattern = pattern.replace("{case_id}", "*")

    files = {}

    for path in sorted(folder.glob(glob_pattern)):
        if not path.is_file():
            continue

        relative_path = path.relative_to(folder).as_posix()
        match = regex.fullmatch(relative_path)

        if match is None:
            continue

        case_id = match.group("case_id")

        if case_id in files:
            raise ValueError(
                f"Multiple files matched case {case_id}: "
                f"{files[case_id]} and {path}"
            )

        files[case_id] = path

    return files


def read_image(path):
    image = sitk.ReadImage(str(path))

    if (
        image.GetDimension() != 3
        or image.GetNumberOfComponentsPerPixel() != 1
    ):
        raise ValueError("Expected a scalar 3D image.")

    array = sitk.GetArrayFromImage(image)

    if not np.all(np.isfinite(array)):
        raise ValueError("Image contains non-finite values.")

    return image, array


def same_geometry(a, b):
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(
            a.GetSpacing(), b.GetSpacing(),
            atol=1e-5, rtol=0,
        )
        and np.allclose(
            a.GetOrigin(), b.GetOrigin(),
            atol=1e-4, rtol=0,
        )
        and np.allclose(
            a.GetDirection(), b.GetDirection(),
            atol=1e-5, rtol=0,
        )
    )


def describe(values):
    values = pd.to_numeric(
        pd.Series(list(values), dtype=object),
        errors="coerce",
    ).to_numpy(dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "n": 0,
            **{
                name: np.nan
                for name in [
                    "mean", "sd", "median",
                    "q1", "q3", "min", "max",
                ]
            },
        }

    return {
        "n": len(values),
        "mean": values.mean(),
        "sd": (
            values.std(ddof=1)
            if len(values) > 1 else np.nan
        ),
        "median": np.median(values),
        "q1": np.quantile(values, 0.25),
        "q3": np.quantile(values, 0.75),
        "min": values.min(),
        "max": values.max(),
    }


# -------------------------------------------------------------------
# Raw data analysis
# -------------------------------------------------------------------

def analyse(args):
    for folder in [args.ct_dir, args.seg_dir]:
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Folder does not exist: {folder}"
            )

    art_files = discover_files(
        args.ct_dir, args.art_pattern
    )
    pv_files = discover_files(
        args.ct_dir, args.pv_pattern
    )
    seg_files = discover_files(
        args.seg_dir, args.seg_pattern
    )

    # Prevent accidentally assigning the same file to both phases
    overlapping_files = (
        set(art_files.values()) & set(pv_files.values())
    )

    if overlapping_files:
        raise ValueError(
            "ART and PV patterns match the same files. "
            "Use distinct phase patterns."
        )

    case_ids = sorted(
        set(art_files) | set(pv_files) | set(seg_files)
    )

    if not case_ids:
        raise ValueError(
            "No files matched. Check folders and filename patterns."
        )

    cases = []
    volumes = []
    lesions = []
    issues = []

    structure = ndimage.generate_binary_structure(
        3,
        3 if args.connectivity == 26 else 1,
    )

    for case_id in case_ids:
        key = {
            "dataset": args.dataset,
            "case_id": case_id,
        }

        def issue(path, message):
            issues.append({
                **key,
                "file": str(path) if path else "",
                "issue": message,
            })

        case = {
            **key,
            "seg_available": False,
            "tumour_present": np.nan,
            "n_lesions": np.nan,
            "total_tumour_volume_ml": np.nan,
            "mean_lesion_volume_ml": np.nan,
            "median_lesion_volume_ml": np.nan,
            "has_ART": False,
            "has_PV": False,
            "ART_zero_filled": False,
            "PV_zero_filled": False,
        }

        reference = None
        tumour = None
        components = None
        case_lesions = []


        # Segmentation measurements
        seg_path = seg_files.get(case_id)

        if seg_path is None:
            issue(
                None,
                "Missing segmentation; tumour status unknown.",
            )
        else:
            try:
                reference, segmentation = read_image(seg_path)

                tumour = np.isin(
                    segmentation,
                    args.tumour_labels,
                )

                components, n_lesions = ndimage.label(
                    tumour,
                    structure=structure,
                )

                counts = np.bincount(
                    components.ravel(),
                    minlength=n_lesions + 1,
                )[1:]

                voxel_volume_ml = (
                    np.prod(reference.GetSpacing()) / 1000.0
                )

                lesion_volumes = counts * voxel_volume_ml

                case.update({
                    "seg_available": True,
                    "tumour_present": bool(n_lesions),
                    "n_lesions": n_lesions,
                    "total_tumour_volume_ml": float(
                        lesion_volumes.sum()
                    ),
                    "mean_lesion_volume_ml": (
                        float(lesion_volumes.mean())
                        if n_lesions else np.nan
                    ),
                    "median_lesion_volume_ml": (
                        float(np.median(lesion_volumes))
                        if n_lesions else np.nan
                    ),
                })

                case_lesions = [
                    {
                        **key,
                        "lesion_id": index + 1,
                        "volume_ml": float(volume),
                        "ART_mean_hu": np.nan,
                        "PV_mean_hu": np.nan,
                    }
                    for index, volume in enumerate(lesion_volumes)
                ]

            except Exception as error:
                reference = None
                tumour = None
                components = None

                issue(
                    seg_path,
                    f"Segmentation analysis failed: {error}",
                )


        # Phase availability, image geometry, and tumour HU
        for phase, phase_files in [
            ("ART", art_files),
            ("PV", pv_files),
        ]:
            path = phase_files.get(case_id)

            if path is None:
                issue(None, f"No {phase} file.")
                continue

            volume = {
                **key,
                "phase": phase,
                "file": str(path),
                "read_ok": False,
                "zero_filled": False,
                "geometry_matches_seg": False,
                "n_slices": np.nan,
                "spacing_x_mm": np.nan,
                "spacing_y_mm": np.nan,
                "spacing_z_mm": np.nan,
                "tumour_mean_hu": np.nan,
                "tumour_sd_hu": np.nan,
            }

            try:
                image, ct = read_image(path)

                zero_filled = not np.any(ct != 0)

                volume["read_ok"] = True
                volume["zero_filled"] = zero_filled
                case[f"{phase}_zero_filled"] = zero_filled

                if zero_filled:
                    issue(
                        path,
                        f"Zero-filled {phase}; treated as missing.",
                    )
                else:
                    case[f"has_{phase}"] = True

                    spacing = image.GetSpacing()

                    volume.update({
                        "n_slices": image.GetSize()[2],
                        "spacing_x_mm": spacing[0],
                        "spacing_y_mm": spacing[1],
                        "spacing_z_mm": spacing[2],
                    })

                    for column in [
                        "n_slices",
                        "spacing_x_mm",
                        "spacing_y_mm",
                        "spacing_z_mm",
                    ]:
                        case[f"{phase}_{column}"] = volume[column]

                    if reference is not None:
                        matches = same_geometry(image, reference)

                        volume["geometry_matches_seg"] = matches

                        if not matches:
                            issue(
                                path,
                                "CT/SEG geometry mismatch; HU omitted.",
                            )

                        elif tumour.any():
                            tumour_hu = ct[tumour]

                            volume["tumour_mean_hu"] = float(
                                tumour_hu.mean(dtype=np.float64)
                            )
                            volume["tumour_sd_hu"] = float(
                                tumour_hu.std(dtype=np.float64)
                            )

                            case[f"{phase}_tumour_mean_hu"] = (
                                volume["tumour_mean_hu"]
                            )

                            lesion_means = ndimage.mean(
                                ct,
                                labels=components,
                                index=np.arange(
                                    1, len(case_lesions) + 1
                                ),
                            )

                            for lesion, mean_hu in zip(
                                case_lesions, lesion_means
                            ):
                                lesion[f"{phase}_mean_hu"] = float(
                                    mean_hu
                                )

            except Exception as error:
                issue(path, f"CT analysis failed: {error}")

            volumes.append(volume)

        if case["has_ART"] and case["has_PV"]:
            case["phase_pattern"] = "ART+PV"
        elif case["has_ART"]:
            case["phase_pattern"] = "ART_only"
        elif case["has_PV"]:
            case["phase_pattern"] = "PV_only"
        else:
            case["phase_pattern"] = "no_valid_phase"

        cases.append(case)
        lesions.extend(case_lesions)

        print(
            f"{case_id}: {case['phase_pattern']}, "
            f"lesions={case['n_lesions']}",
            flush=True,
        )

    case_df = pd.DataFrame(cases)

    volume_df = pd.DataFrame(
        volumes,
        columns=[
            "dataset", "case_id", "phase", "file",
            "read_ok", "zero_filled", "geometry_matches_seg",
            "n_slices", "spacing_x_mm", "spacing_y_mm",
            "spacing_z_mm", "tumour_mean_hu", "tumour_sd_hu",
        ],
    )

    lesion_df = pd.DataFrame(
        lesions,
        columns=[
            "dataset", "case_id", "lesion_id",
            "volume_ml", "ART_mean_hu", "PV_mean_hu",
        ],
    )

    issue_df = pd.DataFrame(
        issues,
        columns=["dataset", "case_id", "file", "issue"],
    )

    return case_df, volume_df, lesion_df, issue_df


# -------------------------------------------------------------------
# Dataset summary
# -------------------------------------------------------------------

def summarise(dataset, cases, volumes, lesions):
    rows = []

    def add_count(metric, value, denominator=None):
        rows.append({
            "dataset": dataset,
            "metric": metric,
            "unit": "count",
            "value": int(value),
            "denominator": denominator,
            "percent": (
                100.0 * value / denominator
                if denominator else np.nan
            ),
        })

    def add_stats(metric, values, unit):
        rows.append({
            "dataset": dataset,
            "metric": metric,
            "unit": unit,
            **describe(values),
        })

    labelled = cases[cases["seg_available"]]
    positive = labelled[labelled["tumour_present"].eq(True)]
    negative = labelled[labelled["tumour_present"].eq(False)]

    # One case = one unique patient
    add_count("patients", len(cases))

    add_count(
        "patients_with_valid_CT",
        (cases["has_ART"] | cases["has_PV"]).sum(),
        len(cases),
    )

    add_count(
        "patients_with_readable_SEG",
        len(labelled),
        len(cases),
    )

    add_count(
        "patients_with_annotated_tumours",
        len(positive),
        len(labelled),
    )

    add_count(
        "patients_without_annotated_tumours",
        len(negative),
        len(labelled),
    )

    add_count(
        "patients_with_unknown_tumour_status",
        len(cases) - len(labelled),
        len(cases),
    )

    for pattern in [
        "ART+PV",
        "ART_only",
        "PV_only",
        "no_valid_phase",
    ]:
        add_count(
            f"patients_{pattern}",
            cases["phase_pattern"].eq(pattern).sum(),
            len(cases),
        )

    add_count(
        "total_annotated_lesions",
        labelled["n_lesions"].sum(),
    )

    add_stats(
        "lesions_per_patient_all_labelled",
        labelled["n_lesions"],
        "lesions",
    )

    add_stats(
        "lesions_per_tumour_positive_patient",
        positive["n_lesions"],
        "lesions",
    )

    add_stats(
        "individual_lesion_volume_lesion_weighted",
        lesions["volume_ml"],
        "mL",
    )

    add_stats(
        "mean_lesion_volume_per_tumour_positive_patient",
        positive["mean_lesion_volume_ml"],
        "mL",
    )

    add_stats(
        "total_tumour_volume_per_patient_all_labelled",
        labelled["total_tumour_volume_ml"],
        "mL",
    )

    add_stats(
        "total_tumour_volume_per_tumour_positive_patient",
        positive["total_tumour_volume_ml"],
        "mL",
    )

    for phase in ["ART", "PV"]:
        add_count(
            f"patients_with_{phase}",
            cases[f"has_{phase}"].sum(),
            len(cases),
        )

        add_count(
            f"{phase}_zero_filled_channels",
            cases[f"{phase}_zero_filled"].sum(),
            len(cases),
        )

        phase_volumes = volumes[volumes["phase"].eq(phase)]

        usable = phase_volumes[
            phase_volumes["read_ok"].eq(True)
            & phase_volumes["zero_filled"].eq(False)
        ]

        add_count(
            f"{phase}_valid_CT_volumes",
            len(usable),
        )

        for column, unit in [
            ("n_slices", "slices"),
            ("spacing_x_mm", "mm"),
            ("spacing_y_mm", "mm"),
            ("spacing_z_mm", "mm"),
            ("tumour_mean_hu", "HU"),
        ]:
            add_stats(
                f"{phase}_{column}",
                usable[column],
                unit,
            )

    return pd.DataFrame(rows)



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate raw CT and tumour segmentation characteristics "
            "for one complete dataset, without splits or metadata."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name used in the CSV files.",
    )

    parser.add_argument(
        "--ct-dir",
        required=True,
        type=Path,
        help="Folder containing the raw CT images.",
    )

    parser.add_argument(
        "--seg-dir",
        required=True,
        type=Path,
        help="Folder containing the reference segmentations.",
    )

    parser.add_argument(
        "--art-pattern",
        default="{case_id}_0000.nii.gz",
        help="ART filename pattern containing {case_id}.",
    )

    parser.add_argument(
        "--pv-pattern",
        default="{case_id}_0001.nii.gz",
        help="PV filename pattern containing {case_id}.",
    )

    parser.add_argument(
        "--seg-pattern",
        default="{case_id}.nii.gz",
        help="SEG filename pattern containing {case_id}.",
    )

    parser.add_argument(
        "--tumour-labels",
        type=int,
        nargs="+",
        default=[1],
        help="Tumour label values in the SEG files; default: 1.",
    )

    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[6, 26],
        default=26,
        help="Connectivity for lesion counting; default: 26.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("characteristics"),
        help="Output folder.",
    )

    args = parser.parse_args()

    cases, volumes, lesions, issues = analyse(args)

    summary = summarise(
        args.dataset,
        cases,
        volumes,
        lesions,
    )

    settings = pd.DataFrame([{
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
    }])

    args.output.mkdir(parents=True, exist_ok=True)

    outputs = {
        "cases.csv": cases,
        "volumes.csv": volumes,
        "lesions.csv": lesions,
        "summary.csv": summary,
        "issues.csv": issues,
        "analysis_settings.csv": settings,
    }

    for filename, dataframe in outputs.items():
        path = args.output / filename

        dataframe.to_csv(
            path,
            index=False,
            float_format="%.6f",
        )

        print(f"Saved: {path}")

    print(
        f"\nFinished: {len(cases)} patients, "
        f"{len(lesions)} annotated lesions."
    )
    print("Review issues.csv for missing phases and geometry mismatches.")


if __name__ == "__main__":
    main()