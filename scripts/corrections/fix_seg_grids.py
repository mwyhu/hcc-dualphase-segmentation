from pathlib import Path
import argparse
import shutil
import SimpleITK as sitk


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Copy CT1 and SEG folders and resample mismatching "
            "segmentations onto their corresponding CT1 grids."
        )
    )

    parser.add_argument(
        "--ct_dir",
        type=Path,
        required=True,
        help="Directory containing the reference CT1 images.",
    )

    parser.add_argument(
        "--seg_dir",
        type=Path,
        required=True,
        help="Directory containing the original segmentations.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="New output dataset directory.",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance used when comparing image geometry.",
    )

    return parser.parse_args()


def same_values(values_a, values_b, tolerance):
    return all(
        abs(float(a) - float(b)) <= tolerance
        for a, b in zip(values_a, values_b)
    )


def compare_grids(reference, segmentation, tolerance):
    return {
        "shape": reference.GetSize() == segmentation.GetSize(),
        "spacing": same_values(
            reference.GetSpacing(),
            segmentation.GetSpacing(),
            tolerance,
        ),
        "origin": same_values(
            reference.GetOrigin(),
            segmentation.GetOrigin(),
            tolerance,
        ),
        "direction": same_values(
            reference.GetDirection(),
            segmentation.GetDirection(),
            tolerance,
        ),
    }


def remove_nifti_extension(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]

    if filename.endswith(".nii"):
        return filename[:-4]

    return filename


def get_case_id(image_path):
    case_id = remove_nifti_extension(image_path.name)

    if case_id.endswith("_0000"):
        case_id = case_id[:-5]

    return case_id


def find_segmentation(seg_dir, case_id):
    names = [
        f"{case_id}.nii.gz",
        f"{case_id}_0000.nii.gz",
        f"{case_id}.nii",
        f"{case_id}_0000.nii",
    ]

    if case_id.startswith("WAW_TACE_"):
        short_id = case_id.removeprefix("WAW_TACE_")

        names.extend(
            [
                f"{short_id}.nii.gz",
                f"{short_id}_0000.nii.gz",
                f"{short_id}.nii",
                f"{short_id}_0000.nii",
            ]
        )

    for name in names:
        candidate = seg_dir / name

        if candidate.is_file():
            return candidate

    return None


def resample_segmentation(segmentation, reference):
    return sitk.Resample(
        segmentation,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        segmentation.GetPixelID())


def collect_nifti_files(directory):
    files = list(directory.glob("*.nii.gz"))
    files.extend(directory.glob("*.nii"))

    # Avoid counting .nii.gz files again through the *.nii pattern
    return sorted(set(files))


def copy_input_directories(source_ct_dir, source_seg_dir, output_ct_dir, output_seg_dir):
    shutil.copytree(source_ct_dir, output_ct_dir)
    shutil.copytree(source_seg_dir, output_seg_dir)

    print(f"Copied CT images to:      {output_ct_dir}")
    print(f"Copied segmentations to:  {output_seg_dir}")


def main():
    args = parse_arguments()

    if not args.ct_dir.is_dir():
        raise FileNotFoundError(
            f"CT directory does not exist: {args.ct_dir}"
        )

    if not args.seg_dir.is_dir():
        raise FileNotFoundError(
            f"Segmentation directory does not exist: {args.seg_dir}"
        )

    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}\n"
            "Use a new output path to avoid overwriting existing data."
        )

    output_ct_dir = args.output_dir / "CT1"
    output_seg_dir = args.output_dir / "SEG"

    args.output_dir.mkdir(parents=True)

    copy_input_directories(
        source_ct_dir=args.ct_dir,
        source_seg_dir=args.seg_dir,
        output_ct_dir=output_ct_dir,
        output_seg_dir=output_seg_dir,
    )

    ct_files = collect_nifti_files(output_ct_dir)

    if not ct_files:
        raise RuntimeError(
            f"No NIfTI images found in: {output_ct_dir}"
        )

    fixed = 0
    already_correct = 0
    missing = []
    failed = []

    for ct_path in ct_files:
        case_id = get_case_id(ct_path)
        seg_path = find_segmentation(output_seg_dir, case_id)

        if seg_path is None:
            print(f"[MISSING] {case_id}: segmentation not found")
            missing.append(case_id)
            continue

        try:
            reference = sitk.ReadImage(str(ct_path))
            segmentation = sitk.ReadImage(str(seg_path))

            comparison_before = compare_grids(
                reference,
                segmentation,
                args.tolerance,
            )

            if all(comparison_before.values()):
                print(f"[OK] {case_id}")
                already_correct += 1
                continue

            mismatches = [
                name
                for name, matches in comparison_before.items()
                if not matches
            ]

            print(
                f"[FIX] {case_id}: "
                f"{', '.join(mismatches)} mismatch"
            )

            corrected = resample_segmentation(
                segmentation=segmentation,
                reference=reference,
            )

            # Overwrite only copy
            sitk.WriteImage(corrected, str(seg_path))

            written = sitk.ReadImage(str(seg_path))
            comparison_after = compare_grids(
                reference,
                written,
                args.tolerance,
            )

            if not all(comparison_after.values()):
                raise RuntimeError(
                    f"Output verification failed: {comparison_after}"
                )

            print(f"      Saved: {seg_path}")
            fixed += 1

        except Exception as error:
            print(f"[FAILED] {case_id}: {error}")
            failed.append((case_id, str(error)))

    print("\nSummary")
    print(f"CT images checked:       {len(ct_files)}")
    print(f"Already matching:        {already_correct}")
    print(f"Segmentations resampled: {fixed}")
    print(f"Missing segmentations:   {len(missing)}")
    print(f"Failed cases:            {len(failed)}")

    if missing:
        print("\nMissing segmentations:")
        for case_id in missing:
            print(f"  {case_id}")

    if failed:
        print("\nFailed cases:")
        for case_id, error in failed:
            print(f"  {case_id}: {error}")


if __name__ == "__main__":
    main()