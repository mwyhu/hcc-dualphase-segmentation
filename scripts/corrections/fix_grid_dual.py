from pathlib import Path
import argparse
import json
import shutil

import numpy as np
import SimpleITK as sitk


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fix dual-phase CT and segmentation grids according to phase_selection.json.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Folder containing _0000 and _0001 CT images.")
    parser.add_argument("--seg_dir", type=Path, required=True, help="Folder containing the segmentations.")
    parser.add_argument("--ct1_dir", type=Path, required=True, help="Folder containing external CT1 reference images.")
    parser.add_argument("--phase_json", type=Path, required=True, help="Path to phase_selection.json.")
    parser.add_argument("--output_dir", type=Path, required=True, help="New output directory.")
    return parser.parse_args()


def remove_nifti_extension(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]

    if filename.endswith(".nii"):
        return filename[:-4]

    return filename


def get_case_id(path):
    case_id = remove_nifti_extension(path.name)

    if case_id.endswith("_0000") or case_id.endswith("_0001"):
        case_id = case_id[:-5]

    return case_id


def get_json_patient_id(case_id):
    numeric_patient_id = int(case_id.split("_")[-1])
    return numeric_patient_id - 1000


def grids_match(reference, moving):
    return {
        "shape": reference.GetSize() == moving.GetSize(),
        "spacing": np.allclose(reference.GetSpacing(), moving.GetSpacing()),
        "origin": np.allclose(reference.GetOrigin(), moving.GetOrigin()),
        "direction": np.allclose(reference.GetDirection(), moving.GetDirection()),
    }


def build_phase_lookup(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    phase_lookup = {}

    if isinstance(data, dict):
        for patient_id, value in data.items():
            if isinstance(value, dict):
                if "label_phase" not in value:
                    continue

                phase_lookup[str(int(patient_id))] = int(value["label_phase"])

            elif isinstance(value, (int, float, str)):
                phase_lookup[str(int(patient_id))] = int(value)

    elif isinstance(data, list):
        for record in data:
            if not isinstance(record, dict):
                continue

            patient_id = record.get("patient_id")

            if patient_id is None:
                patient_id = record.get("patientid")

            if patient_id is None:
                patient_id = record.get("patient")

            if patient_id is None:
                patient_id = record.get("id")

            if patient_id is None or "label_phase" not in record:
                continue

            phase_lookup[str(int(patient_id))] = int(record["label_phase"])

    else:
        raise ValueError("phase_selection.json must contain a dictionary or list.")

    if not phase_lookup:
        raise ValueError("No patient IDs and label_phase values found in the JSON file.")

    return phase_lookup


def collect_channel_zero_files(directory):
    files = list(directory.glob("*_0000.nii.gz"))
    files.extend(directory.glob("*_0000.nii"))
    return sorted(set(files))


def find_channel_file(directory, case_id, channel):
    candidates = [
        directory / f"{case_id}_{channel:04d}.nii.gz",
        directory / f"{case_id}_{channel:04d}.nii",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Channel _{channel:04d} not found for {case_id} in {directory}")


def find_segmentation(directory, case_id, json_patient_id):
    numeric_case_id = case_id.split("_")[-1]

    candidates = [
        directory / f"{case_id}.nii.gz",
        directory / f"{case_id}.nii",
        directory / f"{case_id}_0000.nii.gz",
        directory / f"{case_id}_0000.nii",
        directory / f"{numeric_case_id}.nii.gz",
        directory / f"{numeric_case_id}.nii",
        directory / f"{numeric_case_id}_0000.nii.gz",
        directory / f"{numeric_case_id}_0000.nii",
        directory / f"{json_patient_id}.nii.gz",
        directory / f"{json_patient_id}.nii",
        directory / f"{json_patient_id}_0000.nii.gz",
        directory / f"{json_patient_id}_0000.nii",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No segmentation found for {case_id}. Tried:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def find_ct1_reference(directory, case_id, json_patient_id):
    numeric_case_id = case_id.split("_")[-1]

    candidates = [
        directory / f"{case_id}_0000.nii.gz",
        directory / f"{case_id}_0000.nii",
        directory / f"{case_id}.nii.gz",
        directory / f"{case_id}.nii",
        directory / f"{numeric_case_id}_0000.nii.gz",
        directory / f"{numeric_case_id}_0000.nii",
        directory / f"{numeric_case_id}.nii.gz",
        directory / f"{numeric_case_id}.nii",
        directory / f"{json_patient_id}_0000.nii.gz",
        directory / f"{json_patient_id}_0000.nii",
        directory / f"{json_patient_id}.nii.gz",
        directory / f"{json_patient_id}.nii",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No CT1 reference found for {case_id}. Tried:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def resample_if_needed(moving_path, reference_path, interpolator, default_value, image_type):
    moving = sitk.ReadImage(str(moving_path))
    reference = sitk.ReadImage(str(reference_path))

    comparison = grids_match(reference, moving)
    mismatches = [name for name, matches in comparison.items() if not matches]

    if not mismatches:
        print(f"  [OK] {image_type}: {moving_path.name}")
        return False

    print(f"  [RESAMPLE] {image_type}: {moving_path.name}")
    print(f"      Reference:  {reference_path.name}")
    print(f"      Mismatches: {', '.join(mismatches)}")
    print(f"      Old size:   {moving.GetSize()}")
    print(f"      New size:   {reference.GetSize()}")
    print(f"      Old spacing:{moving.GetSpacing()}")
    print(f"      New spacing:{reference.GetSpacing()}")
    print(f"      Old origin: {moving.GetOrigin()}")
    print(f"      New origin: {reference.GetOrigin()}")

    resampled = sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        interpolator,
        default_value,
        moving.GetPixelID(),
    )

    sitk.WriteImage(resampled, str(moving_path))

    written = sitk.ReadImage(str(moving_path))
    verification = grids_match(reference, written)

    if not all(verification.values()):
        raise RuntimeError(f"Grid verification failed for {moving_path}: {verification}")

    print(f"      Saved: {moving_path}")
    return True


def main():
    args = parse_arguments()

    for directory in (args.image_dir, args.seg_dir, args.ct1_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

    if not args.phase_json.is_file():
        raise FileNotFoundError(f"JSON file not found: {args.phase_json}")

    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}\n"
            "Use a new output directory so existing data is not overwritten."
        )

    output_image_dir = args.output_dir / "CT"
    output_seg_dir = args.output_dir / "SEG"

    args.output_dir.mkdir(parents=True)
    shutil.copytree(args.image_dir, output_image_dir)
    shutil.copytree(args.seg_dir, output_seg_dir)

    print(f"Copied dual-phase images to: {output_image_dir}")
    print(f"Copied segmentations to:     {output_seg_dir}")

    phase_lookup = build_phase_lookup(args.phase_json)
    channel_zero_files = collect_channel_zero_files(output_image_dir)

    if not channel_zero_files:
        raise RuntimeError(f"No _0000 NIfTI files found in {output_image_dir}")

    successful_cases = 0
    fixed_ct_images = 0
    fixed_segmentations = 0
    failed_cases = []

    for channel_zero_path in channel_zero_files:
        case_id = get_case_id(channel_zero_path)

        print(f"\nCase: {case_id}")

        try:
            channel_one_path = find_channel_file(output_image_dir, case_id, 1)
            json_patient_id = get_json_patient_id(case_id)
            json_key = str(json_patient_id)

            if json_key not in phase_lookup:
                raise KeyError(
                    f"Patient {json_patient_id} was not found in phase_selection.json"
                )

            label_phase = phase_lookup[json_key]
            segmentation_path = find_segmentation(output_seg_dir, case_id, json_patient_id)

            print(f"  JSON patient ID: {json_patient_id}")
            print(f"  label_phase:     {label_phase}")

            if label_phase in (0, 3):
                final_reference = find_ct1_reference(args.ct1_dir, case_id, json_patient_id)

                print(f"  Final reference: external CT1 ({final_reference.name})")

                fixed_ct_images += int(
                    resample_if_needed(
                        channel_zero_path,
                        final_reference,
                        sitk.sitkLinear,
                        -1024.0,
                        "CT _0000",
                    )
                )

                fixed_ct_images += int(
                    resample_if_needed(
                        channel_one_path,
                        final_reference,
                        sitk.sitkLinear,
                        -1024.0,
                        "CT _0001",
                    )
                )

            elif label_phase == 1:
                final_reference = channel_zero_path

                print(f"  Final reference: {final_reference.name}")

                fixed_ct_images += int(
                    resample_if_needed(
                        channel_one_path,
                        final_reference,
                        sitk.sitkLinear,
                        -1024.0,
                        "CT _0001",
                    )
                )

            elif label_phase == 2:
                final_reference = channel_one_path

                print(f"  Final reference: {final_reference.name}")

                fixed_ct_images += int(
                    resample_if_needed(
                        channel_zero_path,
                        final_reference,
                        sitk.sitkLinear,
                        -1024.0,
                        "CT _0000",
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported label_phase {label_phase} for {case_id}"
                )

            fixed_segmentations += int(
                resample_if_needed(
                    segmentation_path,
                    final_reference,
                    sitk.sitkNearestNeighbor,
                    0,
                    "SEG",
                )
            )

            successful_cases += 1

        except Exception as error:
            print(f"  [FAILED] {error}")
            failed_cases.append((case_id, str(error)))

    print("\nSummary")
    print(f"Cases found:                  {len(channel_zero_files)}")
    print(f"Cases processed successfully: {successful_cases}")
    print(f"CT images resampled:          {fixed_ct_images}")
    print(f"Segmentations resampled:      {fixed_segmentations}")
    print(f"Failed cases:                 {len(failed_cases)}")

    if failed_cases:
        print("\nFailed cases:")

        for case_id, error in failed_cases:
            print(f"  {case_id}: {error}")


if __name__ == "__main__":
    main()