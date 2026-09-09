from pathlib import Path
import argparse
import random
import shutil


def patient_id_from_image(image_path, modality):
    """
    Remove .nii.gz and the nnU-Net modality suffix
    """
    suffix = f"_{modality}.nii.gz"

    if not image_path.name.endswith(suffix):
        raise ValueError(
            f"Expected {image_path.name} to end with {suffix}"
        )

    return image_path.name[:-len(suffix)]


def index_images(image_dir, modality):
    """
    Create a mapping: patient_id -> image path
    """
    image_dir = Path(image_dir)
    images = {}

    for image_path in sorted(image_dir.glob(f"*_{modality}.nii.gz")):
        patient_id = patient_id_from_image(image_path, modality)

        if patient_id in images:
            raise ValueError(
                f"Duplicate {modality} image for {patient_id} in {image_dir}"
            )

        images[patient_id] = image_path

    return images


def index_segmentations(seg_dir):
    """
    Create a mapping: patient_id -> segmentation path
    """
    seg_dir = Path(seg_dir)
    segmentations = {}

    for seg_path in sorted(seg_dir.glob("*.nii.gz")):
        patient_id = seg_path.name[:-7]

        if patient_id in segmentations:
            raise ValueError(
                f"Duplicate segmentation for {patient_id}"
            )

        segmentations[patient_id] = seg_path

    return segmentations


def create_output_folders(output_dir):
    output_dir = Path(output_dir)

    folders = {
        "imagesTr": output_dir / "imagesTr",
        "labelsTr": output_dir / "labelsTr",
        "imagesTs": output_dir / "imagesTs",
        "labelsTs": output_dir / "labelsTs",
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

        # Prevent old cases from remaining in the output dataset
        existing_files = list(folder.glob("*.nii.gz"))

        if existing_files:
            raise RuntimeError(
                f"Output folder is not empty: {folder}\n"
                "Use a new/empty output directory."
            )

        print(f"Checked output folder: {folder.resolve()}")

    return folders


def copy_single_channel_cases(patient_ids, ct_1_images, segmentations, image_output_dir, label_output_dir):
    for patient_id in patient_ids:
        shutil.copy2(
            ct_1_images[patient_id],
            image_output_dir / f"{patient_id}_0000.nii.gz",
        )

        shutil.copy2(
            segmentations[patient_id],
            label_output_dir / f"{patient_id}.nii.gz",
        )

        print(f"Single channel: copied {patient_id}")


def copy_multi_channel_cases(patient_ids, ct_2_0000_images, ct_2_0001_images, segmentations, image_output_dir, label_output_dir):
    for patient_id in patient_ids:
        shutil.copy2(
            ct_2_0000_images[patient_id],
            image_output_dir / f"{patient_id}_0000.nii.gz",
        )

        shutil.copy2(
            ct_2_0001_images[patient_id],
            image_output_dir / f"{patient_id}_0001.nii.gz",
        )

        shutil.copy2(
            segmentations[patient_id],
            label_output_dir / f"{patient_id}.nii.gz",
        )

        print(f"Multi-channel: copied {patient_id}")


def create_combined_split(ct_1_dir, ct_2_dir, seg_dir, output_1_dir, output_2_dir, train_ratio=0.8, seed=42):
    ct_1_dir = Path(ct_1_dir)
    ct_2_dir = Path(ct_2_dir)
    seg_dir = Path(seg_dir)
    output_1_dir = Path(output_1_dir)
    output_2_dir = Path(output_2_dir)

    if output_1_dir.resolve() == output_2_dir.resolve():
        raise ValueError(
            "--output_1_dir and --output_2_dir must be different"
        )

    if not 0 < train_ratio < 1:
        raise ValueError("--train_ratio must be between 0 and 1")

    # Single-channel input
    ct_1_0000 = index_images(ct_1_dir, "0000")

    # Multi-channel input
    ct_2_0000 = index_images(ct_2_dir, "0000")
    ct_2_0001 = index_images(ct_2_dir, "0001")

    # Shared segmentations
    segmentations = index_segmentations(seg_dir)

    print("\nInput counts")
    print(f"Single-channel _0000: {len(ct_1_0000)}")
    print(f"Multi-channel _0000:  {len(ct_2_0000)}")
    print(f"Multi-channel _0001:  {len(ct_2_0001)}")
    print(f"Segmentations:         {len(segmentations)}")

    # Only retain patients available in every required input
    common_patients = sorted(
        set(ct_1_0000)
        & set(ct_2_0000)
        & set(ct_2_0001)
        & set(segmentations)
    )

    if not common_patients:
        raise RuntimeError(
            "No patients were found in all input folders"
        )

    print(f"\nPatients available for both datasets: {len(common_patients)}")

    # Report excluded patients
    all_patients = (
        set(ct_1_0000)
        | set(ct_2_0000)
        | set(ct_2_0001)
        | set(segmentations)
    )

    excluded_patients = sorted(all_patients - set(common_patients))

    for patient_id in excluded_patients:
        missing = []

        if patient_id not in ct_1_0000:
            missing.append("single-channel _0000")

        if patient_id not in ct_2_0000:
            missing.append("multi-channel _0000")

        if patient_id not in ct_2_0001:
            missing.append("multi-channel _0001")

        if patient_id not in segmentations:
            missing.append("segmentation")

        print(
            f"EXCLUDED {patient_id}: missing {', '.join(missing)}"
        )

    # Create one shared patient split
    rng = random.Random(seed)
    rng.shuffle(common_patients)

    split_index = int(len(common_patients) * train_ratio)

    train_patients = common_patients[:split_index]
    test_patients = common_patients[split_index:]

    print("\nShared split")
    print(f"Training patients: {len(train_patients)}")
    print(f"Testing patients:  {len(test_patients)}")

    single_output = create_output_folders(output_1_dir)
    multi_output = create_output_folders(output_2_dir)

    print("\nCopying single-channel training cases")
    copy_single_channel_cases(
        patient_ids=train_patients,
        ct_1_images=ct_1_0000,
        segmentations=segmentations,
        image_output_dir=single_output["imagesTr"],
        label_output_dir=single_output["labelsTr"],
    )

    print("\nCopying single-channel test cases")
    copy_single_channel_cases(
        patient_ids=test_patients,
        ct_1_images=ct_1_0000,
        segmentations=segmentations,
        image_output_dir=single_output["imagesTs"],
        label_output_dir=single_output["labelsTs"],
    )

    print("\nCopying multi-channel training cases")
    copy_multi_channel_cases(
        patient_ids=train_patients,
        ct_2_0000_images=ct_2_0000,
        ct_2_0001_images=ct_2_0001,
        segmentations=segmentations,
        image_output_dir=multi_output["imagesTr"],
        label_output_dir=multi_output["labelsTr"],
    )

    print("\nCopying multi-channel test cases")
    copy_multi_channel_cases(
        patient_ids=test_patients,
        ct_2_0000_images=ct_2_0000,
        ct_2_0001_images=ct_2_0001,
        segmentations=segmentations,
        image_output_dir=multi_output["imagesTs"],
        label_output_dir=multi_output["labelsTs"],
    )

    print("\nDone")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create matched single-channel and multi-channel "
            "nnU-Net datasets"
        )
    )

    parser.add_argument(
        "--ct_1_dir",
        required=True,
        help="Folder containing single-channel _0000 images",
    )

    parser.add_argument(
        "--ct_2_dir",
        required=True,
        help="Folder containing multi-channel _0000 and _0001 images",
    )

    parser.add_argument(
        "--seg_dir",
        required=True,
        help="Folder containing shared segmentation masks",
    )

    parser.add_argument(
        "--output_1_dir",
        required=True,
        help="Output folder for the single-channel dataset",
    )

    parser.add_argument(
        "--output_2_dir",
        required=True,
        help="Output folder for the multi-channel dataset",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training fraction (default: 0.8)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    create_combined_split(
        ct_1_dir=args.ct_1_dir,
        ct_2_dir=args.ct_2_dir,
        seg_dir=args.seg_dir,
        output_1_dir=args.output_1_dir,
        output_2_dir=args.output_2_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )