from pathlib import Path
import shutil
import random
import argparse


def get_patient_id(ct_file):
    """
    Extract patient id from ct filename by removing modality suffix
    """

    name = ct_file.name

    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]

    # Remove nnU-Net modality suffix
    if name.endswith("_0000") or name.endswith("_0001"):
        name = name[:-5]

    return name


def find_segmentation(patient_id, seg_dir):
    """
    Find matching segmentation file
    """

    possible_files = [
        seg_dir / f"{patient_id}.nii.gz",
        seg_dir / f"{patient_id}.nii"
    ]

    for file in possible_files:
        if file.exists():
            return file

    return None


def create_split(ct_dir, seg_dir, output_dir, train_ratio=0.8, seed=42):

    ct_dir = Path(ct_dir)
    seg_dir = Path(seg_dir)
    output_dir = Path(output_dir)

    imagesTr = output_dir / "imagesTr"
    labelsTr = output_dir / "labelsTr"
    imagesTs = output_dir / "imagesTs"
    labelsTs = output_dir / "labelsTs"

    for folder in [imagesTr, labelsTr, imagesTs, labelsTs]:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Check folder created: {folder.resolve()}")

    # Find CT files
    ct_files = (
        list(ct_dir.glob("*.nii")) +
        list(ct_dir.glob("*.nii.gz"))
    )

    print(f"Found CT files: {len(ct_files)}")

    # Group modalities by patient
    patients = {}

    for ct_file in ct_files:

        patient_id = get_patient_id(ct_file)

        if patient_id not in patients:
            patients[patient_id] = {}

        if "_0000" in ct_file.name:
            patients[patient_id]["0000"] = ct_file

        elif "_0001" in ct_file.name:
            patients[patient_id]["0001"] = ct_file


    # Check all files exist
    cases = []

    for patient_id, images in patients.items():

        if "0000" not in images:
            print(f"WARNING: Missing _0000 for {patient_id}")
            continue

        if "0001" not in images:
            print(f"WARNING: Missing _0001 for {patient_id}")
            continue

        seg_file = find_segmentation(
            patient_id,
            seg_dir
        )

        if seg_file is None:
            print(
                f"WARNING: No segmentation found for {patient_id}"
            )
            continue

        cases.append(
            (
                patient_id,
                images["0000"],
                images["0001"],
                seg_file
            )
        )

    print(f"Matched cases: {len(cases)}")


    # Shuffle and split by patient
    random.seed(seed)
    random.shuffle(cases)

    split_idx = int(len(cases) * train_ratio)

    train_cases = cases[:split_idx]
    test_cases = cases[split_idx:]

    print(f"Training cases: {len(train_cases)}")
    print(f"Testing cases: {len(test_cases)}")


    def copy_cases(case_list, image_dir, label_dir):

        for patient_id, ct_0000, ct_0001, seg_file in case_list:

            # Copy modality 0
            shutil.copy(
                ct_0000,
                image_dir / f"{patient_id}_0000.nii.gz"
            )

            # Copy modality 1
            shutil.copy(
                ct_0001,
                image_dir / f"{patient_id}_0001.nii.gz"
            )

            # Copy segmentation
            shutil.copy(
                seg_file,
                label_dir / f"{patient_id}.nii.gz"
            )

            print(
                f"Copied {patient_id}"
            )


    print("\nCopying training cases")
    copy_cases(
        train_cases,
        imagesTr,
        labelsTr
    )

    print("\nCopying test cases")
    copy_cases(
        test_cases,
        imagesTs,
        labelsTs
    )

    print("\nDone")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create nnU-Net multi-channel train/test split"
    )

    parser.add_argument(
        "--ct_dir",
        required=True,
        help="Folder containing CT seg"
    )

    parser.add_argument(
        "--seg_dir",
        required=True,
        help="Folder containing segmentation masks"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output nnU-Net dataset folder"
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training fraction (default=0.8)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()


    create_split(
        args.ct_dir,
        args.seg_dir,
        args.output_dir,
        args.train_ratio,
        args.seed
    )