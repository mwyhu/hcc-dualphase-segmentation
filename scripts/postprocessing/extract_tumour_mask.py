import nibabel as nib
import numpy as np
from pathlib import Path
import argparse


def extract_tumour_masks(input_dir, output_dir, tumour_label=2):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz"))

    for file in files:
        nii = nib.load(file)
        seg = nii.get_fdata()

        tumour_mask = (seg == tumour_label).astype(np.uint8)

        out = nib.Nifti1Image(
            tumour_mask,
            nii.affine,
            nii.header
        )

        output_file = output_dir / f"{file.stem}_tumour_mask.nii.gz"

        nib.save(out, output_file)

        print(f"Processed {file.name} → {output_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract tumour class from NIfTI segmentation masks"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing segmentation masks"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder to save extracted tumour masks"
    )

    parser.add_argument(
        "--tumour_label",
        type=int,
        default=2,
        help="Label value representing tumour (default: 2)"
    )

    args = parser.parse_args()

    extract_tumour_masks(
        args.input_dir,
        args.output_dir,
        args.tumour_label
    )