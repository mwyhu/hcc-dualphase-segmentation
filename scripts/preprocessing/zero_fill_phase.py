import argparse
from pathlib import Path
import SimpleITK as sitk


def create_missing_channel(images_dir):
    images_dir = Path(images_dir)

    cases = set()

    for f in images_dir.glob("*.nii.gz"):
        if f.name.endswith("_0000.nii.gz"):
            cases.add(f.name.replace("_0000.nii.gz", ""))
        elif f.name.endswith("_0001.nii.gz"):
            cases.add(f.name.replace("_0001.nii.gz", ""))

    for case_name in sorted(cases):

        ch0 = images_dir / f"{case_name}_0000.nii.gz"
        ch1 = images_dir / f"{case_name}_0001.nii.gz"

        # Both channels exist
        if ch0.exists() and ch1.exists():
            print(f"Skipping {case_name}: both channels exist")
            continue

        # Only channel 0 exists -> create zero channel 1
        if ch0.exists():
            print(f"{case_name}: creating zero _0001")

            ref_img = sitk.ReadImage(str(ch0))

            zero_img = sitk.Image(
                ref_img.GetSize(),
                sitk.sitkInt16
            )
            zero_img.CopyInformation(ref_img)

            sitk.WriteImage(
                zero_img,
                str(ch1)
            )

        # only channel 1 exists -> create zero channel 0
        elif ch1.exists():
            print(f"{case_name}: creating zero _0000")

            ref_img = sitk.ReadImage(str(ch1))

            zero_img = sitk.Image(
                ref_img.GetSize(),
                sitk.sitkInt16
            )
            zero_img.CopyInformation(ref_img)

            sitk.WriteImage(
                zero_img,
                str(ch0)
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create missing zero-filled nnUNet CT channel"
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        required=True
    )

    args = parser.parse_args()

    create_missing_channel(args.images_dir)