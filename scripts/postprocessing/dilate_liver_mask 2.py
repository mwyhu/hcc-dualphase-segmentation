from pathlib import Path
import SimpleITK as sitk
import argparse


def dilate_liver_mask(liver_path, output_path, margin_mm=10.0):

    liver_img = sitk.ReadImage(str(liver_path))

    # Convert liver mask to binary
    liver_binary = sitk.Cast(liver_img > 0, sitk.sitkUInt8)

    # Distance from the liver in millimetres
    distance_mm = sitk.SignedMaurerDistanceMap(
        liver_binary,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True
    )

    # Include the liver and voxels within the margin
    liver_dilated = sitk.Cast(
        distance_mm <= margin_mm,
        sitk.sitkUInt8
    )

    sitk.WriteImage(liver_dilated, str(output_path))


def process_folder(liver_dir, output_dir, margin_mm=10.0):

    liver_paths = sorted(liver_dir.glob("*.nii"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(liver_paths)} liver masks")

    for liver_path in liver_paths:
        output_path = output_dir / liver_path.name

        print(f"Dilating {liver_path.name} by {margin_mm} mm")

        dilate_liver_mask(
            liver_path,
            output_path,
            margin_mm=margin_mm
        )

    print("Finished saving dilated liver masks")


def main():

    parser = argparse.ArgumentParser(
        description="Dilate liver masks by a margin in millimetres"
    )

    parser.add_argument("--liver_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--margin_mm", type=float, default=10.0)

    args = parser.parse_args()

    if args.margin_mm < 0:
        parser.error("--margin_mm must be non-negative")

    process_folder(
        args.liver_dir,
        args.output_dir,
        margin_mm=args.margin_mm
    )


if __name__ == "__main__":
    main()