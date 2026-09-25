from pathlib import Path
import SimpleITK as sitk
import argparse


def apply_liver_mask(prediction_path, liver_path, output_path, margin_mm=10.0):

    prediction_img = sitk.ReadImage(str(prediction_path))
    liver_img = sitk.ReadImage(str(liver_path))

    # Resample liver mask to prediction geometry
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(prediction_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)

    liver_img = resampler.Execute(liver_img)

    prediction = sitk.GetArrayFromImage(prediction_img)

    # Convert liver mask to binary
    liver_binary = sitk.Cast(
        liver_img > 0,
        sitk.sitkUInt8
    )

    # Distance in mm from the liver, using the image spacing
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

    print(f"Applying {margin_mm} mm liver margin")

    liver = sitk.GetArrayFromImage(liver_dilated)

    # Check geometry
    if prediction.shape != liver.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"Prediction: {prediction.shape}\n"
            f"Liver: {liver.shape}"
        )

    # Remove predictions outside dilated liver
    prediction[liver == 0] = 0

    # Save result
    output_img = sitk.GetImageFromArray(prediction)
    output_img.CopyInformation(prediction_img)

    sitk.WriteImage(output_img, str(output_path))


def process_folder(prediction_dir, liver_dir, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = sorted(prediction_dir.glob("*.nii.gz"))

    print(f"Found {len(predictions)} predictions")

    for pred_path in predictions:

        case_id = pred_path.name.replace(".nii.gz", "")

        liver_path = next(liver_dir.rglob(pred_path.name), None)

        if liver_path is None:
            raise FileNotFoundError(
                f"No liver mask found for {case_id}"
            )

        output_path = output_dir / pred_path.name

        print(f"Processing {case_id}")

        apply_liver_mask(
            pred_path,
            liver_path,
            output_path,
            margin_mm=10.0
        )

    print("Finished liver post-processing")


def main():

    parser = argparse.ArgumentParser(
        description="Apply liver mask post-processing to nnU-Net predictions"
    )

    parser.add_argument(
        "--prediction_dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--liver_dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    process_folder(
        args.prediction_dir,
        args.liver_dir,
        args.output_dir
    )


if __name__ == "__main__":
    main()