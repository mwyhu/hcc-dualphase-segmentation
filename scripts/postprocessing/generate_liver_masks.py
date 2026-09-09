from pathlib import Path
import subprocess
import shutil
import argparse


def generate_liver_masks(input_dir, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(input_dir.glob("*.nii.gz"))

    print(f"Found {len(images)} seg")

    for image_path in images:

        case_id = image_path.name.replace(".nii.gz", "")

        print(f"\nProcessing {case_id}")

        # Temporary TotalSegmentator output
        temp_dir = output_dir / f"{case_id}_tmp"

        cmd = [
            "TotalSegmentator",
            "-i",
            str(image_path),
            "-o",
            str(temp_dir),
            "--roi_subset",
            "liver"
        ]

        subprocess.run(cmd, check=True)

        liver_file = temp_dir / "liver.nii.gz"

        if not liver_file.exists():
            raise FileNotFoundError(
                f"Liver mask not found for {case_id}"
            )

        output_file = output_dir / f"{case_id}.nii.gz"

        shutil.copy(
            liver_file,
            output_file
        )

        print(f"Saved: {output_file}")

        # Remove temporary TotalSegmentator folder
        shutil.rmtree(temp_dir)

    print("\nFinished generating liver masks")


def main():

    parser = argparse.ArgumentParser(
        description="Generate liver masks using TotalSegmentator"
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing CT seg"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where liver masks are saved"
    )

    args = parser.parse_args()

    generate_liver_masks(
        args.input_dir,
        args.output_dir
    )


if __name__ == "__main__":
    main()