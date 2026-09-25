from pathlib import Path
import argparse
import shutil
import subprocess


def generate_liver_masks(input_dir: Path, output_dir: Path) -> None:
    """
    Generate one liver mask per case from the _0000 singlephase
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(input_dir.glob("*_0000.nii.gz"))
    print(f"Found {len(images)} CT scans")

    if not images:
        raise FileNotFoundError(
            f"No files ending in _0000.nii.gz found in {input_dir}"
        )

    for image_path in images:
        case_id = image_path.name.removesuffix("_0000.nii.gz")
        output_file = output_dir / f"{case_id}.nii.gz"
        temp_dir = output_dir / f"{case_id}_tmp"

        print(f"\nProcessing {case_id}")

        if temp_dir.exists():
            raise FileExistsError(
                f"Temporary directory already exists: {temp_dir}"
            )

        try:
            subprocess.run(
                [
                    "TotalSegmentator",
                    "-i", str(image_path),
                    "-o", str(temp_dir),
                    "--roi_subset", "liver",
                ],
                check=True,
            )

            liver_file = temp_dir / "liver.nii.gz"
            if not liver_file.is_file():
                raise FileNotFoundError(
                    f"TotalSegmentator did not produce a liver mask for {case_id}"
                )

            shutil.copy2(liver_file, output_file)
            print(f"Saved: {output_file}")

        finally:
            # Keep the temp output if the command failed
            if output_file.is_file() and temp_dir.exists():
                shutil.rmtree(temp_dir)

    print("\nFinished generating liver masks")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate liver masks with TotalSegmentator."
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing nnU-Net CT images named <case_id>_0000.nii.gz",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for liver masks named <case_id>.nii.gz",
    )

    args = parser.parse_args()
    generate_liver_masks(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()