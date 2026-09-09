import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from feature_utils import (load_nifti, tumour_volume, hu_statistics)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_phase_files(
    image_dir,
    patient_id,
    phase_mapping
):

    phases = {}

    for phase, suffix in phase_mapping.items():

        path = (
            image_dir /
            f"{patient_id}{suffix}.nii.gz"
        )

        if path.exists():
            phases[phase] = path

    return phases


def extract_dataset_features(cfg):

    results = []

    for dataset_name, dataset_cfg in cfg["datasets"].items():

        print(f"Processing {dataset_name}")

        image_dir = Path(dataset_cfg["image_dir"])
        mask_dir = Path(dataset_cfg["mask_dir"])

        phase_mapping = dataset_cfg["phase_mapping"]

        masks = sorted(mask_dir.glob("*.nii.gz"))

        for mask_path in masks:

            # remove extension
            patient_id = mask_path.name.replace(
                ".nii.gz", ""
            )

            # load mask
            mask, spacing = load_nifti(mask_path)

            n_voxels = np.sum(mask > 0)

            if n_voxels == 0:
                print(
                    dataset_name,
                    patient_id,
                    "EMPTY MASK - SKIPPING",
                    flush=True
                )
                continue

            volume = tumour_volume(
                mask,
                spacing
            )

            phases = find_phase_files(
                image_dir,
                patient_id,
                phase_mapping
            )

            for phase, image_path in phases.items():

                image, _ = load_nifti(image_path)

                # print(
                #     dataset_name,
                #     patient_id,
                #     phase,
                #     "mask voxels:",
                #     np.sum(mask > 0),
                #     "image shape:",
                #     image.shape,
                #     "mask shape:",
                #     mask.shape
                # )

                hu = hu_statistics(
                    image,
                    mask
                )

                row = {
                    "dataset": dataset_name,
                    "domain": dataset_cfg["domain"],
                    "patient": patient_id,
                    "phase": phase,
                    "tumour_volume_cm3": volume,
                }

                row.update(hu)

                results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":

    cfg = load_config(
        "/projects/prjs2180/code/hcc_active_learning/analysis/config.yaml"
    )

    df = extract_dataset_features(cfg)

    output_path = Path(
        "/projects/prjs2180/dataset_characterisation"
    )

    output_path.mkdir(
        exist_ok=True
    )

    df.to_csv(
        output_path / "patient_features.csv",
        index=False
    )

    print(f"Saved {len(df)} rows")