import SimpleITK as sitk
import numpy as np


def load_nifti(path):

    img = sitk.ReadImage(str(path))

    array = sitk.GetArrayFromImage(img)

    spacing = img.GetSpacing()

    return array, spacing


def tumour_volume(mask, spacing):

    voxel_volume = (
        spacing[0]
        *
        spacing[1]
        *
        spacing[2]
    )

    n_voxels = np.sum(mask > 0)

    volume = (
        n_voxels *
        voxel_volume
    ) / 1000

    return volume


def hu_statistics(image, mask):

    values = image[mask > 0]

    if len(values) == 0:
        return {
            "mean_HU": np.nan,
            "median_HU": np.nan,
            "std_HU": np.nan,
            "min_HU": np.nan,
            "max_HU": np.nan,
            "p5_HU": np.nan,
            "p95_HU": np.nan,
            "iqr_HU": np.nan,
        }

    return {
        "mean_HU": values.mean(),
        "median_HU": np.median(values),
        "std_HU": values.std(),
        "min_HU": values.min(),
        "max_HU": values.max(),
        "p5_HU": np.percentile(values,5),
        "p95_HU": np.percentile(values,95),
        "iqr_HU": (
            np.percentile(values,75)
            -
            np.percentile(values,25)
        )
    }