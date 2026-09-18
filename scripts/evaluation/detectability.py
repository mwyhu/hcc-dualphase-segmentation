import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label

import nibabel as nib


def calculate_tumour_detectability(prediction, ground_truth, thresholds=(0.0, 0.15, 0.25)):

    prediction = prediction > 0
    ground_truth = ground_truth > 0

    pred_labels, n_pred = label(prediction)
    gt_labels, n_gt = label(ground_truth)
    
    iou_matrix = np.zeros((n_gt, n_pred), dtype=float)

    for gt_id in range(1, n_gt + 1):
        gt_component = gt_labels == gt_id

        for pred_id in range(1, n_pred + 1):
            pred_component = pred_labels == pred_id

            intersection = np.logical_and(
                gt_component,
                pred_component
            ).sum()

            union = np.logical_or(
                gt_component,
                pred_component
            ).sum()

            if union > 0:
                iou_matrix[gt_id - 1, pred_id - 1] = (
                    intersection / union
                )

    results = {}

    for threshold in thresholds:

        remaining_iou = iou_matrix.copy()

        matched_gt = set()
        matched_pred = set()

        while remaining_iou.size > 0:

            gt_idx, pred_idx = np.unravel_index(
                np.argmax(remaining_iou),
                remaining_iou.shape
            )

            max_iou = remaining_iou[gt_idx, pred_idx]

            if threshold == 0:
                if max_iou <= 0:
                    break
            else:
                if max_iou < threshold:
                    break

            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)

            remaining_iou[gt_idx, :] = -1
            remaining_iou[:, pred_idx] = -1

        tp = len(matched_gt)
        fn = n_gt - tp
        fp = n_pred - len(matched_pred)

        detectability = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else np.nan
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else np.nan
        )

        results[threshold] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "detectability": detectability,
        }

    return results


def load_mask(path):
    """
    Load a segmentation mask
    If the NIfTI contains invalid direction metadata, nibabel is used as fallback
    """
    try:
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)  # (z, y, x)

    except RuntimeError as error:
        if "orthonormal direction cosines" not in str(error):
            raise RuntimeError(f"Failed to read NIfTI file: {path}") from error

        print(
            f"WARNING: Invalid direction metadata in {path}. "
            "Loading voxel data with nibabel."
        )

        import nibabel as nib

        image = nib.load(str(path))
        array = np.asarray(image.dataobj)

        # nibabel: (x, y, z) → SimpleITK convention: (z, y, x)
        array = np.transpose(array, (2, 1, 0))

    return array > 0


def main():

    parser = argparse.ArgumentParser(
        description="Calculate tumour detectability."
    )

    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Directory containing predicted tumour masks."
    )

    parser.add_argument(
        "--gt_dir",
        type=Path,
        required=True,
        help="Directory containing ground-truth tumour masks."
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.15, 0.25],
        help="IoU thresholds."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV file."
    )

    args = parser.parse_args()

    rows = []

    gt_files = sorted(args.gt_dir.glob("*.nii.gz"))

    for gt_path in gt_files:

        pred_path = args.pred_dir / gt_path.name

        if not pred_path.exists():
            print(f"Prediction not found: {pred_path}")
            continue

        prediction = load_mask(pred_path)
        ground_truth = load_mask(gt_path)

        results = calculate_tumour_detectability(
            prediction,
            ground_truth,
            thresholds=args.thresholds,
        )

        for threshold, result in results.items():

            rows.append({
                "case": gt_path.name,
                "iou_threshold": threshold,
                "TP": result["TP"],
                "FP": result["FP"],
                "FN": result["FN"],
                "precision": result["precision"],
                "detectability": result["detectability"],
            })

    df = pd.DataFrame(rows)

    print(df)

    if args.output is not None:
        df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

