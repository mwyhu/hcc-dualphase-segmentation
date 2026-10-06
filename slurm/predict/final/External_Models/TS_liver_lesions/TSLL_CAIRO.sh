#!/bin/bash
#SBATCH --job-name=totalseg_CAIRO_lesions
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=25:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /projects/prjs2180/code/hcc-dualphase-segmentation
source setup_env.sh

INPUT_DIR=/projects/prjs2180/data/nnUNet_raw/0_External_Validation/SP/CAIRO5/imagesTs
OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/CAIRO5/TotalSegmentator_liver_lesions

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
images=("$INPUT_DIR"/*_0001.nii.gz)

if ((${#images[@]} == 0)); then
    echo "No *_0001.nii.gz images found in $INPUT_DIR" >&2
    exit 1
fi

for image in "${images[@]}"; do
    filename=${image##*/}
    case_id=${filename%_0001.nii.gz}

    echo "Processing $case_id"
    TotalSegmentator \
        -i "$image" \
        -o "$OUTPUT_DIR/$case_id.nii.gz" \
        --task liver_lesions \
        --ml
done