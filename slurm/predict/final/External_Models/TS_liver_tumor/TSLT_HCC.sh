#!/bin/bash
#SBATCH --job-name=totalseg_HCC_tumor
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

INPUT_DIR=/projects/prjs2180/data/nnUNet_raw/0_External_Validation/SP/Full_HCC_TACE/imagesTs
OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/HCC_TACE/TotalSegmentator_liver_tumor

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
images=("$INPUT_DIR"/*_0000.nii.gz)

if ((${#images[@]} == 0)); then
    echo "No *_0000.nii.gz images found in $INPUT_DIR" >&2
    exit 1
fi

for image in "${images[@]}"; do
    filename=${image##*/}
    case_id=${filename%_0000.nii.gz}
    output_file="$OUTPUT_DIR/$case_id.nii.gz"

    echo "Processing $case_id"
    temp_dir=$(mktemp -d "$OUTPUT_DIR/.${case_id}.XXXXXX")

    TotalSegmentator \
        -i "$image" \
        -o "$temp_dir" \
        --task liver_vessels

    mv "$temp_dir/liver_tumor.nii.gz" "$output_file"
    rm -r "$temp_dir"

    echo "Saved $output_file"
done