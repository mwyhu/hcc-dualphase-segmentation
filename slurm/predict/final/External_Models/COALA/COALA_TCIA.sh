#!/bin/bash
#SBATCH --job-name=COALA_TCIA
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

MODEL_DIR=/projects/prjs2180/code/AbdomentAtlasNet
INPUT_DIR=/projects/prjs2180/data/nnUNet_raw/0_External_Validation/SP/Full_TCIA_CRLM/imagesTs
OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/TCIA_CRLM/COALA

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
images=("$INPUT_DIR"/*_0000.nii.gz)

if ((${#images[@]} == 0)); then
    echo "No *_0000.nii.gz images found in $INPUT_DIR" >&2
    exit 1
fi

for required_file in \
    "$MODEL_DIR/dataset.json" \
    "$MODEL_DIR/plans.json"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing model file: $required_file" >&2
        exit 1
    fi
done

for fold in 0 1 2 3 4; do
    checkpoint="$MODEL_DIR/fold_${fold}/checkpoint_final.pth"

    if [[ ! -f "$checkpoint" ]]; then
        echo "Missing fold checkpoint: $checkpoint" >&2
        exit 1
    fi
done

echo "Predicting ${#images[@]} scans with COALA using folds 0–4"
echo "Output directory: $OUTPUT_DIR"

nnUNetv2_predict_from_modelfolder \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -m "$MODEL_DIR" \
    -f 0 1 2 3 4 \
    -chk checkpoint_final.pth \
    -device cuda

echo "COALA prediction completed"