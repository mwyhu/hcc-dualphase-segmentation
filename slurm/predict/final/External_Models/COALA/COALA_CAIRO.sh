#!/bin/bash
#SBATCH --job-name=COALA_CAIRO
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

LOWRES_MODEL_DIR=/projects/prjs2180/code/COALA/nnUNetTrainer__nnUNetPlans__3d_lowres
CASCADE_MODEL_DIR=/projects/prjs2180/code/COALA/nnUNetTrainer__nnUNetPlans__3d_cascade_fullres

INPUT_DIR=/projects/prjs2180/data/nnUNet_raw/0_External_Validation/SP/CAIRO5/imagesTs

OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/CAIRO5/COALA

mkdir -p "$LOWRES_OUTPUT_DIR" "$OUTPUT_DIR"

nnUNetv2_predict \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -m "$CASCADE_MODEL_DIR" \
    -f 0 1 2 3 4 \
    -prev_stage_predictions "$LOWRES_MODEL_DIR"

echo "COALA prediction completed"
echo "Final predictions: $OUTPUT_DIR"