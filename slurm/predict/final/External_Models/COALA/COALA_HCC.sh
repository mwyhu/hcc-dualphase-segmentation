#!/bin/bash
#SBATCH --job-name=COALA_HCC
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

INPUT_DIR=/projects/prjs2180/data/nnUNet_raw/0_External_Validation/SP/Full_HCC_TACE/imagesTs

LOWRES_OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/HCC_TACE/COALA_lowres
OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/HCC_TACE/COALA

mkdir -p "$LOWRES_OUTPUT_DIR" "$OUTPUT_DIR"

echo "Starting low-resolution prediction"

nnUNetv2_predict_from_modelfolder \
    -i "$INPUT_DIR" \
    -o "$LOWRES_OUTPUT_DIR" \
    -m "$LOWRES_MODEL_DIR" \
    -f 0 1 2 3 4 \
    --continue_prediction

echo "Starting cascade prediction"

nnUNetv2_predict_from_modelfolder \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -m "$CASCADE_MODEL_DIR" \
    -f 0 1 2 3 4 \
    -prev_stage_predictions "$LOWRES_OUTPUT_DIR" \
    --continue_prediction

echo "COALA prediction completed"
echo "Final predictions: $OUTPUT_DIR"