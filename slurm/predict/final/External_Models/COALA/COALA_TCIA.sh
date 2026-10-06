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
export PYTHONNOUSERSITE=1

cd /projects/prjs2180/code/COALA/nnUNet

LOWRES_MODEL=/projects/prjs2180/code/COALA/nnUNetTrainer__nnUNetPlans__3d_lowres
CASCADE_MODEL=/projects/prjs2180/code/COALA/nnUNetTrainer__nnUNetPlans__3d_cascade_fullres

INPUT_DIR=/projects/prjs2180/data/nnUNet_raw/0_External_Validation/SP/Full_TCIA_CRLM/imagesTs
LOWRES_OUTPUT=/projects/prjs2180/evaluation/predictions/TCIA_CRLM/COALA_lowres
CASCADE_OUTPUT=/projects/prjs2180/evaluation/predictions/TCIA_CRLM/COALA

mkdir -p "$LOWRES_OUTPUT" "$CASCADE_OUTPUT"

echo "Starting lowres prediction: $(date)"

nnUNetv2_predict_from_modelfolder \
    -i "$INPUT_DIR" \
    -o "$LOWRES_OUTPUT" \
    -m "$LOWRES_MODEL" \
    -f 0 1 2 3 4 \
    -chk checkpoint_final.pth

echo "Lowres finished; starting cascade: $(date)"

nnUNetv2_predict_from_modelfolder \
    -i "$INPUT_DIR" \
    -o "$CASCADE_OUTPUT" \
    -m "$CASCADE_MODEL" \
    -f 0 1 2 3 4 \
    -chk checkpoint_final.pth \
    -prev_stage_predictions "$LOWRES_OUTPUT"

echo "Cascade finished: $(date)"