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

LOWRES_MODEL_DIR=/projects/prjs2180/code/COALA/nnUNetTrainer__nnUNetPlans__3d_lowres
CASCADE_MODEL_DIR=/projects/prjs2180/code/COALA/nnUNetTrainer__nnUNetPlans__3d_cascade_fullres

INPUT_DIR=/projects/prjs2180/data/nnUNet_raw/0_External_Validation/SP/Full_TCIA_CRLM/imagesTs

LOWRES_OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/TCIA_CRLM/COALA_lowres
OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/TCIA_CRLM/COALA

mkdir -p "$LOWRES_OUTPUT_DIR" "$OUTPUT_DIR"

shopt -s nullglob
images=("$INPUT_DIR"/*_0000.nii.gz)

if ((${#images[@]} == 0)); then
    echo "No *_0000.nii.gz images found in $INPUT_DIR" >&2
    exit 1
fi

# Check both models before starting inference.
for model_dir in "$LOWRES_MODEL_DIR" "$CASCADE_MODEL_DIR"; do
    for filename in dataset.json plans.json; do
        if [[ ! -f "$model_dir/$filename" ]]; then
            echo "Missing model file: $model_dir/$filename" >&2
            exit 1
        fi
    done

    for fold in 0 1 2 3 4; do
        checkpoint="$model_dir/fold_${fold}/checkpoint_final.pth"

        if [[ ! -f "$checkpoint" ]]; then
            echo "Missing checkpoint: $checkpoint" >&2
            exit 1
        fi
    done
done

echo "Stage 1: COALA low-resolution prediction, folds 0–4"

nnUNetv2_predict_from_modelfolder \
    -i "$INPUT_DIR" \
    -o "$LOWRES_OUTPUT_DIR" \
    -m "$LOWRES_MODEL_DIR" \
    -f 0 1 2 3 4 \
    -chk checkpoint_final.pth \
    -device cuda

echo "Stage 2: COALA cascade prediction, folds 0–4"

nnUNetv2_predict_from_modelfolder \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -m "$CASCADE_MODEL_DIR" \
    -f 0 1 2 3 4 \
    -chk checkpoint_final.pth \
    -prev_stage_predictions "$LOWRES_OUTPUT_DIR" \
    -device cuda

echo "COALA prediction completed"
echo "Final predictions: $OUTPUT_DIR"