#!/bin/bash
#SBATCH --job-name=predict_ceiling_internal_DP_testset
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

export nnUNet_results=/projects/prjs2180/data/nnUNet_results/ceiling
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

INPUT_DIR=/projects/prjs2180/data/nnUNet_raw/Dataset002_dualphase/imagesTs
OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/ceiling/internal_DP_testset
mkdir -p "$OUTPUT_DIR"

nnUNetv2_predict \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -d 3 \
    -f all \
    -c 3d_fullres \
    -tr nnUNetTrainer \
    -p TSLL_ceiling_plans \
    -chk checkpoint_final.pth