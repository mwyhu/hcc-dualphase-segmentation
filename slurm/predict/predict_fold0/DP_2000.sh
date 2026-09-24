#!/bin/bash
#SBATCH --job-name=predict_fold0_2000epochs
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /projects/prjs2180/code/hcc-dualphase-segmentation
source setup_env.sh

export nnUNet_results=/projects/prjs2180/data/nnUNet_results/extend_epochs/DP_poly_1e3_2000epochs
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

INPUT_DIR=/projects/prjs2180/data/nnUNet_raw/Full_HCC_TACE_DP/imagesTs
OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/HCC_TACE/DP_fold0_2000
mkdir -p "$OUTPUT_DIR"

nnUNetv2_predict \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -d 2 \
    -c 3d_fullres \
    -tr nnUNetTrainer \
    -p TSLL_DP_plans \
    -f 0 \
    -chk checkpoint_final.pth