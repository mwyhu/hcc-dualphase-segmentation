#!/bin/bash
#SBATCH --job-name=predict_TCIA_CRLM_SP
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

export nnUNet_results=/projects/prjs2180/data/nnUNet_results/extend_epochs/SP_poly_1e3_2000epochs
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

INPUT_DIR=/projects/prjs2180/data/nnUNet_raw/0_External_Validation/SP/Full_TCIA_CRLM/imagesTs
OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/TCIA_CRLM/SP
mkdir -p "$OUTPUT_DIR"

nnUNetv2_predict \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -d 1 \
    -c 3d_fullres \
    -tr nnUNetTrainer \
    -p TSLL_SP_plans \
    -chk checkpoint_final.pth