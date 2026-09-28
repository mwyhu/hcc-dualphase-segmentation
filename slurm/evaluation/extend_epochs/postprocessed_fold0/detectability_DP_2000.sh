#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --job-name=detectability_HCC_2000
#SBATCH --output=logs/%x_%A_fold%a.out
#SBATCH --error=logs/%x_%A_fold%a.err

set -euo pipefail

cd /projects/prjs2180/code/hcc-dualphase-segmentation
source setup_env.sh

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"


OUTPUT_DIR="/projects/prjs2180/evaluation/detectability/extend_epochs/external_val_HCC"

PRED_DIR="/projects/prjs2180/evaluation/predictions/HCC_TACE/DP_fold0_2000"

GT_DIR="/projects/prjs2180/data/nnUNet_raw/Full_HCC_TACE_DP/labelsTs"

mkdir -p "$OUTPUT_DIR"

echo "Job started"
echo "Date and time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"


python -u scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR" \
    --gt_dir "$GT_DIR" \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR/DP_2000_epochs.csv"


echo "Date and time: $(date)"