#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --job-name=detectability_DP_1500_epochs
#SBATCH --output=logs/%x_%A_fold%a.out
#SBATCH --error=logs/%x_%A_fold%a.err

set -euo pipefail

cd /projects/prjs2180/code/hcc-dualphase-segmentation
source setup_env.sh

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"


OUTPUT_DIR="/projects/prjs2180/evaluation/detectability/extend_epochs"

PRED_DIR="/projects/prjs2180/data/nnUNet_results/extend_epochs/DP_poly_1e3_1500epochs/Dataset002_dualphase/nnUNetTrainer__TSLL_DP_plans__3d_fullres/fold_0/validation"

GT_DIR="/projects/prjs2180/data/nnUNet_raw/Dataset002_dualphase/labelsTr"

mkdir -p "$OUTPUT_DIR"

echo "Job started"
echo "Date and time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Fold: $FOLD"

python -u scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR" \
    --gt_dir "$GT_DIR" \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR/DP_1500_epochs.csv"

echo "Fold $FOLD finished"
echo "Date and time: $(date)"