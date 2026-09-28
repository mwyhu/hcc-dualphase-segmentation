#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --job-name=detectability_1500_pp
#SBATCH --output=logs/%x_%A_fold%a.out
#SBATCH --error=logs/%x_%A_fold%a.err

set -euo pipefail

cd /projects/prjs2180/code/hcc-dualphase-segmentation
source setup_env.sh

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"


OUTPUT_DIR="/projects/prjs2180/evaluation/detectability/extend_epochs/postprocessed_fold0"

PRED_DIR="/projects/prjs2180/postprocessed/validation/DP_1500ep"

GT_DIR="/projects/prjs2180/GT_fold0"

mkdir -p "$OUTPUT_DIR"

echo "Job started"
echo "Date and time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"


python -u scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR" \
    --gt_dir "$GT_DIR" \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR/DP_1500_epochs.csv"


echo "Date and time: $(date)"