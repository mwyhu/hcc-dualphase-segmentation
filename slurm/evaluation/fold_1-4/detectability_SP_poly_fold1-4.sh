#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --array=1-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --job-name=detectability_SP_poly
#SBATCH --output=logs/%x_%A_fold%a.out
#SBATCH --error=logs/%x_%A_fold%a.err

set -euo pipefail

cd /projects/prjs2180/code/hcc-dualphase-segmentation
source setup_env.sh

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

FOLD="$SLURM_ARRAY_TASK_ID"

OUTPUT_DIR="/projects/prjs2180/evaluation/detectability/fold_1-4"

PRED_DIR="/projects/prjs2180/data/nnUNet_results/TSLL_SP_1e3/Dataset001_singlephase/nnUNetTrainer__TSLL_SP_plans__3d_fullres/fold_${FOLD}/validation"

GT_DIR="/projects/prjs2180/data/nnUNet_raw/Dataset001_singlephase/labelsTr"

mkdir -p "$OUTPUT_DIR"

echo "Job started"
echo "Date and time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Fold: $FOLD"

python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR" \
    --gt_dir "$GT_DIR" \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR/poly_SP_fold${FOLD}.csv"

echo "Fold $FOLD finished"
echo "Date and time: $(date)"