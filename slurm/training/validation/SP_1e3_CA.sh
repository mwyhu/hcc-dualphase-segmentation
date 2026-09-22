#!/bin/bash
#SBATCH --job-name=SP_CA_1e3
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/%x_fold%A_%a.out
#SBATCH --error=logs/%x_fold%A_%a.err

set -euo pipefail

cd /projects/prjs2180/code/hcc-dualphase-segmentation
source setup_env.sh

export nnUNet_n_proc_DA="$SLURM_CPUS_PER_TASK"
export nnUNet_results=/projects/prjs2180/data/nnUNet_results/cosannealing/TSLL_SP_1e3

FOLD="$SLURM_ARRAY_TASK_ID"

echo "Job started"
echo "Date and time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Training fold: $FOLD"

nnUNetv2_train \
    1 \
    3d_fullres \
    "$FOLD" \
    -tr nnUNetTrainerCosAnneal \
    -p TSLL_SP_plans \
    --val

echo "Fold $FOLD finished"
echo "Date and time: $(date)"