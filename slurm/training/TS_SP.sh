#!/bin/bash
#SBATCH --job-name=TS_SP
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=60:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /projects/prjs2180/code/hcc-dualphase-segmentation

source setup_env.sh

export nnUNet_n_proc_DA=$SLURM_CPUS_PER_TASK

echo "Job started"
echo "Date and time:"
date
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

export nnUNet_results=/projects/prjs2180/data/nnUNet_results/TS_SP

nnUNetv2_train \
    1 \
    3d_fullres \
    0 \
    -tr nnUNetTrainer \
    -p TSLL_SP_plans \
    --npz

echo "Job finished"
echo "Date and time:"
date