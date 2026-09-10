#!/bin/bash
#SBATCH --job-name=TSLL_DP
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

export nnUNet_results=/projects/prjs2180/data/nnUNet_results/TSLL_DP

nnUNetv2_train \
    2 \
    3d_fullres \
    0 \
    -tr nnUNetTrainer \
    -p TSLL_DP_plans \
    -pretrained_weights /projects/prjs2180/pretrained_models/TotalSegmentator/591_liver_lesions/checkpoint_final_dualphase.pth \
    --npz

echo "Job finished"
echo "Date and time:"
date

