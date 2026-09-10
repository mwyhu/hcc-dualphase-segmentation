#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --job-name="preprocess_DP"
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


cd /projects/prjs2180/code/hcc-dualphase-segmentation

source setup_env.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started"
echo "Date and time:"
date
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLUR_JOB_ID"


nnUNetv2_preprocess \
    -d 002 \
    -plans_name TSLL_DP_plans \
    -c 3d_fullres \
    -np 16


echo "Job finished"
echo "Date and time:"
date