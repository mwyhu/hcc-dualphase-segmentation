#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name="extract_fingerprint2"
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


nnUNetv2_extract_fingerprint \
    -d 001 \
    --verify_dataset_integrity


nnUNetv2_extract_fingerprint \
    -d 002 \
    --verify_dataset_integrity


echo "Job finished"
echo "Date and time:"
date