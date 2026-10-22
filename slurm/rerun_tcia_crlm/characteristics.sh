#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --job-name="dataset_characteristics_TCIA"
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

cd /projects/prjs2180/code/hcc-dualphase-segmentation

source setup_env.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started"
echo "Date and time:"
date
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLUR_JOB_ID"


python -u /projects/prjs2180/code/hcc-dualphase-segmentation/data_characteristics/characteristics.py \
    --dataset TCIA_CRLM \
    --ct-dir /projects/prjs2180/data/nnUNet_raw/TCIA_CRLM/CT \
    --seg-dir /projects/prjs2180/data/nnUNet_raw/TCIA_CRLM/SEG \
    --output /projects/prjs2180/dataset_characteristics/TCIA_CRLM


echo "Job finished"
echo "Date and time:"
date