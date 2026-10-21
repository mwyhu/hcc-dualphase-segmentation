#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name="SP_on_ceiling_testset"
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p /projects/prjs2180/evaluation/detectability

cd /projects/prjs2180/code/hcc-dualphase-segmentation

source setup_env.sh


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started"
echo "Date and time:"
date
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLUR_JOB_ID"

PRED_DIR=/projects/prjs2180/evaluation/predictions/ceiling_SP/SP_on_ceiling_testset
GT_DIR=/projects/prjs2180/data/nnUNet_raw/Dataset004_ceiling_SP/labelsTs
OUTPUT_DIR=/projects/prjs2180/evaluation/detectability/ceiling
mkdir -p "$OUTPUT_DIR"


python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR" \
    --gt_dir "$GT_DIR" \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR"/Detect_SP_on_ceiling_testset.csv




echo "Job finished"
echo "Date and time:"
date

