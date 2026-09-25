#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --job-name="apply_liver_mask_2000ep"
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


python /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/postprocessing/apply_liver_mask.py \
    --prediction_dir /projects/prjs2180/data/nnUNet_results/extend_epochs/DP_poly_1e3_2000epochs/Dataset002_dualphase/nnUNetTrainer__TSLL_DP_plans__3d_fullres/fold_0/validation \
    --liver_dir /projects/prjs2180/liver_masks \
    --output_dir /projects/prjs2180/postprocessed/validation/DP_2000ep


echo "Job finished"
echo "Date and time:"
date