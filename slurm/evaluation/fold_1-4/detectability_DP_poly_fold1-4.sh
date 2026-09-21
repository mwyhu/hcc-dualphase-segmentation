#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --job-name="detectability_DP_poly_fold1-4"
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



echo "Dual Phase, Poly"
echo "Fold 1"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir /projects/prjs2180/data/nnUNet_results/TSLL_DP_1e3/Dataset002_dualphase/nnUNetTrainer__TSLL_DP_plans__3d_fullres/fold_1/validation \
    --gt_dir /projects/prjs2180/data/nnUNet_raw/Dataset002_dualphase/labelsTr \
    --thresholds 0.15 0.2 0.5 \
    --output /projects/prjs2180/evaluation/detectability/fold_1-4/poly_DP_fold1.csv

echo "Fold 2"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir /projects/prjs2180/data/nnUNet_results/TSLL_DP_1e3/Dataset002_dualphase/nnUNetTrainer__TSLL_DP_plans__3d_fullres/fold_2/validation \
    --gt_dir /projects/prjs2180/data/nnUNet_raw/Dataset002_dualphase/labelsTr \
    --thresholds 0.15 0.2 0.5 \
    --output /projects/prjs2180/evaluation/detectability/fold_1-4/poly_DP_fold2.csv

echo "Fold 3"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir /projects/prjs2180/data/nnUNet_results/TSLL_DP_1e3/Dataset002_dualphase/nnUNetTrainer__TSLL_DP_plans__3d_fullres/fold_3/validation \
    --gt_dir /projects/prjs2180/data/nnUNet_raw/Dataset002_dualphase/labelsTr \
    --thresholds 0.15 0.2 0.5 \
    --output /projects/prjs2180/evaluation/detectability/fold_1-4/poly_DP_fold3.csv

echo "Fold 4"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir /projects/prjs2180/data/nnUNet_results/TSLL_DP_1e3/Dataset002_dualphase/nnUNetTrainer__TSLL_DP_plans__3d_fullres/fold_4/validation \
    --gt_dir /projects/prjs2180/data/nnUNet_raw/Dataset002_dualphase/labelsTr \
    --thresholds 0.15 0.2 0.5 \
    --output /projects/prjs2180/evaluation/detectability/fold_1-4/poly_DP_fold4.csv
echo "Dual Phase Finished"

echo "Job finished"
echo "Date and time:"
date