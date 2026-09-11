#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --job-name="detectability_TS_vs_TSLL"
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


echo "TS DualPhase started"
python /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir /projects/prjs2180/data/nnUNet_results/TS_DP/Dataset002_dualphase/nnUNetTrainer__TSLL_DP_plans__3d_fullres/fold_0/validation \
    --gt_dir /projects/prjs2180/data/nnUNet_raw/Dataset002_dualphase/labelsTr \
    --thresholds 0.15 0.2 0.5 \
    --output /projects/prjs2180/evaluation/detectability/detectability_TS_DP.csv
echo "TS DualPhase finished"


echo "TSLL DualPhase started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir /projects/prjs2180/data/nnUNet_results/TSLL_DP/Dataset002_dualphase/nnUNetTrainer__TSLL_DP_plans__3d_fullres/fold_0/validation \
    --gt_dir /projects/prjs2180/data/nnUNet_raw/Dataset002_dualphase/labelsTr \
    --thresholds 0.15 0.2 0.5 \
    --output /projects/prjs2180/evaluation/detectability/detectability_TSLL_DP.csv
echo "TSLL DualPhase finished"


echo "TS SinglePhase started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir /projects/prjs2180/data/nnUNet_results/TS_DP/Dataset001_singlephase/nnUNetTrainer__TSLL_DP_plans__3d_fullres/fold_0/validation \
    --gt_dir /projects/prjs2180/data/nnUNet_raw/Dataset001_singlephase/labelsTr \
    --thresholds 0.15 0.2 0.5 \
    --output /projects/prjs2180/evaluation/detectability/detectability_TS_SP.csv
echo "TS SinglePhase finished"


echo "TSLL SinglePhase started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir /projects/prjs2180/data/nnUNet_results/TSLL_DP/Dataset001_singlephase/nnUNetTrainer__TSLL_DP_plans__3d_fullres/fold_0/validation \
    --gt_dir /projects/prjs2180/data/nnUNet_raw/Dataset001_singlephase/labelsTr \
    --thresholds 0.15 0.2 0.5 \
    --output /projects/prjs2180/evaluation/detectability/detectability_TSLL_SP.csv
echo "TSLL SinglePhase finished"


echo "Job finished"
echo "Date and time:"
date