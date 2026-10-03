#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name="final_detectability_TSLT"
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

PRED_DIR=/projects/prjs2180/evaluation/predictions
GT_DIR=/projects/prjs2180/data/nnUNet_raw/0_External_Validation/SP
OUTPUT_DIR=/projects/prjs2180/evaluation/detectability/final/TotalSegmentator_liver_tumor
mkdir -p "$OUTPUT_DIR"

echo "HCC Started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR"/HCC_TACE/TotalSegmentator_liver_tumor \
    --gt_dir "$GT_DIR"/Full_HCC_TACE/labelsTs \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR"/Detect_TSLT_HCC.csv
echo "HCC Finished"


echo "LiTS Started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR"/LiTS/TotalSegmentator_liver_tumor \
    --gt_dir "$GT_DIR"/Full_LiTS/labelsTs \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR"/Detect_TSLT_LiTS.csv
echo "LiTS Finished"


echo "WORC Started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR"/WORC_CRLM/TotalSegmentator_liver_tumor \
    --gt_dir "$GT_DIR"/Full_WORC_CRLM/labelsTs \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR"/Detect_TSLT_WORC.csv
echo "WORC Finished"


echo "TCIA Started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR"/TCIA_CRLM/TotalSegmentator_liver_tumor \
    --gt_dir "$GT_DIR"/Full_TCIA_CRLM/labelsTs \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR"/Detect_TSLT_TCIA.csv
echo "TCIA Finished"


echo "Job finished"
echo "Date and time:"
date

