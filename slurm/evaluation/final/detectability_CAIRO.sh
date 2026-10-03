#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name="final_detectability_CAIRO"
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
GT_DIR=/projects/prjs2180/data/nnUNet_raw/0_External_Validation
OUTPUT_DIR=/projects/prjs2180/evaluation/detectability/final
mkdir -p "$OUTPUT_DIR"

echo "AN Started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR"/CAIRO5/AN \
    --gt_dir "$GT_DIR"/SP/CAIRO5/labelsTs \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR"/AN/Detect_AN_CAIRO5.csv
echo "AN Finished"


echo "COALA Started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR"/CAIRO5/COALA \
    --gt_dir "$GT_DIR"/SP/CAIRO5/labelsTs \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR"/COALA/Detect_COALA_CAIRO5.csv
echo "COALA Finished"


echo "TSLL Started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR"/CAIRO5/TotalSegmentator_liver_lesions \
    --gt_dir "$GT_DIR"/SP/CAIRO5/labelsTs \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR"/TotalSegmentator_liver_lesions/Detect_TSLL_CAIRO5.csv
echo "TSLL Finished"


echo "TSLT Started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR"/CAIRO5/TotalSegmentator_liver_tumor \
    --gt_dir "$GT_DIR"/SP/CAIRO5/labelsTs \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR"/TotalSegmentator_liver_tumor/Detect_TSLT_CAIRO5.csv
echo "TSLT Finished"


echo "SP Started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR"/CAIRO5/SP \
    --gt_dir "$GT_DIR"/SP/CAIRO5/labelsTs \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR"/SP/Detect_SP_CAIRO5.csv
echo "SP Finished"


echo "DP Started"
python -u /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/evaluation/detectability.py \
    --pred_dir "$PRED_DIR"/CAIRO5/DP \
    --gt_dir "$GT_DIR"/DP/CAIRO5/labelsTs \
    --thresholds 0.15 0.2 0.5 \
    --output "$OUTPUT_DIR"/DP/Detect_DP_CAIRO5.csv
echo "DP Finished"


echo "Job finished"
echo "Date and time:"
date

