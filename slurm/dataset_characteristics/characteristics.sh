#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --job-name="dataset_characteristics"
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


python -u /projects/prjs2180/code/hcc-dualphase-segmentation/data_characteristics/extract_characteristics.py \
    --dataset WAW_TACE_grid_fixed \
    --ct-dir /projects/prjs2180/data/nnUNet_raw/WAW_TACE_grid_fixed/CT \
    --seg-dir /projects/prjs2180/data/nnUNet_raw/WAW_TACE_grid_fixed/SEG \
    --output /projects/prjs2180/dataset_characteristics/WAW_TACE


python -u /projects/prjs2180/code/hcc-dualphase-segmentation/data_characteristics/extract_characteristics.py \
    --dataset HCC_TACE \
    --ct-dir /projects/prjs2180/data/nnUNet_raw/HCC_TACE/CT \
    --seg-dir /projects/prjs2180/data/nnUNet_raw/HCC_TACE/SEG \
    --output /projects/prjs2180/dataset_characteristics/HCC_TACE


python -u /projects/prjs2180/code/hcc-dualphase-segmentation/data_characteristics/extract_characteristics.py \
    --dataset LiTS \
    --ct-dir /projects/prjs2180/data/nnUNet_raw/LiTS/CT \
    --seg-dir /projects/prjs2180/data/nnUNet_raw/LiTS/SEG \
    --output /projects/prjs2180/dataset_characteristics/LiTS


python -u /projects/prjs2180/code/hcc-dualphase-segmentation/data_characteristics/extract_characteristics.py \
    --dataset Liver_Lesions \
    --ct-dir /projects/prjs2180/data/nnUNet_raw/Liver_Lesions/CT \
    --seg-dir /projects/prjs2180/data/nnUNet_raw/Liver_Lesions/SEG \
    --output /projects/prjs2180/dataset_characteristics/Liver_Lesions


python -u /projects/prjs2180/code/hcc-dualphase-segmentation/data_characteristics/extract_characteristics.py \
    --dataset MCT_LTDiag_grid_fixed \
    --ct-dir /projects/prjs2180/data/nnUNet_raw/MCT_LTDiag_grid_fixed/CT \
    --seg-dir /projects/prjs2180/data/nnUNet_raw/MCT_LTDiag_grid_fixed/SEG \
    --output /projects/prjs2180/dataset_characteristics/MCT_LTDiag


python -u /projects/prjs2180/code/hcc-dualphase-segmentation/data_characteristics/extract_characteristics.py \
    --dataset TCIA_CRLM \
    --ct-dir /projects/prjs2180/data/nnUNet_raw/TCIA_CRLM/CT \
    --seg-dir /projects/prjs2180/data/nnUNet_raw/TCIA_CRLM/SEG \
    --output /projects/prjs2180/dataset_characteristics/TCIA_CRLM


python -u /projects/prjs2180/code/hcc-dualphase-segmentation/data_characteristics/extract_characteristics.py \
    --dataset WORC_CRLM \
    --ct-dir /projects/prjs2180/data/nnUNet_raw/WORC_CRLM/CT \
    --seg-dir /projects/prjs2180/data/nnUNet_raw/WORC_CRLM/SEG \
    --output /projects/prjs2180/dataset_characteristics/WORC_CRLM


python -u /projects/prjs2180/code/hcc-dualphase-segmentation/data_characteristics/extract_characteristics.py \
    --dataset MSD \
    --ct-dir /projects/prjs2180/data/nnUNet_raw/MSD/CT \
    --seg-dir /projects/prjs2180/data/nnUNet_raw/MSD/SEG \
    --output /projects/prjs2180/dataset_characteristics/MSD


python -u /projects/prjs2180/code/hcc-dualphase-segmentation/data_characteristics/extract_characteristics.py \
    --dataset CAIRO5 \
    --ct-dir /projects/prjs2180/data/nnUNet_raw/0_External_Validation/DP/CAIRO5/imagesTs \
    --seg-dir /projects/prjs2180/data/nnUNet_raw/0_External_Validation/DP/CAIRO5/labelsTs \
    --output /projects/prjs2180/dataset_characteristics/CAIRO5


echo "Job finished"
echo "Date and time:"
date