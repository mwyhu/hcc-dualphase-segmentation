#!/bin/bash

# load Snellius modules
module load 2024
module load Miniconda3/24.7.1-0

# activate conda environment
source activate hcc

# nnUNet paths
export nnUNet_raw=/projects/prjs2180/data/nnUNet_raw
export nnUNet_preprocessed=/projects/prjs2180/data/nnUNet_preprocessed
export nnUNet_results=/projects/prjs2180/data/nnUNet_results

# prjs paths
export PROJECT_DIR=/projects/prjs2180/code/hcc_active_learning
export PRETRAINED_MODELS=/projects/prjs2180/pretrained_models/TotalSegmentator

# TotalSegmentator model cache
export TOTALSEG_HOME=/projects/prjs2180/pretrained_models/TotalSegmentator

echo "Environment loaded:"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"
