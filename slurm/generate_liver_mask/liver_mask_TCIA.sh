#!/bin/bash
#SBATCH --job-name=livermask_TCIA
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname)"
echo "Date: $(date)"

cd /projects/prjs2180/code/hcc-dualphase-segmentation

source setup_env.sh

export nnUNet_n_proc_DA=8

python -c "import torch; print(torch.cuda.is_available())"


PYTHONPATH=/projects/prjs2180/code/hcc-dualphase-segmentation/scripts/postprocessing \
python /projects/prjs2180/code/hcc-dualphase-segmentation/scripts/postprocessing/generate_liver_masks.py \
    --input_dir /projects/prjs2180/data/nnUNet_raw/TCIA_CRLM/CT1 \
    --output_dir /projects/prjs2180/liver_masks/TCIA_CRLM


echo "Finished at $(date)"
