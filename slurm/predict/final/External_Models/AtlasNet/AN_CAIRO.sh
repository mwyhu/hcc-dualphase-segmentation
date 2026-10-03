#!/bin/bash
#SBATCH --job-name=AtlasNet_CAIRO
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=25:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /projects/prjs2180/code/hcc-dualphase-segmentation
source setup_env.sh

MODEL_DIR=/projects/prjs2180/code/AbdomentAtlasNet
INPUT_DIR=/projects/prjs2180/data/nnUNet_raw/0_External_Validation/SP/CAIRO5/imagesTs
OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/CAIRO5/AtlasNet
BINARY_OUTPUT_DIR=/projects/prjs2180/evaluation/predictions/CAIRO5/AN

mkdir -p "$OUTPUT_DIR" "$BINARY_OUTPUT_DIR"

shopt -s nullglob
images=("$INPUT_DIR"/*_0000.nii.gz)

if ((${#images[@]} == 0)); then
    echo "No *_0000.nii.gz images found in $INPUT_DIR" >&2
    exit 1
fi

for required_file in \
    "$MODEL_DIR/dataset.json" \
    "$MODEL_DIR/plans.json" \
    "$MODEL_DIR/fold_all/checkpoint_final.pth"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing model file: $required_file" >&2
        exit 1
    fi
done

echo "Predicting ${#images[@]} scans with AtlasNet"
echo "Output directory: $OUTPUT_DIR"

nnUNetv2_predict_from_modelfolder \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -m "$MODEL_DIR" \
    -f all \
    -chk checkpoint_final.pth \
    -device cuda

echo "AtlasNet prediction completed"


echo "Extracting binary liver-tumour masks"
python - "$OUTPUT_DIR" "$BINARY_OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(input_dir.glob("*.nii.gz"))
if not files:
    raise SystemExit(f"No predictions found in {input_dir}")

for path in files:
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)

    # Merge labels 23–31 into one binary mask.
    mask = np.isin(data, range(23, 32)).astype(np.uint8)

    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)

    output = image.__class__(mask, image.affine, header)

    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    output.set_qform(qform, int(qcode))
    output.set_sform(sform, int(scode))

    nib.save(output, output_dir / path.name)
    print(f"Saved: {path.name}")

print(f"Done. Binary liver-tumour masks saved to {output_dir}")
PY

echo "AtlasNet prediction and binary mask extraction completed"