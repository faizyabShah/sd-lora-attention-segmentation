#!/bin/bash
#SBATCH --job-name=merge-voc-ca-tr
#SBATCH --output=logs/merge_voc_ca_tr_%j.out
#SBATCH --error=logs/merge_voc_ca_tr_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs

package_dir="outputs/segmentation/controlled_attention_cv/ca_time_resolution"
python scripts/select_controlled_cv.py \
    --package-dir "$package_dir" \
    --spec configs/controlled_ca_time_resolution_screens.json
python scripts/finalize_controlled_ca_time_resolution.py --package-dir "$package_dir"
