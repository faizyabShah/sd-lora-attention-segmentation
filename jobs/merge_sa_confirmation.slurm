#!/bin/bash
#SBATCH --job-name=merge-voc-sa-c
#SBATCH --output=logs/merge_voc_sa_c_%j.out
#SBATCH --error=logs/merge_voc_sa_c_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs
package_dir="outputs/segmentation/controlled_attention_cv/sa_confirmation"
python scripts/select_controlled_cv.py --package-dir "$package_dir" \
  --spec configs/controlled_sa_confirmation_screens.json
python scripts/finalize_sa_confirmation.py --package-dir "$package_dir"
