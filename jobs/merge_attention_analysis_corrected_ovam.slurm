#!/bin/bash
#SBATCH --job-name=voc-attn-v2-report
#SBATCH --output=logs/voc_attn_v2_report_%j.out
#SBATCH --error=logs/voc_attn_v2_report_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/evaluate_released_ovam_masks.py
python scripts/evaluate_released_ovam_heatmaps.py
python scripts/merge_attention_analysis.py \
    --input-dir outputs/analysis/voc_sim_attention_report_corrected_ovam \
    --expected-shards 20 \
    --expected-images 490 \
    --report attention_analysis.md
