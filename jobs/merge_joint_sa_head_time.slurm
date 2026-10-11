#!/bin/bash
#SBATCH --job-name=voc-joint-merge
#SBATCH --output=logs/voc_joint_merge_%j.out
#SBATCH --error=logs/voc_joint_merge_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/merge_joint_sa_head_time.py \
    --input-dir outputs/segmentation/voc_sim_full_joint_sa_head_time \
    --expected-shards 20
