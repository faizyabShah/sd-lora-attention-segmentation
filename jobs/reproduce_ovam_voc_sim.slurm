#!/bin/bash
#SBATCH --job-name=ovam-reproduce
#SBATCH --output=logs/ovam_reproduce_%j.out
#SBATCH --error=logs/ovam_reproduce_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail

cd /home/shafa/sd-lora-attention-segmentation

python scripts/reproduce_ovam_voc_sim.py \
    --count 5 \
    --output-dir outputs/ovam_reproduction
