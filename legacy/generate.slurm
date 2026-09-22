#!/bin/bash
#SBATCH --job-name=gen-frozen
#SBATCH --output=gen_%A_%a.out
#SBATCH --error=gen_%A_%a.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7:00:00

# =========================
# SETUP
# =========================

source ~/.bashrc
conda activate trdp


# =========================
# RANK LIST
# =========================





# =========================
# RUN GENERATION
# =========================

python generate_lora_images_frozen_1.5.py
