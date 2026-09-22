#!/bin/bash
#SBATCH --job-name=lora-train
#SBATCH --output=train_%A_%a.out
#SBATCH --error=train_%A_%a.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00

#SBATCH --array=0-4   # 5 jobs → 8,16,32,64,128

# =========================
# SETUP
# =========================

source ~/.bashrc
conda activate trdp

echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# =========================
# RANK LIST
# =========================

RANKS=(8 16 32 64 128)
RANK=${RANKS[$SLURM_ARRAY_TASK_ID]}

echo "Running training for rank: $RANK"

# =========================
# RUN TRAINING
# =========================

python train_lora_1.5.py --rank $RANK

echo "Done rank: $RANK"