#!/bin/bash
#SBATCH --job-name=lora-train
#SBATCH --output=train_%A_%a.out
#SBATCH --error=train_%A_%a.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00

#SBATCH --array=0-24

# =========================
# SETUP
# =========================

source ~/.bashrc
conda activate trdp

echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# =========================
# PARAMETER SWEEP
# =========================

RANKS=(8 16 32 64 128)
SEEDS=(1 2 3 4 5)

NUM_SEEDS=${#SEEDS[@]}

RANK_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

RANK=${RANKS[$RANK_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running rank=$RANK seed=$SEED"

# =========================
# TRAIN
# =========================

python cityscapes_workflow.py train \
    --rank $RANK \
    --seed $SEED

echo "Finished rank=$RANK seed=$SEED"
