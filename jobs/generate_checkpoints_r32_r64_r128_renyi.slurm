#!/bin/bash
#SBATCH --job-name=gen-ckpt-r32-plus
#SBATCH --output=logs/gen_ckpt_r32_plus_%j.out
#SBATCH --error=logs/gen_ckpt_r32_plus_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Sequential and resumable: ranks 32/64/128 for every training seed and
# checkpoint. Each run uses captions 0,10,...,4990.
source ~/.bashrc
conda activate trdp
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"

SEEDS=(1 2 3 4 5)
RANKS=(32 64 128)
CHECKPOINTS=(1200 2400 3600)

# Validate all checkpoint weights before allocating time to generation.
for RANK in "${RANKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for CHECKPOINT in "${CHECKPOINTS[@]}"; do
            python cityscapes_workflow.py generate \
                --rank "$RANK" --seed "$SEED" --checkpoint "$CHECKPOINT" \
                --caption-stride 10 --num-images 500 --dry-run
        done
    done
done

for RANK in "${RANKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for CHECKPOINT in "${CHECKPOINTS[@]}"; do
            python cityscapes_workflow.py generate \
                --rank "$RANK" --seed "$SEED" --checkpoint "$CHECKPOINT" \
                --caption-stride 10 --num-images 500
        done
    done
done

