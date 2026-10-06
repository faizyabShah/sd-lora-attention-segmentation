#!/bin/bash
#SBATCH --job-name=gen-ckpt-o1-r8-r16
#SBATCH --output=logs/gen_ckpt_o1_r8_r16_%j.out
#SBATCH --error=logs/gen_ckpt_o1_r8_r16_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Sequential and resumable: 5 frozen subsets, then ranks 8/16 for every
# training seed and checkpoint. Each run uses captions 1,11,...,4991.
source ~/.bashrc
conda activate trdp
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"

SEEDS=(1 2 3 4 5)
RANKS=(8 16)
CHECKPOINTS=(1200 2400 3600)

# Validate all checkpoint weights before allocating time to generation.
for RANK in "${RANKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for CHECKPOINT in "${CHECKPOINTS[@]}"; do
            python cityscapes_workflow.py generate \
                --rank "$RANK" --seed "$SEED" --checkpoint "$CHECKPOINT" \
                --caption-stride 10 --caption-offset 1 --num-images 500 \
                --subset-name every_10_offsets_0_1_count_1000 --output-index-offset 500 --dry-run
        done
    done
done

for SEED in "${SEEDS[@]}"; do
    python cityscapes_workflow.py generate --frozen --seed "$SEED" \
        --caption-stride 10 --caption-offset 1 --num-images 500 \
                --subset-name every_10_offsets_0_1_count_1000 --output-index-offset 500
done

for RANK in "${RANKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for CHECKPOINT in "${CHECKPOINTS[@]}"; do
            python cityscapes_workflow.py generate \
                --rank "$RANK" --seed "$SEED" --checkpoint "$CHECKPOINT" \
                --caption-stride 10 --caption-offset 1 --num-images 500 \
                --subset-name every_10_offsets_0_1_count_1000 --output-index-offset 500
        done
    done
done

