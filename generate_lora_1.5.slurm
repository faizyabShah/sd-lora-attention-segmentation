#!/bin/bash
#SBATCH --job-name=gen-seed5-resume
#SBATCH --output=gen_seed5_%j.out
#SBATCH --error=gen_seed5_%j.err

#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7:00:00

# =========================
# SETUP
# =========================

source ~/.bashrc
conda activate trdp

echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# =========================
# SETTINGS
# =========================

RANKS=(8 16 32 64 128)
SEED=5

# =========================
# RESUME GENERATION
# =========================

for RANK in "${RANKS[@]}"
do
    echo ""
    echo "========================================"
    echo "RESUMING rank=$RANK, seed=$SEED"
    echo "========================================"

    OUTPUT_DIR="./outputs/generated/rank_${RANK}/seed_${SEED}"

    if [ -d "$OUTPUT_DIR" ]; then
        EXISTING=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.png" | wc -l)
    else
        EXISTING=0
    fi

    echo "Existing images: $EXISTING / 5000"

    if [ "$EXISTING" -ge 5000 ]; then
        echo "Already complete. Skipping rank=$RANK."
        continue
    fi

    python cityscapes_workflow.py generate \
        --rank "$RANK" \
        --seed "$SEED"

    if [ $? -ne 0 ]; then
        echo "ERROR: Generation failed for rank=$RANK seed=$SEED"
        exit 1
    fi

    FINAL_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.png" | wc -l)

    echo "Completed rank=$RANK seed=$SEED"
    echo "Final image count: $FINAL_COUNT / 5000"
done

echo ""
echo "========================================"
echo "ALL SEED 5 GENERATIONS COMPLETE"
echo "========================================"
