#!/bin/bash
#SBATCH --job-name=ckptsweep
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/narmyn_ckptsweep_%x_%j.log
#
# Generate the SAME strided prompt sample under every checkpoint of one rank,
# so the checkpoints can be compared quantitatively (KID/FID) rather than by
# eye. One job per rank; each checkpoint writes to
#   output/lora_images/rank{RANK}_ckpt{STEP}/
# so nothing already generated from the final weights is touched.
#
#   sbatch --job-name=sweep_r8 --export=ALL,RANK=8,STRIDE=50 \
#          jobs/narmyn/checkpoint_sweep.slurm
#
# STRIDE 50 over 24,995 prompts = 500 images per checkpoint, spread across
# test/train/train_extra. Same indices for every rank and checkpoint, so the
# comparison is paired. Resume-safe: rerunning skips finished images.
#
# On erdos/neumann add CONDA_ENV=gen (their drivers cannot run trdp's cu130).

set -eo pipefail

RANK="${RANK:-8}"
STRIDE="${STRIDE:-50}"
CONDA_ENV="${CONDA_ENV:-trdp}"
STEPS="${STEPS:-3000 6000 9000 12000 15000 18000 21000 24000 27000 30000}"

source ~/.bashrc
conda activate "${CONDA_ENV}"

REPO="${REPO:-$HOME/sd-lora-attention-segmentation}"
# Data/outputs tree (cropped images, captions, weights, generations).
export TRDP_ROOT="${TRDP_ROOT:-$HOME/TRDP}"
cd "$REPO"
mkdir -p logs "$TRDP_ROOT/output/logs"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== rank=${RANK} stride=${STRIDE} env=${CONDA_ENV} node=$(hostname -s) ==="
echo "=== checkpoints: ${STEPS} ==="

for S in ${STEPS}; do
  echo ""
  echo "########## rank ${RANK} checkpoint ${S} ##########"
  python scripts/narmyn/06_generate_lora.py \
    --rank "${RANK}" --checkpoint "${S}" --stride "${STRIDE}" "$@"
done

echo ""
echo "=== sweep complete for rank ${RANK} ==="
