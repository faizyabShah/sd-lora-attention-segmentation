#!/bin/bash
#SBATCH --job-name=b2_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/b2_smoke_%j.log
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs
python scripts/optimize_b2_tokens.py --track b2a --task-id 0 --epochs 2 \
  --output-dir outputs/segmentation/controlled_attention_cv/token_learning_b2/smoke

