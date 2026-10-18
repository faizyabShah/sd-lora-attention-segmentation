#!/bin/bash
#SBATCH --job-name=smoke-sg9
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$repo"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
python scripts/run_sg9_reciprocal_propagation.py --dataset voc --class-shard 0 \
  --max-samples 1 --smoke-diagnostics \
  --output-dir outputs/segmentation/controlled_attention_cv/sg9_reciprocal_propagation_smoke
