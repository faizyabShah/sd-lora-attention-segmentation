#!/bin/bash
#SBATCH --job-name=smoke-sg8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -eo pipefail
repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$repo"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
python scripts/run_sg8_observation_consensus.py --dataset voc --class-shard 0 \
  --max-samples 3 --save-diagnostic-panels \
  --output-dir outputs/segmentation/controlled_attention_cv/sg8_observation_consensus_smoke
