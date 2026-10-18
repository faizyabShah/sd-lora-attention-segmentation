#!/bin/bash
#SBATCH --job-name=smoke-sg1-sg3
#SBATCH --partition=gpu
#SBATCH --nodelist=erdos,neumann
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$repo"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
python scripts/run_sg1_sg3_graph_refinement.py --dataset voc --class-shard 0 \
  --max-samples 1 \
  --output-dir outputs/segmentation/controlled_attention_cv/sg1_sg3_graph_refinement_smoke
