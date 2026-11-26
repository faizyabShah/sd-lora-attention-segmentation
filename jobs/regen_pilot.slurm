#!/bin/bash
# Regeneration pilot (GPU, node wald). Usage: sbatch jobs/regen_pilot.slurm scripts/regen_pilot.py run --dataset voc|coco
#SBATCH --job-name=regen-pilot
#SBATCH --partition=gpu
#SBATCH --nodelist=wald

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd /home/shafa/sd-lora-attention-segmentation
source ~/.bashrc
conda activate trdp
set -u
export BRACE_CACHE_DIR=/home/shafa/sg_comp PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$@"
