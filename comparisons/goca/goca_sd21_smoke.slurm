#!/bin/bash
# SD 2.1-base smoke (mirror weights) + SD 1.5 regression of the edited adapter on the same 7 items.
#SBATCH --job-name=n-goca21-smoke
#SBATCH --partition=gpu
#SBATCH --exclude=wald
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/logs/%x_%j.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/logs/%x_%j.err
set -eo pipefail
PY=/home/sidna/miniconda3/envs/goca/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore
N=/home/sidna/TRDP/external/n_side
SNAP=/home/sidna/.cache/huggingface/hub/models--sd2-community--stable-diffusion-2-1-base/snapshots/4e63672c03103b6c636b8fb4119ba982469b2955
cd $N
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY scripts/goca_full.py --manifest outputs/manifest_smoke_sd21.csv --config configs/config_model_21.py \
  --model-path "$SNAP" --out outputs/goca_sd21_smoke --seed 0
$PY scripts/goca_full.py --manifest outputs/manifest_smoke_sd21.csv --out outputs/goca_sd15_regress --seed 0
