#!/bin/bash
# Full GoCA class-word run (resumable). Usage: sbatch [--nodelist=renyi] goca_full.slurm <config.py> <out_dir> [extra args]
#SBATCH --job-name=n-goca-full
#SBATCH --partition=gpu
#SBATCH --exclude=wald
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/logs/%x_%j.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/logs/%x_%j.err
set -eo pipefail
export TMPDIR=/home/sidna/tmp_pip
source ~/.bashrc || true
conda activate goca || true
set -u
PY=/home/sidna/miniconda3/envs/goca/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore
N=/home/sidna/TRDP/external/n_side
CFG=$1; OUT=$2; shift 2
cd $N
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
date; $PY scripts/goca_full.py --manifest outputs/manifest_full_voc_sim.csv --manifest outputs/manifest_full_coco_cap.csv \
  --config "$CFG" --out "$OUT" --seed 0 "$@"
date
