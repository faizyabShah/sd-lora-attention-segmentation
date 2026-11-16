#!/bin/bash
# DiffSegmenter class-word full run on SD 1.5 (caption mode none), VOC-Sim + COCO-Cap. Resumable.
#SBATCH --job-name=n-diffseg-full
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/diffseg/logs/%x_%j.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/diffseg/logs/%x_%j.err
set -eo pipefail
cd /home/sidna/TRDP/external/n_side/diffseg
PY=/home/sidna/miniconda3/envs/diffseg/bin/python
source ~/.bashrc >/dev/null 2>&1 && conda activate diffseg >/dev/null 2>&1 && PY=python || echo "conda activate failed; using $PY"
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 TMPDIR=/home/sidna/tmp_pip
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
M=/home/sidna/TRDP/external/n_side/outputs
$PY diffseg_classword.py --seed 0 --caption-mode none \
  --manifest $M/manifest_full_voc_sim.csv $M/manifest_full_coco_cap.csv \
  --out outputs/full_sd15 --label "sanity, fixed threshold 0.5 on min-max map, not official" "$@"
