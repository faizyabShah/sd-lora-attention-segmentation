#!/bin/bash
# NERVE SD 1.5 full run, variants A,V,W, VOC-Sim + COCO-Cap manifests. Resumable (rerun the same command).
#SBATCH --job-name=n-nerve-full
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/nerve/logs/%x_%j.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/nerve/logs/%x_%j.err
set -eo pipefail
export TMPDIR=/home/sidna/tmp_pip PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
PY=/home/sidna/miniconda3/envs/nerve/bin/python
M=/home/sidna/TRDP/external/n_side/outputs
cd /home/sidna/TRDP/external/n_side/nerve
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
md5sum nerve_classword.py $M/manifest_full_voc_sim.csv $M/manifest_full_coco_cap.csv
$PY -u nerve_classword.py --variants A,V,W \
  --manifest $M/manifest_full_voc_sim.csv,$M/manifest_full_coco_cap.csv \
  --out /home/sidna/TRDP/external/n_side/nerve/outputs/full_sd15 \
  --label "sanity, fixed threshold, not a result" --summary-name sanity_summary.json
