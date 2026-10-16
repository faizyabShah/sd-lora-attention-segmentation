#!/bin/bash
#SBATCH --job-name=smoke-p3c
#SBATCH --output=logs/smoke_p3c_%j.out
#SBATCH --error=logs/smoke_p3c_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs
smoke_dir="/tmp/${USER}_phase3_confirmation_smoke_${SLURM_JOB_ID}"
python scripts/run_phase3_confirmation.py \
  --class-shard 0 --max-samples 1 --verify-reference \
  --crf-site "${CRF_SITE:-outputs/dependencies/simplecrf_py310}" \
  --output-dir "$smoke_dir"
python - "$smoke_dir" <<'PY'
import sys
from pathlib import Path
import numpy as np

paths = list((Path(sys.argv[1]) / "shards").glob("*.npz"))
assert len(paths) == 1
with np.load(paths[0], allow_pickle=False) as data:
    assert len(data["configurations"]) == 30
    assert data["counts"].shape == (30, 1, 17, 4)
print("P3-C smoke artifact verified: 30 configurations, 1 image, 17 thresholds")
PY
rm -rf "$smoke_dir"
