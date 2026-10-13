#!/bin/bash
#SBATCH --job-name=smoke-voc-ca-tr
#SBATCH --output=logs/smoke_voc_ca_tr_%j.out
#SBATCH --error=logs/smoke_voc_ca_tr_%j.err
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

smoke_dir="/tmp/${USER}_controlled_ca_time_resolution_smoke_${SLURM_JOB_ID}"
python scripts/run_controlled_ca_time_resolution.py \
    --class-shard 0 \
    --max-samples 1 \
    --output-dir "$smoke_dir" \
    --device cuda \
    --storage-dtype float16 \
    --model-dtype float32
test -f "$smoke_dir/shards/class_00_aeroplane.npz"
rm -rf "$smoke_dir"
