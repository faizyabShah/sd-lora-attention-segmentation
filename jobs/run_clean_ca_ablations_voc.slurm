#!/bin/bash
#SBATCH --job-name=voc-clean-ca
#SBATCH --output=logs/voc_clean_ca_%A_%a.out
#SBATCH --error=logs/voc_clean_ca_%A_%a.err
#SBATCH --array=0-19%4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs

child_pid=""
requeue_at_timeout() {
    if [[ -n "$child_pid" ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue_at_timeout USR1

python scripts/run_clean_ca_ablations.py \
    --class-shard "$SLURM_ARRAY_TASK_ID" \
    --device cuda \
    --storage-dtype float16 \
    --model-dtype float32 \
    --threshold 0.45 \
    --normalization minmax \
    --exhaustive-batch-size 256 &
child_pid=$!
wait "$child_pid"
