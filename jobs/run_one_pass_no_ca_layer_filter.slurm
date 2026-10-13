#!/bin/bash
#SBATCH --job-name=onepass-allca
#SBATCH --output=logs/onepass_allca_%A_%a.out
#SBATCH --error=logs/onepass_allca_%A_%a.err
#SBATCH --array=0-39%4
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

if (( SLURM_ARRAY_TASK_ID < 20 )); then
    dataset="voc_sim"
    class_shard="$SLURM_ARRAY_TASK_ID"
else
    dataset="coco_cap"
    class_shard="$((SLURM_ARRAY_TASK_ID - 20))"
fi

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

python scripts/run_one_pass_no_ca_layer_filter.py \
    --dataset "$dataset" \
    --class-shard "$class_shard" \
    --device cuda \
    --storage-dtype float16 \
    --model-dtype float32 &
child_pid=$!
wait "$child_pid"
