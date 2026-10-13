#!/bin/bash
#SBATCH --job-name=voc-progression-revised
#SBATCH --output=logs/voc_progression_revised_%j.out
#SBATCH --error=logs/voc_progression_revised_%j.err
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

output_dir="docs/attention_segmentation_report/figures/progression_revised_25"
mkdir -p "$output_dir"

python scripts/run_voc_progression_visuals.py \
    --count 25 \
    --selection-seed 20261013 \
    --exclude-selection outputs/analysis/voc_sim_progression_25/selection.json \
    --device cuda \
    --storage-dtype float16 \
    --model-dtype float32 \
    --output-dir "$output_dir" \
    --tex-fragment "$output_dir/gallery.tex" &
child_pid=$!
wait "$child_pid"
