#!/bin/bash
#SBATCH --job-name=tf_backlog
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# One resumable job: smoke test, 40 class shards over two GPUs, then finalize.
# Run from the repository root with:
#   sbatch jobs/run_training_free_backlog.slurm
set -eo pipefail
repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$repo"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
workers="${WORKERS:-4}"
mkdir -p logs
queue="logs/tf_backlog_queue_${SLURM_JOB_ID}.txt"
failed="logs/tf_backlog_failed_${SLURM_JOB_ID}.txt"
children=()

requeue_at_timeout() {
  for pid in "${children[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${children[@]}"; do wait "$pid" 2>/dev/null || true; done
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_at_timeout USR1

python -m src.segmentation.training_free_backlog
smoke_dir="${SLURM_TMPDIR:-/tmp}/tf_backlog_smoke_${SLURM_JOB_ID}"
CUDA_VISIBLE_DEVICES=0 python scripts/run_training_free_backlog.py \
  --dataset voc --class-shard 0 --max-samples 1 --output-dir "$smoke_dir"
rm -rf "$smoke_dir"
echo "CPU and live one-image smoke tests passed"

: > "$failed"
{ for shard in $(seq 0 19); do echo "coco $shard"; done
  for shard in $(seq 0 19); do echo "voc $shard"; done; } > "$queue"

next_task() {
  flock "$queue.lock" bash -c 'line=$(head -n 1 "$1"); [ -n "$line" ] && sed -i 1d "$1"; echo "$line"' _ "$queue"
}

worker() {
  local gpu=$1 task dataset shard
  while task=$(next_task) && [ -n "$task" ]; do
    read -r dataset shard <<< "$task"
    echo "$(date +%T) gpu=$gpu start $dataset $shard"
    if CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_training_free_backlog.py \
      --dataset "$dataset" --class-shard "$shard" \
      > "logs/tf_backlog_${dataset}_$(printf %02d "$shard").log" 2>&1; then
      echo "$(date +%T) gpu=$gpu done $dataset $shard"
    else
      echo "$dataset $shard" >> "$failed"
      echo "$(date +%T) gpu=$gpu FAILED $dataset $shard"
    fi
  done
}

echo "node=$(hostname -s) env=${CONDA_ENV:-trdp} workers=$workers"
for worker_index in $(seq 0 $((workers - 1))); do
  worker $((worker_index % 2)) &
  children+=("$!")
  sleep 25
done
for pid in "${children[@]}"; do wait "$pid"; done
children=()
if [ -s "$failed" ]; then
  echo "FAILED shards:"
  cat "$failed"
  exit 1
fi
python scripts/finalize_training_free_backlog.py
