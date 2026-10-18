#!/bin/bash
#SBATCH --job-name=sg1-sg3
#SBATCH --partition=gpu
#SBATCH --nodelist=erdos,neumann
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-19%8
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
set -euo pipefail
dataset="${1:?usage: sbatch ... jobs/run_sg1_sg3_graph_refinement.slurm voc|coco}"
if [[ "$dataset" != voc && "$dataset" != coco ]]; then
  echo "dataset must be voc or coco" >&2; exit 2
fi
repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$repo"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
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
python scripts/run_sg1_sg3_graph_refinement.py \
  --dataset "$dataset" --class-shard "$SLURM_ARRAY_TASK_ID" &
child_pid=$!
wait "$child_pid"
