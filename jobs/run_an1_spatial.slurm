#!/bin/bash
#SBATCH --job-name=an1_spatial
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/an1_spatial_%A_%a.log
#
# AN1 part 2 (protocol Section 58): tasks 0-19 = VOC-Sim classes, 20-39 = COCO-Cap
# classes. Resumable. After all tasks:
#   python scripts/narmyn/attention/an1_spatial.py --merge
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p logs
if [ "${SLURM_ARRAY_TASK_ID}" -lt 20 ]; then DATASET=voc; SHARD=${SLURM_ARRAY_TASK_ID}; else DATASET=coco; SHARD=$((SLURM_ARRAY_TASK_ID - 20)); fi
echo "node=$(hostname -s) env=${CONDA_ENV:-trdp} dataset=${DATASET} shard=${SHARD}"
python scripts/narmyn/attention/an1_spatial.py --dataset "${DATASET}" --class-shard "${SHARD}" "$@"
