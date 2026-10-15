#!/bin/bash
#SBATCH --job-name=merge-joint-ca-sa
#SBATCH --output=logs/merge_joint_ca_sa_%j.out
#SBATCH --error=logs/merge_joint_ca_sa_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs
package_dir="outputs/segmentation/controlled_attention_cv/joint_ca_sa_cartesian"
python scripts/select_controlled_cv.py --package-dir "$package_dir" \
  --spec configs/joint_ca_sa_cartesian_screen.json
python scripts/finalize_joint_ca_sa_cartesian.py --package-dir "$package_dir"
