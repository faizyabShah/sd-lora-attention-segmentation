#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p logs

# This is cheap validation on the login node. It also guarantees that every
# evaluated annotation has matching caption metadata and a released image.
python scripts/build_coco_cap_manifest.py

coco_job="$(sbatch --parsable jobs/run_coco_cap_generalization.slurm)"
merge_job="$(sbatch --parsable --dependency="afterok:${coco_job}" jobs/merge_coco_cap_generalization.slurm)"
visual_job="$(sbatch --parsable jobs/run_voc_progression_visuals.slurm)"

echo "COCO-Cap evaluation array: ${coco_job}"
echo "COCO-Cap merge (after evaluation): ${merge_job}"
echo "VOC-Sim progression visuals: ${visual_job}"
echo "Check: squeue -j ${coco_job},${merge_job},${visual_job}"

