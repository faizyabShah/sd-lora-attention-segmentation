#!/bin/bash
# Submit the registered Section 68 smoke, VOC selection, COCO transfer and merge.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
mkdir -p logs
smoke=$(sbatch --parsable -o logs/sg1_sg3_smoke_%j.out -e logs/sg1_sg3_smoke_%j.err \
  jobs/smoke_sg1_sg3_graph_refinement.slurm)
voc=$(sbatch --parsable --dependency="afterok:$smoke" \
  -o logs/sg1_sg3_voc_%A_%a.out -e logs/sg1_sg3_voc_%A_%a.err \
  jobs/run_sg1_sg3_graph_refinement.slurm voc)
coco=$(sbatch --parsable --dependency="afterok:$smoke" \
  -o logs/sg1_sg3_coco_%A_%a.out -e logs/sg1_sg3_coco_%A_%a.err \
  jobs/run_sg1_sg3_graph_refinement.slurm coco)
final=$(sbatch --parsable --dependency="afterok:$voc:$coco" \
  -o logs/sg1_sg3_final_%j.out -e logs/sg1_sg3_final_%j.err \
  jobs/finalize_sg1_sg3_graph_refinement.slurm)
echo "smoke=$smoke voc=$voc coco=$coco finalize=$final"
