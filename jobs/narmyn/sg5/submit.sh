#!/bin/bash
# SG5 (protocol Section 66): VOC-Sim and COCO-Cap arrays in parallel, then finalize.
# Usage, from the repository root:
#   CRF_SITE=/path/to/simplecrf LOG_DIR=/path/to/logs jobs/narmyn/sg5/submit.sh
set -euo pipefail
: "${CRF_SITE:?set CRF_SITE to a SimpleCRF denseCRF build for this Python}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
voc=$(sbatch --parsable --export=ALL -o "$LOG_DIR/sg5_voc_%A_%a.log" jobs/narmyn/sg5/run_voc.slurm)
coco=$(sbatch --parsable --export=ALL -o "$LOG_DIR/sg5_coco_%A_%a.log" jobs/narmyn/sg5/run_coco.slurm)
final=$(sbatch --parsable --export=ALL -o "$LOG_DIR/sg5_final_%j.log" --dependency=afterok:$voc:$coco jobs/narmyn/sg5/finalize.slurm)
echo "voc=$voc coco=$coco finalize=$final"
