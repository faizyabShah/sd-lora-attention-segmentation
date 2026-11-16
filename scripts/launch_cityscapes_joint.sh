#!/usr/bin/env bash
# Runs inside Screen; every stage continues automatically or records failure.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/brace_env.sh
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
out=outputs/segmentation/cityscapes_joint_owner
mkdir -p "$out/logs"
stage=starting
set_status() {
  python - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
p=Path('outputs/segmentation/cityscapes_joint_owner/status.json')
p.write_text(json.dumps({'status':sys.argv[1],'stage':sys.argv[2],'images_expected':65},indent=2)+'\n')
PY
}
on_exit() {
  result=$?
  if ((result)); then
    set_status failed "$stage"
    printf 'FAILED stage=%s exit=%s; see logs for full traceback\n' "$stage" "$result" > "$out/logs/FAILED.txt"
  fi
}
trap on_exit EXIT
stage=capture
set_status running "$stage"
pids=()
for shard in 0 1 2; do
  python -u scripts/run_cityscapes_joint.py capture --shard "$shard" --num-shards 3 > "$out/logs/capture_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if wait "$pid"; then :; else
    mapfile -t active_pids < <(jobs -pr)
    if ((${#active_pids[@]})); then kill "${active_pids[@]}" 2>/dev/null || true; fi
    wait || true
    exit 1
  fi
done
stage=render
set_status running "$stage"
python -u scripts/run_cityscapes_joint.py render --workers 4 > "$out/logs/render.log" 2>&1
stage=report_and_archive
set_status running "$stage"
python -u scripts/run_cityscapes_joint.py report > "$out/logs/report.log" 2>&1
# report writes the final complete status only after validation and ZIP checks.
