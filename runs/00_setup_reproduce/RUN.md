# Run 00: reproduce frozen BRACE on the A40

**Goal.** Before any new experiment, confirm that this machine reproduces the paper's BRACE numbers on the generated sets:
- VOC-Sim: 0.830
- COCO-Cap: 0.696

**Do not change code** under `src/` or `scripts/`. If something fails, stop and report the full error.

**Faiz authorises pushing the result files of this run** (an exception to AGENTS.md's "push only when asked").

## 0. Preconditions

Follow `docs/NEW_MACHINE_SETUP.md`. Then make sure:
1. `.venv` is set up, SimpleCRF is installed, and the SD1.5 weights are downloaded. CLIP isn't needed for this run.
2. The OVAM dataset is extracted:
   `datasets/ovam_experiment_with_dataset/evaluation/ovam/{voc_sim,coco_captions}`
3. The setup check passes. GPU commands need host access (`require_escalated`):

```bash
source scripts/brace_env.sh
export HF_HUB_OFFLINE=1
python scripts/check_brace_setup.py
```

## 1. Smoke test (about 2 min)

```bash
python scripts/brace_repro.py capture --dataset voc --max-images 2
python scripts/brace_repro.py capture --dataset coco --max-images 2
```

Each image prints a line like `[brace:voc] ovam-voc-sim-0 aeroplane iou 0.9xx 12.3s`. Note the GPU memory per process (`nvidia-smi`) while it runs.

## 2. Full run, in a Screen session (about 1–2 h)

Run 3 shards per set in parallel. If one process uses under 12 GB, all six fit on the A40 together. Otherwise run the three VOC shards first, then the three COCO shards.

```bash
mkdir -p outputs/logs
screen -dmS brace_run00 bash -lc '
source scripts/brace_env.sh; export HF_HUB_OFFLINE=1
for ds in voc coco; do for s in 0 1 2; do
  python scripts/brace_repro.py capture --dataset $ds --shard $s --num-shards 3 > outputs/logs/brace_${ds}_${s}.log 2>&1 &
done; done; wait'
```

Progress:

```bash
ls $BRACE_CACHE_DIR/brace_voc | wc -l
ls $BRACE_CACHE_DIR/brace_coco | wc -l
```

Wait for 490 and 707 respectively. Finished images are skipped, so after a crash you can rerun the same command.

## 3. Report

```bash
python scripts/brace_repro.py report
```

This writes `runs/00_setup_reproduce/result.md` and `result.json`.

**Sanity check.** The report prints `PASS` when both sets are within 0.005 of the paper. Per-image differences against the cluster reference (`reference_cluster.json`) are expected, because of cross-GPU nondeterminism; the set-level mIoU should still match. If it prints `FAIL`, report it and do not start other runs.

## 4. Commit and push the results only

Never commit `$BRACE_CACHE_DIR`.

```bash
mkdir -p runs/00_setup_reproduce/logs
cp outputs/logs/brace_*.log runs/00_setup_reproduce/logs/
git add runs/00_setup_reproduce/result.md runs/00_setup_reproduce/result.json runs/00_setup_reproduce/logs
git commit -m "Run 00: BRACE reproduction on A40"
git pull --rebase
git push
```

**Report back to Faiz:**
- the `result.md` table and PASS/FAIL;
- the GPU memory per process;
- the seconds per image.
