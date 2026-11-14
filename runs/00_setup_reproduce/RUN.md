# Run 00: set up the machine and reproduce frozen BRACE

**Goal.** Confirm that this machine (A40 46 GB, no SLURM) reproduces the paper's BRACE numbers on the generated sets before running any new experiment:
- VOC-Sim: 0.830
- COCO-Cap: 0.696

**Do not change any code** under `src/` or `scripts/`. If something fails, stop and report the full error.

## 0. Setup (once)

Follow `SETUP_NEW_MACHINE.md`, sections 1–4:
1. **Conda env `trdp`** with the pinned versions.
2. **SimpleCRF** installed into `outputs/dependencies/simplecrf_py310`.
3. **HF weights** for `stable-diffusion-v1-5/stable-diffusion-v1-5`. CLIP isn't needed for this run.
4. **The OVAM dataset zip,** unzipped so that this path exists:
   `datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv`

Then, from the repo root:

```bash
conda activate trdp
export HF_HUB_OFFLINE=1
export SG_CACHE=$HOME/sg_cache
python scripts/brace_repro.py check
```

Every line must say `OK` and the last line `ALL OK`. If not, fix the setup (not the code) and rerun.

## 1. Smoke test (about 2 min)

```bash
python scripts/brace_repro.py capture --dataset voc --max-images 2
python scripts/brace_repro.py capture --dataset coco --max-images 2
```

Each image prints a line like `[brace:voc] ovam-voc-sim-0 aeroplane iou 0.9xx 12.3s`. Check `nvidia-smi` while it runs and note the memory used per process.

## 2. Full run (about 1–2 h)

Run 3 shards per set as background processes. If one process uses under 12 GB, all six fit on the A40 together. Otherwise run VOC-Sim first, then COCO-Cap.

```bash
mkdir -p logs
for ds in voc coco; do for s in 0 1 2; do
  nohup python scripts/brace_repro.py capture --dataset $ds --shard $s --num-shards 3 > logs/brace_${ds}_${s}.log 2>&1 &
done; done
```

Progress:

```bash
ls $SG_CACHE/brace_voc | wc -l
ls $SG_CACHE/brace_coco | wc -l
```

Wait for 490 and 707 respectively. The capture skips finished images, so if anything crashes, rerun the same command.

## 3. Report

```bash
python scripts/brace_repro.py report
```

This writes `runs/00_setup_reproduce/result.md` and `result.json`.

**Sanity check.** The report prints `PASS` when both sets are within 0.005 of the paper. Small per-image differences against the cluster reference are expected (GPU nondeterminism); the set-level mIoU should still match.

## 4. Commit back

Commit **only** these:
- `runs/00_setup_reproduce/result.md`
- `runs/00_setup_reproduce/result.json`
- the six logs, copied to `runs/00_setup_reproduce/logs/`

Never commit `$SG_CACHE`.

```bash
mkdir -p runs/00_setup_reproduce/logs
cp logs/brace_*.log runs/00_setup_reproduce/logs/
git add runs/00_setup_reproduce/result.md runs/00_setup_reproduce/result.json runs/00_setup_reproduce/logs
git commit -m "Run 00: BRACE reproduction on A40"
git pull --rebase
git push
```

**Report back to Faiz:**
- the `result.md` table and PASS/FAIL;
- the GPU memory per process;
- the seconds per image;
- the absolute path of the repo and of `$SG_CACHE` on this machine.
