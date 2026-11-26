# Run 49 on the VPU-lab machine: small-set component diagnostics (ours vs DiffSegmenter vs iSeg)

For the Codex on the VPU-lab machine (set up with `runs/48_vpu_setup/RUN.md`). Read `runs/49_diag/PREREG.md` first.
- **Status:** exploratory. Nothing is tuned, nothing is promoted, and the frozen method is not changed.
- **Pushing:** Faiz authorises pushing the result files listed in step 5, and nothing else.

**Not yet GPU-tested (2026-11-26).** On the cluster, the item step ran (rounds 0 and 1: 63 / 61 items), but the GPU
smoke was still queued when this was pushed. Run round 0 first.
- **If a step crashes:** fix only what makes it run (shapes, paths, versions), never a method, and record every change
  in `run_notes.md`.
- **Then:** push `run_notes.md` and the round 0 report before starting round 1.

**Scope:** 2 rounds of 30 random COCO **train2017** photos each, with every primary item on them. Each item gets passes
on SD 1.5 and SD 2.1, plus iSeg's own pass per photo.

## 0. Update

```bash
cd sd-lora-attention-segmentation && git pull --rebase
source scripts/brace_env.sh && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cp -rn shipped/cache/. "$BRACE_CACHE_DIR"/                  # adds jc_diag49/word_map.json and fit40_photos.json
mkdir -p "other repositories"
git clone https://github.com/linsun449/iseg.code.git "other repositories/iseg" && git -C "other repositories/iseg" checkout 57e9a17
```

- **iSeg checkout:** only its class token table is read (`data/coco14/coco_name80.py`); none of its code runs.
- **`other repositories/`** is git-ignored by the parent repo. Leave it out of commits.

## 1. SD 2.1 profiles (once; about 1,600 SD 2.1 passes, 20–40 min)

```bash
python scripts/jc_sd21_47.py prof
python scripts/jc_sd21_47.py dump42          # COCO train2017 profile photos, fetched in run 48
python scripts/jc_sd21_47.py fit42           # -> $BRACE_CACHE_DIR/jc_sd21_47/ov42/fit.pt
```

- **Expected:** `fit42 (SD 2.1): 880 object pairs; 880 negative pairs` (80 words × 11 modules), give or take words with
  missing negatives.
- **Not needed:** `dump44` / `fit44` are Context words, and this run uses COCO only.
- **Resumable:** `dump42` skips finished photos if restarted.

## 2. One round (r = 0, then r = 1)

Run each in GNU Screen; the times are estimates for an A40.

```bash
R=0
python scripts/jc_diag49.py items --round $R                                    # CPU, ~2 min (reads instances_train2017.json)
python scripts/fetch_coco_subset.py --list "$BRACE_CACHE_DIR/jc_diag49/r$R/photos.json"   # the round's 30 photos
python scripts/jc_diag49.py caption --round $R                                  # BLIP-large, ~1 min
python scripts/jc_diag49.py capture --round $R --bb 15                          # SD 1.5, 3 passes per item, ~5 min
python scripts/jc_diag49.py capture --round $R --bb 21                          # SD 2.1, 3 passes per item + iSeg per photo, ~6 min
python scripts/jc_diag49.py score --round $R                                    # CPU, ~10 min
python scripts/jc_diag49.py report --rounds $R                                  # -> runs/49_diag/result_r$R.md
python scripts/jc_diag49.py viz --round $R                                      # -> runs/49_diag/viz/r${R}_*.jpg (8 panels)
```

- **Restarts:** `capture` and `score` are resumable per item and per photo.
- **Expected item counts** (from the cluster): round 0 is 63 items / 31 classes, round 1 is 61 items / 32 classes. If
  `items` prints different numbers, stop and report: it means a different annotation file or exclusion list.
- **Speed:** measure the passes per second of the first capture photo and note it in the report.

## 3. Pooled report

```bash
python scripts/jc_diag49.py report --rounds 0,1                                 # -> runs/49_diag/result_r0-1.md
```

## 4. Sanity checks (state each one in the report)

1. **Profiles:** every capture prints a `profiles:` line. For `--bb 21` it must point to `jc_sd21_47/ov42/fit.pt` with
   model `sd2-community/...` or the snapshot path, not an SD 1.5 file.
2. **Gate coverage:** `miss` stays 0 in every `maps_15` / `maps_21` file (every COCO word has a profile).
3. **Item counts:** the reference rows of section 1 of each result file exist (ours, ours on SD 2.1, DiffSegmenter,
   iSeg), each with the round's item count in section 6.
4. **Captions:** printed examples begin with "a photograph of {word}". BLIP's odd starts are kept as generated.

## 5. Commit and push (authorised)

```bash
git add runs/49_diag/result_r*.md runs/49_diag/gaps_r*.json runs/49_diag/viz/
git commit -m "runs/49: VPU diagnostics, rounds 0-1 (exploratory)"
git pull --rebase && git push
```

**Also write `runs/49_diag/run_notes.md`, and commit it with the results.** It holds:
- GPU and timing;
- the sanity checks of step 4;
- anything that deviated from this file;
- any traceback with the fix you applied. Change the driver only to make it run, never to change a method, and describe
  every such change.

**Do not commit** maps, scores, masks, captions or anything under `$BRACE_CACHE_DIR`.
