# Run 50 on the VPU-lab machine: iSeg's seed taken apart on 100 hard COCO photos

For the Codex on the VPU-lab machine (set up by runs 48 and 49).
- **Read first:** `runs/50_iseg_parts/PREREG.md`.
- **Status:** exploratory. Nothing is tuned or promoted, and the frozen method is not changed.
- **Pushing:** Faiz authorises pushing the result files in step 4, and nothing else.

**Not yet GPU-tested (2026-11-27).** Only the CPU parts ran on the cluster: the item step (100 photos, 364 items, 59
classes) and the prompt and caption-matching checks. So step 1 is a 2-photo smoke.
- **If a step crashes:** fix only what makes it run (shapes, paths, versions), never a method, and record every change
  in `runs/50_iseg_parts/run_notes.md`.
- **Built-in checks:** the capture asserts the recorder and prompt checks of PREREG section 3. If one fails, **stop and
  report**; do not relax it.

**2026-11-27: resume after the stop at 318 / 364 items (SD 1.5).** The prompt check was right to stop. iSeg's token
table gives "fire hydrant" 2 tokens, but CLIP's tokenizer gives 3 (fire / hydr / ant). iSeg's position arithmetic,
copied into `multi()`, therefore read every class listed after fire hydrant one token early, on "and" (photo 535981:
car, dog, fire hydrant). The fix (PREREG section 3a):
- `multi()` counts tokens with the pipeline's tokenizer;
- the same decode check now also runs on prompts S1, S2, S2n, B and F;
- `quarantine` moves (never deletes) every saved map whose prompts put fire hydrant before another class, so the
  capture redoes it. Prompt B can name fire hydrant from a caption, so earlier photos may be affected too.

Resume with:

```bash
git pull --rebase && source scripts/brace_env.sh && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python scripts/jc_iseg50.py quarantine                # prints "quarantine maps_15: k of n files moved"; note k in run_notes
python scripts/jc_iseg50.py capture --bb 15           # resumes; redoes the moved items
```

Then continue from step 2 (`capture --bb 21`) and steps 3–4 unchanged. In `run_notes.md`, record the stop, the
traceback, the quarantine line, and that the `*_tokbug` folders are kept and not committed.

## 0. Update

```bash
cd sd-lora-attention-segmentation && git pull --rebase
source scripts/brace_env.sh && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cp -rn shipped/cache/. "$BRACE_CACHE_DIR"/
ls "$BRACE_CACHE_DIR/jc_sd21_47/ov42/fit.pt" "other repositories/iseg/data/coco14/coco_name80.py"   # both from run 49
```

## 1. Items, photos, captions and a 2-photo smoke

```bash
python scripts/jc_iseg50.py items                     # CPU ~3 min; must print: 100 photos (pool positions 2000-2393 ...), 364 primary items (178 small < 5%), 59 classes
python scripts/fetch_coco_subset.py --list "$BRACE_CACHE_DIR/jc_iseg50/r0/photos.json"
python scripts/jc_iseg50.py caption                   # BLIP-large unconditional captions + prompt-B class lists (~1 min)
python scripts/jc_iseg50.py capture --bb 15 --limit 2
python scripts/jc_iseg50.py capture --bb 21 --limit 2
```

- **Item counts:** if `items` prints different counts, stop and report.
- **The smoke must print:**
  - `check: our seed from the new recorder vs make_seed, max |diff|` below 1e-4;
  - `check: iSeg aggregation from head means vs per-head, max relative diff` below 1e-4;
  - one `photo ...: N items, M passes, ... passes/s` line per photo.
- **The smoke maps are kept:** the full capture resumes from them.

## 2. Full capture (in GNU Screen; about 2,750 passes per backbone, roughly 1.5 h each on the A40)

```bash
python scripts/jc_iseg50.py capture --bb 15
python scripts/jc_iseg50.py capture --bb 21           # includes iSeg's own pass per photo
```

Both commands are resumable per photo.

## 3. Score, report and panels

```bash
python scripts/jc_iseg50.py score --workers 16        # CPU; ~280 maps x 364 items
python scripts/jc_iseg50.py report                    # -> runs/50_iseg_parts/result.md
python scripts/jc_iseg50.py viz                       # -> runs/50_iseg_parts/viz/*.jpg (10 panels)
```

## 4. Commit and push (authorised)

Write `runs/50_iseg_parts/run_notes.md`:
- GPU and timing (passes/s and total minutes per backbone);
- the two check lines of the smoke;
- the item count line and the caption summary line;
- any deviation, with every traceback and fix.

```bash
git add runs/50_iseg_parts/result.md runs/50_iseg_parts/gaps.json runs/50_iseg_parts/viz/ runs/50_iseg_parts/run_notes.md
git commit -m "runs/50: iSeg ingredients on 100 hard COCO photos (exploratory)"
git pull --rebase && git push
```

**Do not commit** maps, scores, masks, captions or anything under `$BRACE_CACHE_DIR`.
