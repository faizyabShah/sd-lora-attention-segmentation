# Run 51 on the VPU-lab machine: gate-guided refinements + run 50's leads on random photos

For the Codex on the VPU-lab machine (set up by runs 48–50).
- **Read first:** `runs/51_gate_refine/PREREG.md`.
- **Status:** exploratory. Nothing is tuned or promoted, and the frozen method is not changed.
- **Pushing:** Faiz authorises pushing the result files in step 4, and nothing else.

**Not yet GPU-tested (2026-11-27).** Only these parts ran on the cluster (CPU): the item step, and the new refinement
functions on synthetic tensors (gated propagation with g = 1 equals V0 exactly; region blend with singleton regions
equals the seed; k-means separates a two-texture field).
- **If a step crashes:** fix only what makes it run (shapes, paths, versions), never a method, and record every change
  in `runs/51_gate_refine/run_notes.md`.
- **The `check` step asserts PREREG section 3's checks.** If one fails, **stop and report**; do not relax it.
- **Needs run 50's cache:** `$BRACE_CACHE_DIR/jc_iseg50/r0/items.json`, its masks and its maps. Run 51 writes to
  `$BRACE_CACHE_DIR/jc_gate51` and never modifies run 50's files.

**2026-11-27: fix for the stop at `check --bb 15` (KeyError: 64).** The check read the 64 × 64 self-attention after
`item()` had cleared the recorder. It now redoes the T pass before reading it. Only `check` changed: the capture reads
the attention before clearing it, so its maps are not affected. Pull, then resume at `check --bb 15`; `items` and
`caption` need not be rerun. Record the stop and the traceback in `run_notes.md`.

**2026-11-27, second fix: the stop at `check --bb 21` (8-head shape error).** SD 2.1's setup replaces the attention
helper with its 5-head version when the model loads. The check imported the helper before loading, so it kept the
8-head one. It now imports after loading, as the capture already does, so the capture is unaffected. Pull, then
resume at `check --bb 21` (`check --bb 15` passed and need not be rerun). Record this stop in `run_notes.md`.

## 0. Update

```bash
cd sd-lora-attention-segmentation && git pull --rebase
source scripts/brace_env.sh && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
ls "$BRACE_CACHE_DIR/jc_iseg50/r0/items.json" "$BRACE_CACHE_DIR/jc_iseg50/r0/maps_15" "$BRACE_CACHE_DIR/jc_sd21_47/ov42/fit.pt" >/dev/null && echo ok
```

## 1. Items, photos, captions, checks and a smoke

```bash
python scripts/jc_gate51.py items        # CPU ~3 min; must print exactly the three lines below
python scripts/fetch_coco_subset.py --list "$BRACE_CACHE_DIR/jc_gate51/r0/photos.json"
python scripts/jc_gate51.py caption      # BLIP-large captions of all 200 photos + prompt-B lists (~2 min)
python scripts/jc_gate51.py check --bb 15
python scripts/jc_gate51.py check --bb 21
python scripts/jc_gate51.py capture --bb 15 --limit 1     # 1 photo per half
python scripts/jc_gate51.py capture --bb 21 --limit 1
python scripts/jc_gate51.py score --limit 1 --tag _smoke
python scripts/jc_gate51.py report --tag _smoke --boot 200
```

- **`items` must print:**
  ```
  run 51 hard: 100 photos, 364 primary items (178 small < 5%), 59 classes
  run 51 random: 100 photos, 221 primary items (88 small < 5%), 60 classes
  random half: pool positions 1000-1100
  ```
  If it prints anything else, stop and report.
- **Each `check` must print:**
  - `A with g = 1 vs V0` and `C with singleton regions` lines;
  - two `vs run 50's saved map` lines with |diff| < 2e-3, for the hard photo.
- **The smoke report** must write `runs/51_gate_refine/result_smoke.md`. Do not commit it.
- **The smoke maps are kept:** the full capture resumes from them.

## 2. Full capture (GNU Screen)

```bash
python scripts/jc_gate51.py capture --bb 15     # 6 prompts x 585 items, ~3,500 passes, roughly 60 min on the A40
python scripts/jc_gate51.py capture --bb 21     # 2 prompts x 585 items + iSeg's pass per photo, ~1,400 passes, ~25 min
```

Both commands are resumable per photo.

## 3. Score, report and panels

```bash
python scripts/jc_gate51.py score --workers 16     # CPU
python scripts/jc_gate51.py report                 # -> runs/51_gate_refine/result.md
python scripts/jc_gate51.py viz                    # -> runs/51_gate_refine/viz/*.jpg (10 panels)
```

## 4. Commit and push (authorised)

Write `runs/51_gate_refine/run_notes.md`:
- GPU and timing (passes/s and total minutes per backbone);
- the `items` lines, the caption summary line, and every `check` line;
- any deviation, with every traceback and fix.

```bash
git add runs/51_gate_refine/result.md runs/51_gate_refine/panels.json runs/51_gate_refine/viz/ runs/51_gate_refine/run_notes.md
git commit -m "runs/51: gate-guided refinements and run 50 leads on random photos (exploratory)"
git pull --rebase && git push
```

**Do not commit** maps, scores, masks, captions, smoke outputs, or anything under `$BRACE_CACHE_DIR`.
