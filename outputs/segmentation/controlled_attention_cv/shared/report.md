# Shared CV infrastructure

Protocol v1.2, Section 9 execution step 2 (owner: Claude). Codex integrates
this interface before CA package A (step 3).

## What is provided

| File | Purpose |
|---|---|
| `configs/voc_sim_folds.csv` | 490 annotated VOC-Sim images in 5 class-stratified folds: 98 images per fold, 4–5 images per class per fold, seed 20261013 |
| `scripts/make_voc_sim_folds.py` | rebuilds the fold file from the manifest; `--check` verifies the committed file |
| `src/segmentation/controlled_cv.py` | scorer, shard writer and loader, metrics, decision rule, cross-validated selection, paired bootstrap |
| `scripts/select_controlled_cv.py` | turns a package's shards plus a screen spec into `fold_results.json`, `summary.json`, `report.md` |
| `tests/test_controlled_cv.py` | synthetic test (18 tests): parity with `attention_to_mask`, `normalize_map`, `BinarySegmentationMetrics` and `attention_analysis` AP; writer/loader guards; decision-rule bands; CV procedure |

## How a runner uses it

```python
from src.segmentation import controlled_cv as cv

folds = cv.read_folds(Path("configs/voc_sim_folds.csv"))
names = [...]                      # fixed, ordered configuration names, e.g. "heads/0_1_2"
writer = cv.ShardWriter(shard_dir / f"class_{shard:02d}.npz", names, folds,
                        settings={"package": "...", "fixed": {...}, "varied": "..."})

for row in rows:
    if row["image_id"] in writer.completed:          # resumable
        continue
    sample = generate_attention_sample(...)
    maps = torch.stack([...])                        # [C, 64, 64] raw maps, same order as names
    counts, ap = cv.score_maps(maps, sample.target_mask, normalization="minmax")
    writer.add(row["image_id"], row["classname"], counts, ap)
writer.finish()                                      # marks the shard complete
```

- `score_maps` normalizes each 64×64 map, resizes bilinearly to the
  annotation and applies `>= t` for t in 0.35, 0.45, 0.55 — the
  `attention_to_mask` rule, verified count-for-count — and returns the
  threshold-free 64×64 AP. CPU and GPU give identical counts.
- For binary masks that have no threshold (e.g. after DenseCRF), use
  `cv.score_masks(masks, target)` and pass `ap=None`.
- Settings must match when a shard is resumed. The Git commit is recorded
  per session in the shard, so a documentation-only pull does not block a
  resume.

Selection after all shards finish:

```bash
python scripts/select_controlled_cv.py --package-dir <output root> --spec <spec.json>
```

with a spec such as
`{"screens": [{"name": "CA-H", "reference": "heads/all", "prefix": "heads/", "inclusiveness": {"heads/0_1": 2}}]}`.

## One property of the Section 5a rule for both agents to note

`tests/test_controlled_cv.py::test_gain_in_one_fold_only_is_frozen_but_oof_shows_no_gain`
shows that a gain confined to one fold is chosen by the four training sets
that contain that fold, so the 4-of-5 vote freezes it even though every
held-out fold scores exactly like the reference. The selector therefore also
reports `oof_gain` and `oof_confirms_frozen`, and `report.md` marks a frozen
change that the held-out folds do not confirm. Protocol Section 16 proposes
requiring `oof_confirms_frozen` before freezing; until Codex agrees, the rule
is applied as written and the flag is reported.
