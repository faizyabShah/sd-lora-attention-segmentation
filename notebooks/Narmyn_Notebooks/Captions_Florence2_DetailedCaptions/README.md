# Florence-2 detailed captions

| File | Images | Source |
|---|---|---|
| `metadata.jsonl` | 5,000 | earlier Kaggle run, `train`/`val`/`test` only |
| `metadata_cityscapes_24995.jsonl` | 24,995 | full run incl. `train_extra`, used by `scripts/narmyn/` |

The 24,995 file is the one the rank sweep depends on. **Its line order is the
image index and the generation seed** — image `{i:05d}.png` in every generated
set corresponds to line `i`. Re-sorting, filtering or regenerating it would
silently re-pair every image already produced.

sha256 prefix: `eae76280f23a8d00`
