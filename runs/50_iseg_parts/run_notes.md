# Run 50 execution notes

GPU: NVIDIA A40, 46068 MiB, 520.61.05. The experiment ran offline in GNU Screen.

- run 50: 100 photos (pool positions 2000-2393; 131 passed the fast filter), 364 primary items (178 small < 5%), 59 classes
- captions: 100 photos; prompt-B lists: other primary classes named 0.42 on average, 0.25 named classes not among the photo's primary classes per item
- Both backbones passed the original two-photo smoke before the token-count correction.
- Validated 364 items for every one of the 276 maps, with zero missing profiles in either backbone or the iSeg pass.

## SD 1.5

- profiles: /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_ov42/fit.pt (model SD 1.5)
- 38.4 minutes in the per-photo logs over 100 visits to 100 photos (driver-reported 1.19 passes/s, including repeated visits).
- check: our seed from the new recorder vs make_seed, max |diff| 0.00e+00
- check: iSeg aggregation from head means vs per-head, max relative diff 4.01e-07; prompt 'a photograph of people and dog and sofa and tv monitor and book and other object and backgroundground, land, grass, tree, building, wall, sky, lake, water, river, sea, railway, '

Smoke photo timings:

- [iseg50:15] photo 8781: 5 items, 37 passes, 39.6s (0.93 passes/s), miss 0
- [iseg50:15] photo 13146: 3 items, 23 passes, 19.3s (1.19 passes/s), miss 0

## SD 2.1-base

- profiles: /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_sd21_47/ov42/fit.pt (model sd2-community/stable-diffusion-2-1-base)
- 39.4 minutes in the per-photo logs over 100 visits to 100 photos (driver-reported 1.21 passes/s, including repeated visits).
- check: our seed from the new recorder vs make_seed, max |diff| 0.00e+00
- check: iSeg aggregation from head means vs per-head, max relative diff 2.11e-07; prompt 'a photograph of people and dog and sofa and tv monitor and book and other object and backgroundground, land, grass, tree, building, wall, sky, lake, water, river, sea, railway, '

Smoke photo timings:

- [iseg50:21] photo 8781: 5 items, 38 passes, 39.5s (0.96 passes/s), miss 0
- [iseg50:21] photo 13146: 3 items, 24 passes, 19.0s (1.26 passes/s), miss 0

## November 27 stop and restart

SD 1.5 stopped at 318 of 364 items, at photo 535981. The released table counted “fire hydrant” as two tokens while CLIP used three, shifting the selected token for dog to “and”.
Resumed with amendment 3a at commit `513bdebe`. The amended driver checks tokenizer-derived positions on our-pass prompts. iSeg's released pass retains its original behavior.
The existing items, captions, profiles, and unaffected maps were reused. Quarantined maps remain in the `*_tokbug` folders outside Git.

- `quarantine maps_15: 0 of 318 files moved to maps_15_tokbug`
- `quarantine maps_21: 0 of 8 files moved to maps_21_tokbug`

Timing uses the logged photo durations. Driver pass counts for a partly cached photo describe the full photo workload.

```text
Traceback (most recent call last):
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_iseg50.py", line 633, in <module>
    {"items": items, "caption": caption, "capture": capture, "score": score, "report": report, "viz": viz}[a.cmd](a)
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_iseg50.py", line 391, in capture
    pm, pp, pmiss = rn.photo(xs, img)
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_iseg50.py", line 346, in photo
    assert got and got in names[idx(c)], (c, names[idx(c)], got, prompt)
AssertionError: ('dog', 'dog', 'and', 'a photograph of car and fire hydrant and dog and other object and backgroundground, land, grass, tree, building, wall, sky, lake, water, river, sea, railway, ')
```
