# Run 51 execution notes

GPU: NVIDIA A40, 46068 MiB, 520.61.05. The experiment ran in GNU Screen.

```text
run 51 hard: 100 photos, 364 primary items (178 small < 5%), 59 classes
run 51 random: 100 photos, 221 primary items (88 small < 5%), 60 classes
random half: pool positions 1000-1100
captions: 200 photos; prompt-B lists: other primary classes named 0.46 on average, 0.21 named classes not among the photo's primary classes per item
```

Both smoke captures and the smoke report completed before full capture.
The scorer produced all 174 map cells for each of the 585 items.

## SD 1.5

- profiles: /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_ov42/fit.pt (model SD 1.5)
- 3510 passes over 200 photos in 64.4 minutes of per-photo capture (0.91 passes/s), including smoke photos.

```text
check 5385 (random): A with g = 1 vs V0 max |diff| 0.0e+00; C with singleton regions vs seed 0.0e+00; regions 36
check 8781 (hard): A with g = 1 vs V0 max |diff| 0.0e+00; C with singleton regions vs seed 0.0e+00; regions 36
check 8781: 15|T|UL|V0 vs run 50's saved map, max |diff| 1.2e-04
check 8781: 15|S2|IL|ERS vs run 50's saved map, max |diff| 4.8e-07
```

## SD 2.1-base

- profiles: /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_sd21_47/ov42/fit.pt (model sd2-community/stable-diffusion-2-1-base)
- 1370 passes over 200 photos in 24.7 minutes of per-photo capture (0.92 passes/s), including smoke photos.

```text
check 5385 (random): A with g = 1 vs V0 max |diff| 0.0e+00; C with singleton regions vs seed 0.0e+00; regions 36
check 8781 (hard): A with g = 1 vs V0 max |diff| 0.0e+00; C with singleton regions vs seed 0.0e+00; regions 36
check 8781: 21|T|UL|V0 vs run 50's saved map, max |diff| 1.2e-04
check 8781: 21|S2|IL|ERS vs run 50's saved map, max |diff| 4.8e-07
```

The registered driver and settings were used with the existing Run 50 cache and pinned BLIP-large checkpoint.

## Initial check stop and restart

The original SD 1.5 check raised `KeyError: 64` after `item()` cleared the recorder. Commit `b87e2951` reruns the T pass before reading the attention for the check. The saved item list, photos and captions were reused.

```text
Traceback (most recent call last):
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_gate51.py", line 473, in <module>
    {"items": items, "caption": J.caption, "check": check, "capture": capture, "score": score, "report": report, "viz": viz}[a.cmd](a)
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_gate51.py", line 227, in check
    out, _, _ = rn.item(x, img, BL[x["id"]]); H = _attention(rn.rec.grid_qk[64], 64)
KeyError: 64
```

The first resume launch checked for captions.json and stopped at preflight. The launcher now uses the saved captions_uncond.json. No experiment stage ran during that attempt.

## SD 2.1 check stop and restart

The first SD 2.1 check imported the eight-head attention helper before model setup, which replaces it with the five-head version. Commit `1472064b` moves the import after setup. The SD 1.5 check had passed and was reused.

```text
Traceback (most recent call last):
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_gate51.py", line 478, in <module>
    {"items": items, "caption": J.caption, "check": check, "capture": capture, "score": score, "report": report, "viz": viz}[a.cmd](a)
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_gate51.py", line 232, in check
    H = _attention(rn.rec.grid_qk[64], 64)
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/src/segmentation/sg25_sa.py", line 41, in _attention
    raise ValueError(f"Unexpected {side}x{side} SA Q/K: {tuple(q.shape)}, {tuple(k.shape)}")
ValueError: Unexpected 64x64 SA Q/K: (5, 4096, 64), (5, 4096, 64)
```

## Smoke report stop and restart

The random smoke photo had no small items, so the bootstrap for that empty subset raised `ZeroDivisionError`. Commit `245dcdfb` handles empty subsets. Only the report command was rerun, using the existing smoke scores.

```text
Traceback (most recent call last):
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_gate51.py", line 478, in <module>
    {"items": items, "caption": J.caption, "check": check, "capture": capture, "score": score, "report": report, "viz": viz}[a.cmd](a)
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_gate51.py", line 317, in report
    TS = {hn: D.Table([x for x in r if x["area"] < SMALL], a.boot) for hn, r in H.items()}
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_gate51.py", line 317, in <dictcomp>
    TS = {hn: D.Table([x for x in r if x["area"] < SMALL], a.boot) for hn, r in H.items()}
  File "/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_diag49.py", line 443, in __init__
    self.Wb = rng.multinomial(n, [1 / n] * n, size=boot).astype(np.float64)
ZeroDivisionError: division by zero
```
