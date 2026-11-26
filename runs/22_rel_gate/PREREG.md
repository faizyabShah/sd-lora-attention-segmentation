# Relative-inconsistency pixel gate: small pilot (30 VOC-Sim + 30 COCO-Cap)

Written and committed before any image of this run is captured or scored. Owner request (2026-11-18). Exploratory
pilot; nothing here is promoted into BRACE.

## Motivation (post hoc, from runs/21)

runs/21's pixel gate used r = exp(−|A − PA| / median_map). The post hoc checks (`runs/21_layer_gate/why_drop.py`,
`why_check.py`, 20 images) found that the absolute residual grows with activation, so reliability is lowest where a
layer fires on the object; inside objects weight moves from the strong 8/16-res layers to the noisy 64-res layers and
the object interior drops. One corrected formula is tested here. No other denominator, exponent or ε is tried.

## Images

Per set, images in crc32(image_id) order at positions 10–39 (30 each), i.e. disjoint from the 10 + 10 images that
motivated the hypothesis. One live regeneration each (frozen runtime, runs/20 corrected prompts, safety checker off).
All variants compared within this capture only.

## Seeds (all at pixel level, per (call t, CA layer l), w = BRACE's GOCA-style weights)

- **BR**: BRACE seed (runtime A3). **UN**: uniform layer weights.
- **Gabs**: runs/21's Gpx unchanged: e = |A − PA| / median_i(|A − PA|) per map.
- **Grel** (the one new formula): e = |A − PA| / (A + PA + ε), ε = median over all maps and pixels of the image of
  (A + PA) (one scalar per image); then e ← e / median over all maps and pixels of e (one scalar per image).
- Gated seed for both: S = Σ w r A / Σ w r, r = exp(−e), min–max.
- **Gabsx**, **Grelx**: shuffled controls, the pixels' reliability values permuted with one fixed permutation per
  image (crc32 seed), identical across (t, l), as runs/21.

P = mean over calls 10–20 × 8 heads of the frozen 64×64 SA module, as runs/21.

## Tables

Every seed through seed alone, fine-only (V0), BRACE hierarchy (V3), NERVE walk; raw and + DenseCRF R2; at fixed 0.55
and at the threshold best on the 30 VOC-Sim images (in-sample on VOC-Sim, applied unchanged to COCO-Cap).
Class-macro mIoU; paired image-bootstrap 95% intervals (2000). With 30 images per set, intervals are wide and
class-macro means rest on few images per class.

## Mechanism diagnostics (GT used only for scoring regions)

Regions at 64×64 (GT area ≥ 0.5): object interior (2-px erosion), next-to-object background (4-px dilation band),
far background.

- **M1 magnitude coupling**: per map, Spearman ρ between r and A over all pixels; w-weighted mean over maps.
- **M2 strong-layer weight**: share of Σ w r inside the object interior held by the 8- and 16-res layers, vs their
  global share Σ w.
- **M3 mass**: seed mass share on GT, and seed mean in far background, vs BR.
- **M4 location**: Grel − Grelx at hierarchy + DenseCRF.
- Also r per resolution group per region, as runs/21 why_check.

## Reading (fixed now)

Grel is called **promising** only if, on both sets: M1 ρ ≥ −0.1; M2 8+16 share ≥ global share − 0.02; M3 mass ≥ BR
and far-background mean ≤ BR + 0.01; M4 point estimate > 0; and Grel − BR at hierarchy + DenseCRF ≥ 0 (point
estimate) at both thresholds. Promising only licenses writing a separate pre-registration for the full 490 + 707
sets; it is not evidence for the method. Otherwise the pixel-gate direction is closed.

## Pipeline

`scripts/jc_relgate.py`: `capture` (GPU) → `score` (CPU) → `report` → `runs/22_rel_gate/result.md`. Smoke on 1 + 1
images (tag `_smoke`, kept).
