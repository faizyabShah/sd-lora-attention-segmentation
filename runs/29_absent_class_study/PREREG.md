# Absent-class calibration: component study on VOC-Sim and COCO-Cap (pre-registration)

Owner request (2026-11-19), after the run 28 screen: study absent-class calibration properly before it is
considered for the JC review work. It is tested with and without propagation, on both benchmarks, with
non-transductive profiles and with a control for the absent-class set. Written and committed before any image
of this run is captured or scored.

**Status.** VOC-Sim is development. COCO-Cap was already used to screen this idea (run 28, N1 +0.022 at 0.55).
So every COCO-Cap number here is labelled **"screened"**, not untuned transfer. The arm closest to untuned
transfer is NX on COCO-Cap: its profiles come from VOC-Sim only. BRACE stays frozen. Nothing here changes the
paper's method; adopting the calibration would be an owner decision, recorded separately. Inference is
label-free: the query class and the image's own prompt or caption are inputs, and masks are used only for
scoring.

## Calibration (unchanged from run 28)

The constants are unchanged from run 28 and are not tuned here: top 1% (spurious), top 5% (object), gain 4.

Per class c and per A3 cross-attention module (11), there are two profiles:
- **Spurious profile m_s.** Mean call-averaged, head-merged, L2-normalised query at the top 1% pixels of class
  word c's call-averaged attention (keys from the class-word prompt). It is taken over images whose target ≠ c
  and whose prompt or caption does not mention c (the run 28 synonym list).
- **Object profile m_o.** Mean query at the top 5% pixels of c's attention, on images whose target = c.

Gate: ŵ = (m_o − m_s)/|m_o − m_s|, p_o = ŵ·m_o, p_s = ŵ·m_s,
g(i) = σ(4·(ŵ·q̂_i − (p_o + p_s)/2)/(p_o − p_s)), from the image's own call-averaged query.

The class word's attention at that module is multiplied by g in every call, before the A3 layer weighting. A
module without a profile gets g = 1 (counted). The profiles come from a separate statistics pass, which is a
second generation of the same prompts.

## Arms (one live SD1.5 generation per image; all arms from it, paired)

Folds: fold(image) = crc32(image_id) mod 5. These are the same folds the scorer uses for out-of-fold (OOF)
thresholds.

| Arm | Seed | Profiles from |
|---|---|---|
| **A** | BRACE seed (runtime A3) | — |
| **NC** | calibrated (**primary**) | same benchmark, images of the **other four folds** only (cross-fitted, not transductive) |
| **NO** | control: m_s replaced by the mean query over **all pixels** of the profile images (no absent-class set) | same as NC (other folds) |
| **NL** | calibrated, run 28 form | same benchmark, all images, own image left out of m_o (transductive) |
| **NX** | calibrated | the **other** benchmark, all images (VOC-Sim → COCO-Cap; COCO-Cap → VOC-Sim, descriptive) |

Each seed goes through three propagations:
- none (seed);
- fine-only self-attention (`sg_combined_v2` V0, two passes of 64×64 SA);
- the BRACE hierarchy (V3).

Each map is scored raw and with DenseCRF R2, at thresholds 0.30–0.70 (step 0.025). Metric: class-macro mIoU.
Intervals: paired image bootstrap, 95%, 2000 resamples.

## Threshold rules

- **VOC-Sim:** 5-fold OOF threshold per map (primary); fixed 0.55 (secondary).
- **COCO-Cap:** each map's best threshold on full VOC-Sim, applied unchanged (primary, as in the JC tables);
  fixed 0.55 (secondary, comparable with run 28).
- Own-best thresholds are shown only as in-sample, descriptive values.

## Registered comparisons (DenseCRF; V3 unless stated)

| ID | Benchmark | Comparison | Threshold |
|---|---|---|---|
| **P1 (primary)** | VOC-Sim | NC − A | OOF |
| P1b | VOC-Sim | NC − A | 0.55 |
| P2 | COCO-Cap (screened) | NX − A | VOC-Sim-selected |
| P3 | COCO-Cap (screened) | NC − A | VOC-Sim-selected |
| P4 | both | NC − NO (does the absent-class set matter?) | VOC-Sim OOF; COCO-Cap VOC-Sim-selected |
| P5 | both | NC − A for seed and fine-only maps; V3 − V0 within A and within NC (is the hierarchy still needed?) | same as P4 |
| P6 | both | NL − NC (cost of removing transduction) | same as P4 |

P5 and P6 are descriptive. Raw rows, 0.55 rows on COCO-Cap and per-class rows are descriptive too.

**Reading rule.**
- **The calibration passes development if P1's interval is above zero.**
- The absent-class explanation is supported only if P4's interval is above zero on VOC-Sim. If it is not, a
  pass is reported as "a query-profile gate helps", not as absent-class calibration.
- Bringing the component to the owner for the paper or the JC deck needs P1 and P2 both above zero. Any claim
  on COCO-Cap keeps the "screened" label.
- A fail is reported as a fail; no constant is retuned after scoring.

## Checks

- The recomputed BRACE seed must match the runtime seed (the job stops above 1e-2).
- Missing-profile modules are counted per arm.

## Files

`scripts/jc_absent.py`. The pipeline:
1. `stats --set voc|coco` (GPU, sharded);
2. `fit` (CPU);
3. `capture --set voc|coco` (GPU, sharded);
4. `score` (CPU, sharded);
5. `report`.

Data goes under `<BRACE_CACHE_DIR>/jc_absent/`; the report is `runs/29_absent_class_study/result.md`. The smoke
run (tag `_smoke`) is kept.
