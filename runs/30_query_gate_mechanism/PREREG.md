# Query-profile gate: mechanism and class-adaptive follow-up (pre-registration)

Owner request (2026-11-19), after run 29 and an external review of it. Run 29 failed its registered development
test (VOC-Sim +0.000), and the absent-class set was not needed (NC ≈ NO). It did show:
- seed and fine-only gains on both benchmarks;
- a screened COCO-Cap gain after the hierarchy (+0.014; +0.013 with VOC-Sim-only profiles).

This run asks **what in the gate does the work**, **where it helps**, and whether a **class-adaptive** choice,
selected on VOC-Sim only, helps.

**Status.** Exploratory follow-up. It does not overturn run 29's verdict. COCO-Cap stays labelled "screened".
Nothing is promoted from this run. A positive result here would need a confirmation on fresh annotated images
before any claim. BRACE stays frozen. Inference is label-free; masks are used only for scoring and diagnostics.
Written and committed before any capture or scoring of this run.

## Setup

- **One live SD1.5 generation per image** on node renyi (A100), all arms from it (paired). It is not paired with
  run 29's capture.
- **Profiles reuse run 29's statistics pass** (`jc_absent/fit.pt`): fold sums of object, spurious and all-pixel
  queries per class and module. Folds: crc32(image_id) mod 5.
- **Gate constants unchanged:** top 1% (spurious), top 5% (object), gain 4.

## Arms (seed variants; each through seed, fine-only V0, hierarchy V3)

Every gate has the form g = σ(4·(ŵ·q̂ − (p₊ + p₋)/2)/(p₊ − p₋)), with p₊ = ŵ·m_o. Profiles are cross-fitted
(other folds) unless stated.

| Arm | Direction ŵ | Negative anchor p₋ | Question |
|---|---|---|---|
| A | — | — | BRACE |
| NC | m_o − m_s | ŵ·m_s, absent-class spurious profile | run 29 primary |
| NH | as NC, gate applied as g^0.5 (half strength) | | strength |
| NO | m_o − m_all | ŵ·m_all, all-pixel mean | run 29 control |
| PD | m_o only | ŵ·m_all | is a negative direction needed? |
| PI | m_o only | the image's own mean projection (no population negative at all) | is any population negative needed? |

## Class-adaptive rules (chosen on VOC-Sim only; never on COCO-Cap)

For each class, choose the seed arm with the best VOC-Sim per-class mIoU (hierarchy + DenseCRF, each arm at its
OOF predictions). Ties go to the earlier arm in the list.

- **CAg (strength):** choose among {A, NH, NC}.
- **CAa (any):** choose among {A, NC, NH, NO, PD, PI}.

On VOC-Sim, the choice for each fold's images is made from the other four folds (cross-fitted). On COCO-Cap, the
choice from all of VOC-Sim is applied unchanged, with each chosen arm's VOC-Sim-selected threshold.

## Scoring

Thresholds 0.30–0.70 (step 0.025); raw and DenseCRF R2. Class-macro mIoU, paired image bootstrap (2000).

| Benchmark | Primary threshold | Also reported |
|---|---|---|
| VOC-Sim | 5-fold OOF | 0.55 |
| COCO-Cap | VOC-Sim-selected | 0.55 |

**Comparisons** (hierarchy + DenseCRF, primary thresholds; all exploratory):
- each arm − A;
- NO − NC, PD − NC, PI − NC, NH − NC;
- CAg − A, CAa − A;
- each arm − A for the seed and fine-only maps.

**Mechanism reading** (fixed now, descriptive):
- If PD or PI is within the interval of NC (Δ vs NC interval contains 0) and positive vs A on COCO-Cap, the
  simpler object-prototype gate is what does the work.
- If PI falls clearly below NO, a population negative anchor is needed.

## Diagnostics (masks used only here)

At 64×64, from the downsampled ground-truth mask, each image is split into:
- **foreground;**
- **near background:** within 6 px of the object, Chebyshev distance;
- **far background:** the rest.

| ID | Measure | Split |
|---|---|---|
| D1 | Mean gate (uniform mean of the 11 module gates, upsampled) in fg, near and far, per gate arm | overall and per class |
| D2 | Fraction of min-max seed mass in fg, near and far, per arm | overall |
| D3 | Spearman correlation of per-image gain (IoU of NC − A, hierarchy + DenseCRF, primary threshold) with object area, far-background seed mass of A, and number of other VOC classes named in the caption (COCO-Cap) | overall |
| D3 | Mean gain by tertile of each of those three measures | by tertile |

## Files

`scripts/jc_qgate.py`:
1. `capture --set voc|coco` (GPU, renyi);
2. `score` (CPU, renyi);
3. `report`.

Data goes under `<BRACE_CACHE_DIR>/jc_qgate/`; the report is `runs/30_query_gate_mechanism/result.md`. The
smoke run (tag `_smoke`) is kept.
