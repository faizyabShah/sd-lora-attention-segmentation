# Run 33: noise levels and profile sources on real PASCAL VOC photos (pre-registration)

Committed before any capture or scoring of this run. BRACE stays frozen. **Exploratory.** This follows run 32
(`runs/32_realvoc_factorial/`) and uses the same items, prompts, layer map, components and scoring.

**Second look at VOC val.** Run 32 already reported VOC 2012 val. Nothing in run 33 is chosen on val:
- the contrasts below are fixed now;
- every threshold is chosen on VOC train.

Run 33 val numbers are a second look at the same photos, and are labelled that way.

## 1. Questions

1. **Noise level.** Does reading attention at several noise levels beat the single pass at t = 100? With several
   noise levels, does the hierarchy start to help, as it does in generation-time BRACE?
2. **Profile source.** Run 32's label-free profiles (B: generated VOC-Sim images, absent = the prompt doesn't
   mention c) helped much less than profiles from VOC labels (A). Is that because:
   - B's images are generated;
   - "absent" is decided from captions;
   - or B has fewer fit images?

## 2. What is already known (no new run needed)

JC #5 (`comparisons/backbones/result.md`) captured the same stored VOC-Sim images in one pass at t = 100. With
SD1.5, BRACE, DenseCRF and 0.55 it scores **0.806 / 0.666** (VOC-Sim / COCO-Cap). Generation-time BRACE scores
0.832 / 0.709 (runs/20).

So on generated images, a single pass costs about −0.03 / −0.04, against a gap of about −0.2 between VOC-Sim
and real VOC. Different settings, descriptive only.

## 3. Capture (one capture for every arm)

Items, prompts, layer map and recorder are as in run 32:
- the 2,170 VOC train items and 2,148 val items;
- the prompt `A photograph of a {w}`;
- `scripts/jc_realvoc.py` functions, imported unchanged.

Per item:
- **VAE:** one encode.
- **Noise:** one noise draw ε (CPU generator, seed 0), **shared by all noise levels**.
- **UNet:** three conditional SD1.5 calls, at **t = 100, 300 and 500**.

| Arm | What is read |
|---|---|
| `t100` | the t = 100 call alone (the run 32 setting, re-captured here; not mixed with run 32) |
| `t300` | the t = 300 call alone |
| `t500` | the t = 500 call alone |
| `M` (multi) | all three calls, aggregated as generation-time BRACE does (`sg_combined_v2.aggregate`) |

How arm `M` aggregates:
- **Seed:** the per-call weighted CA sums (each call with its own A3 weights and its own gate) are averaged over
  the 3 calls, then min-max.
- **Propagation:** the seed is propagated through each call's SA (fine-only, uncapped hierarchy, capped
  hierarchy). The 3 results are averaged before min-max.

## 4. Components and profile sources

The components are as in run 32:
- **W:** A3 vs uniform layer weights.
- **H:** uncapped hierarchy V1 vs fine-only V0.
- **G:** run 31 MK4 gate, K = 4.

**Profiles are fitted separately at each noise level**, from one pass of the fit images at that t (same noise
draw). In arm `M`, each call's gate uses the profiles of its own t.

| Source | Fit images | "Absent" means | Labels? | Size |
|---|---|---|---|---|
| **A** | VOC train items (cross-fitted by fold on train; all of train for val) | not in the image-level labels | image labels | 2,170 items |
| **S** | A cut to B's size: per class, as many items as B has, seeded by crc32(item id) | as A | image labels | 490 |
| **B** | the 490 VOC-Sim generated images, runs/20 prompts | the prompt doesn't mention c | none | 490 |
| **C** | **real COCO train2017 photos**, 490 photos, per-class counts matched to B; prompt `A photograph of a {w}` | none of the photo's 5 COCO captions mention c | none (captions only) | 490 |

Selection rules:
- **C:** a photo qualifies for class c when at least one caption mentions c (`jc_trio.mentions`, the run 31/32
  synonym list). Classes are filled rarest first. Photos are drawn without replacement, seeded by
  `random.Random(33)`. COCO train2017 shares no photos with VOC.
- **S, on train:** S takes the per-class counts within the other fold (capped at what that fold has). On val, it
  takes them from all of train.

How the comparisons separate the three explanations:
- **C − B:** real vs generated images, both caption-based and both 490. They still differ in prompt and in the
  caption rule (5 captions vs the generation prompt).
- **S − C:** labels vs captions on real photos at equal size. They still differ in VOC vs COCO photos.
- **A − S:** size alone (2,170 vs 490 items, same rule).

## 5. Maps, thresholds, scoring

Per arm:
- **Seeds:** u-, w-, uA4, wA4, uB4, wB4, uC4, wC4, wS4.
- **Maps:** each seed as seed / V0 / V1, plus w-|V3 (BRACE).
- **DenseCRF R2:** the factorial maps for A, B and C, plus w-|V3 and wS4|V1. That is 18 maps per arm.

Thresholds and scoring:
- **Thresholds:** grid 0.30–0.70 in steps of 0.025, chosen per map on the train items (raw maps). On val,
  DenseCRF is applied at that threshold and at a fixed 0.55.
- **Metric:** class-macro mIoU over items.
- **Intervals:** 95% from a 2000-sample photo bootstrap. The same resamples are used for every map, so
  differences are paired.

## 6. Primary contrasts and reading rule (fixed now)

The primary rule is **val, + DenseCRF, train-selected threshold**:
- CI above 0 = **helps**;
- CI below 0 = **hurts**;
- otherwise **not resolved**.

| # | Contrast | Question |
|---|---|---|
| P1 | `M:wA4\|V1` − `t100:wA4\|V1` (full stack, source A) | Do several noise levels help? |
| P2 | `M:w-\|V3` − `t100:w-\|V3` (BRACE, one capture) | The same, for BRACE |
| P3 | Average hierarchy effect (run 32 definition) in arm `M`, source A | Does the hierarchy help with several noise levels? |
| P4 | `t100:wC4\|V1` − `t100:wB4\|V1` | Real vs generated fit images (captions both) |
| P5 | `t100:wS4\|V1` − `t100:wC4\|V1` | Labels vs captions, real photos, equal size |
| P6 | `t100:wA4\|V1` − `t100:wS4\|V1` | Number of fit images |

Secondary, also reported:
- P4–P6 in arm `M`;
- t300 and t500 against t100;
- the average W, H and G effects in every arm (sources A, B, C);
- the raw and 0.55 rules;
- per-class IoU (person especially);
- a descriptive comparison of `t100` with run 32 (a different capture, so not a paired test).

## 7. Checks (smoke, before the full run)

- At t = 100, the single-call V0 / V1 / V3 match `sg_combined_v2.aggregate` fed that call 11 times:
  max |Δ| < 1e-5 after min-max.
- Arm `M` with three identical calls equals the single-call maps: max |Δ| < 1e-5.
- Every map is finite.
- C photos: every item's class is mentioned in one of its captions, and no absent class is.
- S per-class counts equal B's (or are capped as stated).

## 8. Files

- **Script:** `scripts/jc_realvoc_t.py` (imports `scripts/jc_realvoc.py`).
- **Report:** `runs/33_realvoc_timesteps/result.md`.
- **Data:** `<BRACE_CACHE_DIR>/jc_realvoc_t/`.
- Smoke outputs are tagged `_smoke` and kept.
