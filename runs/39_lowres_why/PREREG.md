# Run 39: why the low-resolution consensus gate works (stage-1 diagnostics; pre-registration)

Committed before any computation of this run. BRACE stays frozen. **Exploratory, diagnostic.** No mIoU is computed
here, and nothing is promoted.

## 1. Questions

Runs 36–38 found that MK4-LR beats MK4 on real VOC and on generated images. MK4-LR is MK4's gates at the six modules
below 64 × 64, averaged and applied at every module. Run 35 / 37 per-module AUROC (object vs far background) is
0.78–0.99 at the low-resolution modules but 0.43–0.68 at the five 64 × 64 modules.

1. **Resolution:** do the low-resolution modules have better *representations*, or do they just benefit from
   *spatial averaging*?
2. **Which modules:** which of the six low-resolution modules carry the information (single-module and
   leave-one-out)?
3. **Robust consensus:** does a median or trimmed mean of the six gates beat their plain mean (LR)?

## 2. Data (no new generation; one real-photo pass)

| Set | Images | Profiles (MK4, held out) | Masks |
|---|---|---|---|
| **Real VOC train** (primary) | the 2,170 VOC train items of runs 32–37; one SD1.5 pass at t = 100 (run 32 noise), recomputed on GPU because no per-module query dump exists | **C** (run 33 `C_all`: real COCO photos, captions; never fitted on VOC). Also **A** (VOC train labels, cross-fitted by fold), as reference. | VOC train |
| Generated dev | run 31 dev dumps: 253 VOC-Sim and 356 COCO-Cap (per-module queries and attention) | run 35 fit; dev image → other fold; regime V on COCO-Cap → all VOC-Sim dev | VOC-Sim, COCO-Cap |

- Generated settings are run 35's: VOC-Sim / V and COCO-Cap / X (rule settings), plus VOC-Sim / X and COCO-Cap / V.
- No VOC val, generated test image or COCO val image is used.

## 3. Measures (per item; then mean over items within a class, then over classes)

Regions (object, near band, far background) are computed by `jc_mneg._regions_at` at the evaluation grid, from the
mask pooled to that grid. **Masks matched to the resolution** are therefore built into every comparison.

**A. Resolution control.** AUROC (object vs far) of the γ = 1 MK4 gate for each of the 11 modules, scored on these
grids:

| Grid | How each gate reaches it |
|---|---|
| native | as computed |
| **16 × 16 (primary)** | 64 × 64 gates area-pooled; the 8 × 8 module upsampled bilinearly |
| 8 × 8 | area-pooled |

Plus two **pooled 64 × 64** variants, scored at 16 × 16:
- **gate pooling:** the 64 × 64 gate, area-pooled to 16 × 16;
- **query pooling:** the 64 × 64 module's queries area-pooled to 16 × 16 and re-normalised, *then* gated with the
  same profile. The profiles were fitted on unpooled queries, which works against query pooling. This is stated as a
  caveat.

**Reading (fixed now).** Primary set real VOC train / C. Let:
- a = the mean 16 × 16 AUROC over the six low-resolution modules;
- h_g = the mean over the five 64 × 64 modules with gate pooling;
- h_q = the same with query pooling.

| Outcome | Reading |
|---|---|
| max(h_g, h_q) ≥ a − 0.05 | **spatial averaging** explains most of the gap |
| max(h_g, h_q) ≤ a − 0.15 | **representation difference** |
| otherwise | **partial** |

The same reading is reported for the generated rule settings, as a descriptive check.

**B. Consensus variants.** Each is a single map on the 64 × 64 grid: bilinear upsampling, then a combination.

| Variant | Definition |
|---|---|
| **LR** | mean of the six low-resolution gates (run 36). This is the reference. |
| LR-med | per-pixel median of the six |
| LR-trim | per-pixel mean after dropping the highest and lowest of the six |
| AV | mean of all 11 |
| HR | mean of the five 64 × 64 gates |
| HR-gp | mean of the five 64 × 64 gates after gate pooling to 16 × 16 and upsampling |
| HR-qp | the same with query pooling |
| only-k | low-resolution module k alone (6 variants) |
| LOO-k | LR without module k (6 variants) |
| MK4 | each module's own gate. Descriptive: retention is averaged over modules on their own grids. |

For each variant:
- AUROC, object vs far, on the 64 × 64 grid;
- retention curves Σ w·g^γ / Σ w for object, near and far, γ ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4}, weighted by the A3
  seed mass (primary, as runs 35/36);
- **ΔR vs LR:** object retention at matched far retention, averaged over LR's γ ∈ [0.5, 2] (run 35's measure).

**Pass rule for robust consensus (fixed now).** LR-med or LR-trim passes if ΔR vs LR ≥ +0.01 with its 95% CI above 0
on **both** real VOC train / C and COCO-Cap dev / X. A passing variant would get its own pre-registered end-to-end
run; nothing is captured end to end here. Single-module, leave-one-out, AV, HR, HR-gp and HR-qp are **diagnostics**
and are not read with the pass rule.

**Intervals:** 2000-sample image bootstrap, class-balanced, paired across variants. Everything is also reported with
person excluded.

## 4. Checks (smoke)

- The LR map equals `jc_gate36.consensus` with LR weights, at a 64 × 64 module: max |Δ| < 1e-6.
- Real-photo pass: the A3 seed and module names match the recorder.
- All gates are in [0, 1] and finite.

## 5. Files and cost

- **Script:** `scripts/jc_res39.py`.
- **Report:** `runs/39_lowres_why/result.md`.
- **Rows:** `<CACHE>/jc_res39/`.

| Part | Cost |
|---|---|
| real VOC train pass | about 20 GPU-minutes over 2 shards |
| generated | about 30 CPU-minutes over 2 shards |
| report | a few CPU-minutes |

- Smoke outputs are tagged `_smoke` and kept.
