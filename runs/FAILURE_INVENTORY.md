# Failure-mechanism inventory (2026-11-15)

This file collects every completed experiment by **why** it failed or helped, not by what it was. Its purpose is to separate information that is **missing** (no label-free signal we measured contains it) from information that is **present but unused**.

Sources: `docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md` (§103–135), `outputs/segmentation/sg_real_atlas/analysis/`, `runs/01–05`, `runs/analysis_cluster/`.

Unless marked otherwise, numbers are VOC-Sim / COCO-Cap mIoU. Most diagnostics below are exploratory.

## 1. Wrong semantic seed (the object is not identified correctly)

**Size of the problem.** This is the largest error source on COCO-Cap.
- With an oracle seed, the same V3 + R2 at τ 0.55 reaches 0.887 / 0.836, against 0.825 / 0.710 (150/150 screening images).
- COCO-Cap "person": BRACE 0.28, CLIP-seeded 0.66.

**What helped:**
- **CLIP seed:** +0.015 / +0.020 (0.845 / 0.717).
- **OVAM's label-trained token:** cross-attention map 0.780 → 0.835 on VOC-Sim. It needs annotations.

**What failed:**

| attempt | result |
|---|---|
| content-token rescaling (SG24 A1/A4) | VOC −0.016; COCO constant by construction (class-only attribution prompt) |
| head-contribution weights (A2) | worse on both sets |
| A3 orientation (SG-A3G) | immaterial |
| caption capture | −0.004 |
| caption inversion | −0.033 |
| best other caption word (today) | oracle +0.03 at 32 px; label-free picks no better than the seed; FLOSS-style entropy pick −0.10 |
| dose–response seed (idea 2) | 0.15 (peaky maps) |
| late / early fusion with the CLIP seed | registered tests fail vs CLIP-seeded; post-hoc multiplicative lead is class-macro only (driven by person / bottle / plant) |
| per-image reliability gating | fails |
| class word present in caption as a selector | no gain |

**Present or missing?**
- **Present but unused:**
  - a better token direction in embedding space (OVAM shows it exists, but it needs labels to find);
  - CLIP semantics (already used).
- **Missing:** a label-free way to know, per image, which seed is right. The oracle pick between BRACE and CLIP-seeded is worth about +0.025 over the best fusion, but no label-free statistic finds it.

## 2. Self-attention spreads evidence beyond the object (bleed)

**Size of the problem.**
- COCO-Cap failures: 97 of 142 are bleed.
- Background-word maps don't flag false positives (AUROC 0.37 / 0.49).

**Mechanism evidence (regeneration pilot, today).**
- SD's own class-token support extends past the object boundary: a ground-truth mask dilated by 16 px often had the lowest regeneration error.
- So attention support is not object extent.

**What failed:**
- **Paper-inspired propagation variants:**
  - SG28 (PEARL: boundary-aware Laplacian): +0.0001 after R2;
  - SG29 (NERVE walks), SG30 (SPARK inflation), SG31 (SeeDiff re-seeding: collapses recall).
- **Operator batch B:** PageRank, fg/bg competition, random walker, graph cut, low-pass, rollout, temperature, sinks: all ≤ noise or worse.
- **Batch A:** components 0.000; geodesic gating −0.06 / −0.075.

**A recurring trade-off.** Tighter propagation helps cluttered COCO-Cap and hurts VOC-Sim:
- SG29 N2: VOC −0.009, COCO +0.007;
- SG30 S2: VOC −0.011, COCO +0.023 (positive paired interval);
- SG31 S1 (16→64): VOC −0.006, COCO +0.036 over S0.

VOC-only selection rejects these every time, by design. This means that **the development set cannot select mechanisms that target clutter.** That is a property of the protocol, not evidence that the mechanisms are useless.

## 3. Inaccurate boundaries

- **R2 (DenseCRF)** is the only boundary step that has ever helped.
- **Failed:** matting (−0.03 / −0.07), graph cut on SA (−0.02 to −0.13), SA-kernel CRF (within noise), stability and area rules.
- **Structural limit:** self-attention is 64×64, 8 px per cell. The pilot's dilated-mask preference is exactly at this scale.
- **SA boundary diagnostic** (`runs/05_sa_boundary`, pre-registered): CLOSED.
  - On its own, SA separates boundary-crossing edges worse than RGB colour (AUROC 0.74 vs 0.84 on COCO-Cap).
  - It adds only +0.015 beyond RGB.
  - On BRACE's bleed cells, measured from BRACE's own core, it adds +0.040 on COCO-Cap (CI −0.010…+0.089): below the +0.05 bar, and the CI crosses 0.

## 4. Threshold calibration

- **Headroom:** an oracle per-image τ is worth +7 / +10 points.
- **What V3 already does:** its gain over V0 is cross-image calibration (§134), not better per-image ranking.
- **Failed:**
  - Otsu, stable-mask τ, Otsu + d (SG-threshold, §134);
  - stability τ and area-matched τ (batch A);
  - image statistics (part 1);
  - constrained regeneration (pilot, all four bars fail).
- **Present or missing?** Missing. No label-free signal we measured predicts the per-image τ.

## 5. Generation-side counterfactuals (still running)

On the A40, ideas 1, 3, 4, 6 and 7 (knockout, chrono, Jacobian, OT, CFG contrast) are in the 150/150 screen.
- **Swap / knockout** already cover prompt substitution and per-step token sensitivity (finite differences).
- **Time-resolved analysis** of those signals is a follow-up only if the screen shows they carry signal.

## What this suggests

1. **The seed is where the headroom is.** Its missing piece is label-free selection, not new map construction. Every label-free selector tried has failed, so the next idea should bring new evidence, not another selector over the same evidence.
2. **Bleed is attention support, not a propagation bug.** Propagation operators can't remove what the attention itself says, and the pilot shows SD's support really is wider than the object.
3. **The VOC↔COCO trade-off (§2) is the most consistent unexploited pattern.** Several independent mechanisms help clutter and hurt simple scenes. Exploiting it needs either:
   - a label-free scene-complexity signal (for example, the number of content words in the prompt); or
   - a development set with clutter.

   Both touch the protocol rule that COCO-Cap is untuned transfer, so this is an owner decision before any experiment.
4. **Per-image τ and boundary placement look like missing information**, not unused information. More selectors of the same kind are unlikely to work.
