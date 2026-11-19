# JC #1 (claim) and #8 (structure): side N draft, 2026-11-19

Status: draft for the owners, written before the meeting. Nothing in `paper/` is changed yet.
Numbers marked **[pending]** wait for the official N-side rows (`comparisons/official/`) or for
the owners' choice between the original and the prompt-corrected benchmark capture (runs/19-20).

## 1. What the evidence now supports

| Claim in the draft | Evidence since the draft | Keep? |
|---|---|---|
| "+0.125 / +0.114 over class-word OVAM" as the headline | Most of it is configuration (CA scales, SA passes, call window, DenseCRF); the two new components add about +0.006 each on VOC-Sim (JC #1) | Keep as a row in the table, not as the headline |
| Capped hierarchy gives +0.034 on COCO-Cap | Entry 9: the +0.034 compares two maps at two different VOC-Sim-selected thresholds (0.55 vs 0.425). At each map's own best threshold +0.018 [+0.006, +0.029]; at one shared threshold (0.55) +0.012 [-0.002, +0.026] | Keep, restated as **cross-image calibration**: about +0.018 is calibration (survives each map's best threshold; the per-image oracle is a tie, 0.896 vs 0.895, so it is not better ranking within an image) and about +0.016 is threshold transfer. This is the paper's own oracle finding (Sec. 5.5), so the claim and the analysis now agree |
| SA-guided layer weighting helps | Entry 8 rerun, class word: +0.004 VOC-Sim, **+0.024 [+0.013, +0.036] COCO-Cap** | Keep: the cleanest positive result on cluttered scenes |
| One-sided cap beats no cap | Entry 8 rerun: +0.010 VOC-Sim, -0.005 [-0.011, -0.001] COCO-Cap at 0.55 | Weaken: helps on VOC-Sim, not on COCO-Cap at a fixed threshold. Keep the mass analysis (13% background vs 0.5% object removed) and the comparison with exact conservation (which costs up to five points) |
| "DenseCRF gains six times more on capped maps" (abstract, `tab_refine.tex`) | N audit: same two-threshold artefact; at matched thresholds DenseCRF adds about +0.014 to both maps | **Drop** |
| Components hold with optimized tokens | Entry 8: none of the three holds on COCO-Cap with OVAM's token | State as a limitation of scope: the design is for class-word seeds |
| Best training-free method on both benchmarks | runs/20 (corrected prompts): NERVE's CLIP seed + its walk 0.843 / 0.735 vs BRACE 0.832 / 0.709 (VOC-sel., DenseCRF). Entry 10: with the SD-only seed, our hierarchy beats NERVE's walk (-0.019 for the walk); with NERVE's CLIP seed the walk wins (+0.033) | Narrow to **best SD-only (no second model) training-free method**; NERVE and SCLIP-seeded rows go in a separate "with an extra vision-language model" block |
| Selection protocol chose the final method | The registered rule kept the simpler reference; the final method is an owner override (protocol Section 135) | Say so in the protocol section and in the contribution list (JC #1) |

## 2. Draft abstract (reframed)

> Text-to-image diffusion models localize words in cross-attention (CA) and group pixels in
> self-attention (SA), so the images they generate can be segmented without training. We study
> how these signals should be combined across spatial scales, and find that the choice matters
> most in cluttered scenes. Our training-free pipeline, BRACE, weights CA layers by their
> agreement with SA, propagates the resulting class-word seed from coarse to fine SA, and lets each
> finer level redistribute evidence only within the mass its coarse parent assigns. Selected on
> class-stratified folds of single-object VOC-Sim, where the new components add little
> (+0.006 each), the frozen pipeline transfers to the cluttered COCO-Cap at 0.696 **[pending:
> 0.709 on the corrected capture]**, 0.034 above a fine-only pipeline selected the same way. An
> oracle analysis shows why: the hierarchy does not rank pixels better within an image, it makes
> the maps comparable across images, so a threshold chosen on simple scenes still works on
> cluttered ones; even with each map given its own best threshold, about half of the gain remains. SA-guided
> layer weighting is the most robust component in clutter (+0.024 on COCO-Cap). Among methods that
> use only the diffusion model, BRACE is the strongest we evaluated; a method that adds a CLIP
> seed is stronger on COCO-Cap **[pending: official paired Δ]**, and we report it separately.
> All settings are selected on VOC-Sim; the choice of the hierarchy over the simpler reference
> preferred by our registered rule was made after seeing COCO-Cap, and we report both pipelines
> and every negative result.

## 3. Draft contribution list

1. A diffusion-only, training-free pipeline that combines CA and SA across scales: SA-guided CA
   layer weighting, coarse-to-fine SA propagation, and a one-sided cap.
2. Evidence that the scale structure matters in clutter, not on single objects: on VOC-Sim the
   new components add about +0.006 each; on COCO-Cap the frozen pipeline gains +0.034 over
   fine-only, about half of it with each map at its own best threshold, all of it cross-image
   calibration rather than better ranking within an image.
3. An oracle threshold analysis that explains the gain as cross-image calibration, and its main
   failure (objects that fill the frame).
4. A fold-safe protocol with the registered rule, the owner override that chose the final method
   (made after COCO-Cap was seen, and labelled so), and all negative results reported; plus paired reruns of published methods on the same images.

## 4. Sentence for the owner override (Sec. 4, protocol)

**Checked (protocol Section 135, 31 October 2026): the override was made AFTER COCO-Cap was
scored, and its evidence table cites the COCO-Cap +0.0341.** So COCO-Cap is not untouched
transfer for the decision to adopt the hierarchy (every individual setting was still selected on
VOC-Sim). The registered rule kept the fine-only reference (FINE-P2 + R2) because the hierarchy
fails the -0.050 worst-class guard in every fold, although its VOC-Sim gain is +0.0063 (5/5 folds,
CI [+0.0010, +0.0115]). Proposed wording:

> All settings were selected on VOC-Sim folds. The registered selection rule kept the simpler
> fine-only pipeline, because the hierarchy, although better on VOC-Sim in every fold
> (+0.006), failed our worst-class guard. We adopted the hierarchy as an explicit owner decision
> **after** observing its COCO-Cap result, so COCO-Cap is not an untouched test set for that one
> decision. We therefore report both pipelines on both benchmarks (Table X), and treat the
> COCO-Cap gain as evidence for the calibration analysis of Sec. 5.x rather than as a held-out
> confirmation.

A clean held-out check of this one decision needs a third benchmark scored once with both
pipelines frozen (candidate: a fixed-rule subset of generated Cityscapes scenes, or another
cluttered generated set). Raise this in the meeting as an option.

## 5. #8 proposed structure

| Now | Proposed |
|---|---|
| 4 Experimental setup | 4 Experimental setup: benchmarks, protocol (with the override sentence), metrics, **implementation details: GPU, driver, library versions, runtime and peak memory per image for BRACE and fine-only** (F's #8 numbers) |
| 5 What each stage contributes | **5 Experimental results** |
| 6 Comparison with published methods | 5.1 Comparison with published and rerun methods (former Sec. 6; paired rerun block, "with extra model" block, published block) |
| | 5.2-5.6 the former 5.1-5.5 (seed, envelope, cap, DenseCRF, calibration) |
| 7 Discussion and limitations | 6 Discussion and limitations (adds: optimized-token scope, CLIP-seeded methods, SD 2.1/SDXL scope, SD3/FLUX out of scope) |
| Conclusion | 7 Conclusion |

Moving the comparison to the front of Sec. 5 makes the narrowed claim (#1) the first result the
reader sees. Apply after the meeting, in one commit, after F's Table 1 rewrite (#7), to avoid
conflicts in `paper/sec/5_analysis.tex` and `paper/tables/`.
