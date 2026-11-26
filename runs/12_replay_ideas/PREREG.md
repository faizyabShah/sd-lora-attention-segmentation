# Pre-registration: seed-consistency weighting of self-attention heads (SCW) and closed-form label-free context refinement (ROC) (screen)

Written 2026-11-16, before the full run, after a 2 + 2-image smoke test (below). Code: `scripts/replay_ideas.py`. BRACE stays frozen. Ground truth only scores.

## Ideas

**SCW: weighting self-attention like cross-attention.**
- A3 already weights cross-attention (CA) layers by how well they agree with self-attention (SA) structure. SCW is the converse: it weights SA heads by how well they agree with the CA seed.
- At every V3 level (16, 32, 64) and call, head h propagates the level's input map: P_h = A_h x.
- The head's weight is softmax_h(z-score over heads of corr(P_h, seed resized to the level)), replacing the uniform mean over heads.
- Everything else in V3 is unchanged (the cap at 16→32 only, the 11 calls).

**ROC: refining the class embedding without training.**
- OVAM optimises the class token embedding with labels. ROC instead takes one closed-form, label-free step on the class token's attribution context vector e, the text-encoder output that cross-attention keys are built from.
- **Step:** e' = e + β ‖e‖ g/‖g‖, where g is the gradient of J = Σ (calls, A3 modules, heads) [mean class logit over BRACE's own core F − mean class logit over clear background B].
- **Cores:** F and B come from the replay baseline's V3, exactly as in `runs/08_feat_bleed`: F = min–max V3 ≥ 0.9, or the top 5% if empty; B = min–max V3 ≤ 0.1 and outside the mask; if B has fewer than 5 cells, the bottom 5% outside the mask.
- J is linear in the step, so this is a one-shot relevance-feedback ("Rocchio") step in key space. There is no UNet backpropagation and no iteration.
- **Primary step size:** β = 0.1.

**ENT (reference, not a test):** iSeg-style entropy weighting of SA heads, softmax_h(−z(mean row entropy)). This is prior art (iSeg, arXiv 2409.03209), included to place SCW.

## Data and replay

- **Images:** the 150 VOC-Sim + 150 COCO-Cap screening images of the rich capture (`CACHE/ana_dump`). These are development images used by earlier diagnostics; this is disclosed.
- **Replay:** the stored inputs of all 31 UNet calls are re-run once, conditional prompt only.
  - SA q and k are captured at the three frozen modules for calls 10–20.
  - The CA queries are captured at the 11 A3 modules for all 31 calls.
- **Replay seed:** the A3 modules, the attribution prompt and token convention of the live run, with the 77-token padded softmax and the mean over class pieces. The difference from live A3 is the layer weighting: the replay uses **uniform** layer weights instead of A3's SA-similarity weights (disclosed).
- **Within-capture comparison:** every arm, including the baseline, comes from the same replay on the same GPU.
- **Fidelity arm `live`:** the stored live A3 seed propagated with the replayed SA.
- **Hardware:** any GPU except wald (wald is busy).

## Scoring

- **Mask:** min–max map, bilinear to 512, τ 0.55, DenseCRF R2.
- **Metric:** class-macro mIoU (counts summed per class).
- **Δ:** against the replay baseline `base`, with an image-bootstrap 95% CI (2000 resamples).

## Screen gate (each primary arm separately: `scw`, `roc0.1`)

- **Promising if:**
  - COCO-Cap Δ ≥ **+0.010** with CI lower bound **> 0**; and
  - VOC-Sim Δ ≥ **−0.005**.
- A promising arm only licenses a new pre-registered confirmation on fresh images, ported into the live recorder. It is never a BRACE change.
- **Reported, not deciding:** `ent`, `roc0.05`, `roc0.2`, `scw+roc0.1`, the `live` fidelity arm, and the replay-fidelity correlations.

## Smoke test (2 VOC-Sim + 2 COCO-Cap images, neumann V100)

- **The code runs at about 6 s per image.** The first image includes model warm-up.
- **Replay fidelity:** r(live-seed V3 replay, stored live V3) = 0.96–0.985, and r(replay seed, live A3 seed) = 0.77–0.99.
- No design change was made.
- The per-image smoke IoUs are not evidence.
