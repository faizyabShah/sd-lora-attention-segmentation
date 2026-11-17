# Seed × attention source × propagation rule (pre-registration)

Written and committed before any image of this run is scored. BRACE is not
changed by this run, whatever the outcome.

## Why

The seed × propagation swap (`runs/17_seed_walk_swap`) found a strong
interaction on COCO-Cap: NERVE's walk beats BRACE's capped hierarchy on NERVE's
CLIP seed (+0.033 [+0.019, +0.049]) but loses on BRACE's cross-attention seed
(−0.019 [−0.031, −0.005]). In that swap, "NERVE's walk" changed two things at
once:

- **the self-attention it reads:** one pass of the image, noised to t = 101,
  empty prompt, instead of BRACE's generation-time self-attention (denoising
  calls 10–20, conditional branch);
- **the rule applied to it:** global + local fusion and a closed-form walk
  (δ = 0.9), instead of the capped 16→32→64 hierarchy.

This run separates them. It was suggested by an external review (ChatGPT) of
runs 14–18 and requested by the owner.

## Design

One live generation per image with the frozen runtime
(`scripts/sg27_runtime.py`), on one GPU (erdos or neumann).

| Factor | Levels |
|---|---|
| Seed | **CA**: BRACE's live cross-attention seed. **CLIP**: NERVE's CLIP ViT-L/14 seed from runs/17 (released code, side N's variant A, computed on the stored benchmark image) |
| Attention source | **G**: the frozen runtime's generation self-attention, calls 10–20, conditional branch. **N**: NERVE's source: one pass of the **generated** image through the same UNet (VAE mean latent, noise seed 0, t = 101, empty prompt, no guidance, fp16 autocast) |
| Rule | **H**: BRACE fine-only (two passes, V0) and capped hierarchy (V3), frozen code (`sg_combined_v2.aggregate`) on BRACE's three SA modules (16/32/64). **W**: NERVE's fusion and closed-form walk, ported from its released code (`scripts/nerve_walk_port.py`) on NERVE's two 64-px layers (indices −4, −6) |

- **Each rule keeps its own layers;** only the attention source changes. On G
  the rules read their modules at calls 10–20. On N they read the same modules
  from the single noised pass.
- **NERVE's walk on G:** NERVE's fused matrix is computed per call, averaged
  over the 11 calls, and then walked once.
- **BRACE's rule on N:** the single call stands in for all 11, so the hierarchy
  equals its one-call version.
- **Head weights in NERVE's fusion are uniform** for a single class (side N's
  K = 1 guard). The fused matrix therefore does not depend on the seed, and
  both seeds share it.
- **Seed inputs are as in runs/17.** The walk gets the CA seed at 64 px and the
  CLIP seed at its native grid (NERVE's refinement resizes it). The hierarchy
  gets both at 64 px, min–max scaled.
- **Not changed:** DenseCRF (frozen settings) and the threshold grid
  (0.30–0.70, step 0.025).

Cells per seed: seed alone, then G×{fine-only, hierarchy, walk} and
N×{fine-only, hierarchy, walk}.

## Data and scoring

- All annotated images of VOC-Sim (490) and COCO-Cap (707); both were used to
  develop BRACE.
- Class-macro mIoU; image bootstrap with 2000 resamples, 95% interval.
- **Primary:** threshold selected on VOC-Sim, with DenseCRF: 5-fold
  out-of-fold on VOC-Sim; on COCO-Cap, the full-VOC-Sim-best threshold
  unchanged.
- **Secondary:** fixed 0.55 with DenseCRF, and raw.
- Nothing is selected on COCO-Cap.

## Registered contrasts (primary convention, each seed)

| ID | Contrast |
|---|---|
| R-G | rule, source G: walk − hierarchy |
| R-N | rule, source N: walk − hierarchy |
| S-H | source, rule H (hierarchy): N − G |
| S-W | source, rule W (walk): N − G |
| I-G, I-N | interaction: (walk − hierarchy) on CLIP minus on CA, at each source |

**Reading rule (COCO-Cap, CLIP seed).**
- The walk's advantage on the CLIP seed is **carried by the rule** if R-G and
  R-N are both above zero (95% interval).
- It is **carried by the attention source** if S-H and S-W are both above
  zero.
- Both, one, or neither are reported as such.

The same contrasts on the CA seed and on VOC-Sim are reported without a rule.
They are descriptive, as is the comparison with runs/17's +0.033.

## Fidelity checks (reported)

- **Hook check, every image:** Q/K captured by the new hooks on BRACE's 64-px
  module at calls 10–20 must equal the frozen runtime's own record exactly, or
  the run stops.
- **CA × G × hierarchy:** must reproduce BRACE (about 0.830 / 0.696 at 0.55
  with DenseCRF); its maps are compared with the stored capture.
- **CLIP × N × walk vs runs/17's NERVE walk:** min–max map difference per
  image, on all images and on bit-identical regenerations. Expected to be
  small but not zero (different environment and library versions, generated
  rather than stored image).
- **Port check, smoke only:** before the full run, the ported fusion and walk
  are run in NERVE's own environment on NERVE's own capture for 6 images and
  compared with `Pipeline.refinement`.

## Caveats stated in advance

- The CLIP seed is computed on the stored image. On images whose regeneration
  is not bit-identical it is slightly misaligned with the generated image; the
  bit-identical subset is reported for the main contrasts.
- On G, NERVE's walk sees 11 averaged calls, while NERVE itself uses one pass.
  This is the natural way to give the walk BRACE's source; it is not the only
  one.
- The two rules use different layers; this is part of each rule as released
  and is not separated here.

## Files

- `scripts/jc_srcrule.py`: `capture` (GPU), `score` (CPU, sharded), `report`.
- `scripts/nerve_walk_port.py`: NERVE's fusion and walk, copied from its
  released code with side N's patch.
- `scripts/jc_srcrule_portcheck.py`: port check (NERVE environment).
- Maps: `<BRACE_CACHE_DIR>/jc_srcrule/<set>/<image id>.npz`.
- Report: `runs/19_source_rule/result.md`.
