# Handoff for Codex: runs/19 image-mismatch finding and runs/20 "new reruns" (2026-11-17)

Self-contained summary of what was found and run on the SLURM cluster since runs/17. All numbers are class-macro
mIoU. Nothing here changes the frozen BRACE method (A3 seed, V3 capped 16→32→64 hierarchy, DenseCRF R2, τ 0.55); runs/20
is a **data correction**, and its numbers supersede (do not silently replace) the earlier generated-set numbers.

## 1. Why the reruns were needed (runs/19 addendum, post hoc)

- runs/19's NERVE cell (COCO-Cap 0.720) disagreed with runs/17 / Narmeen's NERVE (0.737). Confirmed from NERVE's code
  that NERVE is a single noised pass: VAE mean latent, `add_noise` at scheduler index 45/50 (t = 101), empty prompt, one
  UNet call. Noise seed, fp16/fp32, GPU type and attention processor made no difference.
- The gap was the **input image**: runs/17 ran NERVE on the official OVAM PNGs; runs/19 ran it on our live regeneration.
  Our regenerations differed from the official images on 50/490 VOC-Sim and 101/707 COCO-Cap.
- Root cause (prompts, not library versions; OVAM's own stack torch 2.1.0 / diffusers 0.20.0 reproduces our images):
  - VOC-Sim: OVAM generated with "dining table" / "potted plant" (spaces); our manifest hyphenated them (all 50 wrong).
  - COCO-Cap: OVAM's captions kept a trailing period; our CSV stripped it (restoring "." fixes 86/101).
  - Our runtime left SD's safety checker on, which blacked 2 COCO outputs (224079 chair, 556255 dog). The checker runs
    after the denoising loop and VAE decode, so attention was unaffected; only DenseCRF and NERVE's noised pass saw black.
- Details: `runs/19_source_rule/addendum_image_mismatch.md`; per-image fixes `runs/19_source_rule/diag/prompt_fix_*.csv`.

## 2. What runs/20 did

- Pre-registration `runs/20_new_reruns/PREREG.md` (commit abb272c92, before any generation/scoring); `--ids` smoke option
  c4b88ad95; result 8baa528ad. Script `scripts/jc_newrerun.py` (gen / bank / capture / score / report); NERVE seed via
  `scripts/jc_swap_nerve.py seed --image-dir`. Corrected prompts `configs/new_reruns_prompts.json`, safety checker off.
- One live capture per image with the frozen runtime; all seeds and propagations come from the **same capture**.
- Seeds: **CAu** cross-attention, uniform layer weights (A0); **CAg** cross-attention, GOCA-style weights (A3 = BRACE
  seed); **NV** NERVE CLIP seed (ViT-L/14); **SC** SCLIP ViT-B/16 VOC21 probability.
- Propagation: seed alone / fine-only (V0) / BRACE hierarchy (V3) / NERVE walk (NERVE's layers on NERVE's noised pass).
- Each cell raw and +DenseCRF, at VOC-selected threshold (5-fold OOF on VOC-Sim; full-VOC-best applied to COCO-Cap) and
  fixed 0.55; COCO best-threshold reported as audit only. Image bootstrap 95% CIs (2000).
- Calibration arm (exploratory; monotone remaps of min–max CAg seed): hA (CAg ranking + per-image NERVE CLIP values),
  TG/TC (global / per-class VOC CLIP templates, fold-held-out), PWγ, FLp, TKk (one parameter per family picked by
  VOC-Sim OOF), FO (Otsu floor).

## 3. Headline

| | VOC-Sim | COCO-Cap |
|---|---:|---:|
| BRACE (CAg × hierarchy + CRF, 0.55) | 0.833 | 0.709 (old capture: 0.696) |
| NERVE (NV × walk, VOC-sel. + CRF) | 0.843 | 0.735 |
| SCLIP + BRACE hierarchy + CRF | 0.852 | 0.734 |
| hA hybrid + hierarchy + CRF (exploratory) | 0.851 | 0.735 |

Key readings:
- **NERVE still leads BRACE on COCO-Cap** after the data fix: +0.027 [+0.002, +0.054] (VOC-Sim +0.011 n.s.). An earlier
  claim that they tie on images matching the official PNG was subset-specific (528 COCO images) and does not hold on the
  full corrected set.
- Hierarchy − fine-only (BRACE seed), COCO: +0.036 [+0.025, +0.047] at VOC-sel., but +0.010 n.s. at fixed 0.55.
- GOCA − uniform weights (hierarchy): COCO +0.035, VOC-Sim +0.004 n.s.
- With NERVE's seed, NERVE's walk beats our hierarchy by 0.016 on COCO-Cap.
- DenseCRF adds +0.007 to +0.016 everywhere.
- Calibration: **only hA helps** (COCO +0.027 [+0.015, +0.039], VOC +0.020, best-threshold +0.017). Every CLIP-free
  remap (templates, power, floor, top-k, Otsu) fails → the gain needs CLIP's image-specific values; the SD-only
  calibration direction is closed.
- Remaining BRACE weak spots on COCO-Cap: person 0.28 (NERVE 0.71), small objects 0.60 (NERVE 0.71); CLIP seeds win
  mainly through fewer false positives (precision 0.81 vs 0.76).
- Data check: 18 VOC-Sim and 44 COCO-Cap images remain ≥ 2 grey levels from the official PNG (all < 5; no prompt
  issue found; likely numerical drift). Excluding them changes nothing material (sensitivity tables below).

## 4. Status / rules for whoever continues

- Committed locally only (latest 8baa528ad); **not pushed**.
- Not done: updating the paper and protocol with corrected numbers; checking whether other baselines were scored on our
  regenerated images rather than the official PNGs; any fresh confirmatory set for a new method.
- Rules: BRACE frozen; label-free inference; pre-register and commit before scoring; VOC-Sim for development, COCO-Cap
  untuned transfer; compare only within one capture; GPU work only through SLURM (never on the login node).

---

# Full results (verbatim copy of `runs/20_new_reruns/result.md`)

`scripts/jc_newrerun.py`. One live generation per image with the frozen runtime, corrected prompts (`configs/new_reruns_prompts.json`), safety checker off. Class-macro mIoU. *VOC-sel.* = threshold selected on VOC-Sim (5-fold OOF on VOC-Sim; full-VOC-Sim-best threshold unchanged on COCO-Cap). Δ with image-bootstrap 95% interval (2000). Seeds: **CAu** = cross-attention, uniform layer weights; **CAg** = cross-attention, GOCA-style weights (BRACE seed); **NV** = NERVE CLIP seed (ViT-L/14); **SC** = SCLIP seed (ViT-B/16).

### Data checks

- VOC-Sim: 490 images, 50 with corrected prompt. Generated vs official PNG: median 0.16 grey levels; < 1 on 452, unresolved (≥ 2) 18 (ovam-voc-sim-111, ovam-voc-sim-132, ovam-voc-sim-146, ovam-voc-sim-170, ovam-voc-sim-177, ovam-voc-sim-19, ovam-voc-sim-211, ovam-voc-sim-257, ovam-voc-sim-258, ovam-voc-sim-259, ovam-voc-sim-269, ovam-voc-sim-28, ovam-voc-sim-281, ovam-voc-sim-304, ovam-voc-sim-31, ovam-voc-sim-314, ovam-voc-sim-469, ovam-voc-sim-480). Capture vs gen step image: max |Δ| 0 on 245/490. Unchanged-prompt images: BRACE map vs stored paper capture median max |Δ| 0.0e+00. NERVE walk vs runs/17 (min–max), images < 1 grey level from official: median max |Δ| 1.8e-02.
- COCO-Cap: 707 images, 86 with corrected prompt. Generated vs official PNG: median 0.21 grey levels; < 1 on 618, unresolved (≥ 2) 44 (ovam-coco-106137-dog-313399030, ovam-coco-114309-bird-3099876533, ovam-coco-166787-train-1839819348, ovam-coco-170164-horse-505864827, ovam-coco-193219-dog-358443573, ovam-coco-193945-dining_table-981272952, ovam-coco-226838-bicycle-4075665076, ovam-coco-254782-motorbike-21976821, ovam-coco-273132-motorbike-2763104245, ovam-coco-296706-train-595262621, ovam-coco-315805-cat-1599336505, ovam-coco-31739-aeroplane-1476606686, ovam-coco-325300-dog-2241101716, ovam-coco-330889-dog-144693040, ovam-coco-381273-aeroplane-3983542586, ovam-coco-386960-chair-2210794860, ovam-coco-395897-bicycle-4266339948, ovam-coco-418199-chair-2511252585, ovam-coco-433064-cat-1322189636, ovam-coco-441941-sheep-2523596243, ovam-coco-446728-aeroplane-2826500723, ovam-coco-470035-person-3542956483, ovam-coco-472470-aeroplane-3198147258, ovam-coco-472976-person-4017470167, ovam-coco-48448-person-811552420, ovam-coco-488074-bicycle-1514556934, ovam-coco-498906-bicycle-2727345106, ovam-coco-537294-aeroplane-581228670, ovam-coco-566695-dog-1071257360, ovam-coco-576973-dog-70278584, ovam-coco-587791-bus-4231869002, ovam-coco-608007-person-1101767043, ovam-coco-61662-chair-4115572993, ovam-coco-622393-car-3336138235, ovam-coco-645843-person-1762850619, ovam-coco-726311-bus-1258085996, ovam-coco-731512-bicycle-3965643736, ovam-coco-73382-dog-3660910269, ovam-coco-736922-train-899624767, ovam-coco-757365-motorbike-2638166565, ovam-coco-770829-car-2343354407, ovam-coco-796622-boat-3048250845, ovam-coco-804598-car-187324739, ovam-coco-816453-dog-3985787832). Capture vs gen step image: max |Δ| 0 on 177/707. Unchanged-prompt images: BRACE map vs stored paper capture median max |Δ| 1.8e-03. NERVE walk vs runs/17 (min–max), images < 1 grey level from official: median max |Δ| 1.9e-02.

### Components, VOC-Sim (490 images)

**VOC-sel. threshold, with DenseCRF (primary)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.748 | 0.820 | 0.828 | 0.827 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.736 | 0.824 | 0.832 | 0.831 |
| NERVE CLIP seed (ViT-L/14) | 0.761 | 0.831 | 0.840 | 0.843 |
| SCLIP seed (ViT-B/16) | 0.757 | 0.838 | 0.852 | 0.836 |

**fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.603 | 0.786 | 0.829 | 0.751 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.628 | 0.797 | 0.833 | 0.759 |
| NERVE CLIP seed (ViT-L/14) | 0.752 | 0.832 | 0.836 | 0.840 |
| SCLIP seed (ViT-B/16) | 0.667 | 0.814 | 0.852 | 0.787 |

**VOC-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.715 | 0.807 | 0.815 | 0.814 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.706 | 0.811 | 0.818 | 0.820 |
| NERVE CLIP seed (ViT-L/14) | 0.727 | 0.819 | 0.828 | 0.831 |
| SCLIP seed (ViT-B/16) | 0.714 | 0.820 | 0.842 | 0.823 |

**fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.591 | 0.782 | 0.815 | 0.751 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.613 | 0.792 | 0.819 | 0.758 |
| NERVE CLIP seed (ViT-L/14) | 0.721 | 0.818 | 0.820 | 0.832 |
| SCLIP seed (ViT-B/16) | 0.638 | 0.810 | 0.838 | 0.789 |

**Best threshold on this set, with DenseCRF (audit only)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.749 | 0.821 | 0.829 | 0.827 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.736 | 0.826 | 0.833 | 0.833 |
| NERVE CLIP seed (ViT-L/14) | 0.761 | 0.832 | 0.841 | 0.843 |
| SCLIP seed (ViT-B/16) | 0.757 | 0.838 | 0.852 | 0.839 |

### Components, COCO-Cap (707 images)

**VOC-sel. threshold, with DenseCRF (primary)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.616 | 0.658 | 0.674 | 0.646 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.629 | 0.673 | 0.709 | 0.675 |
| NERVE CLIP seed (ViT-L/14) | 0.692 | 0.713 | 0.719 | 0.735 |
| SCLIP seed (ViT-B/16) | 0.619 | 0.712 | 0.734 | 0.715 |

**fixed 0.55, with DenseCRF**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.458 | 0.669 | 0.684 | 0.653 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.527 | 0.698 | 0.709 | 0.679 |
| NERVE CLIP seed (ViT-L/14) | 0.684 | 0.719 | 0.708 | 0.743 |
| SCLIP seed (ViT-B/16) | 0.498 | 0.669 | 0.735 | 0.637 |

**VOC-sel. threshold, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.572 | 0.643 | 0.670 | 0.645 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.592 | 0.670 | 0.693 | 0.675 |
| NERVE CLIP seed (ViT-L/14) | 0.647 | 0.714 | 0.712 | 0.737 |
| SCLIP seed (ViT-B/16) | 0.582 | 0.696 | 0.725 | 0.699 |

**fixed 0.55, raw**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.455 | 0.662 | 0.670 | 0.651 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.510 | 0.690 | 0.693 | 0.676 |
| NERVE CLIP seed (ViT-L/14) | 0.639 | 0.707 | 0.692 | 0.737 |
| SCLIP seed (ViT-B/16) | 0.483 | 0.667 | 0.723 | 0.640 |

**Best threshold on this set, with DenseCRF (audit only)**

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| cross-attention, uniform layer weights | 0.616 | 0.671 | 0.697 | 0.671 |
| cross-attention, GOCA-style weights (BRACE seed) | 0.629 | 0.702 | 0.719 | 0.698 |
| NERVE CLIP seed (ViT-L/14) | 0.692 | 0.728 | 0.722 | 0.746 |
| SCLIP seed (ViT-B/16) | 0.619 | 0.712 | 0.735 | 0.719 |

**Headline:** BRACE (CAg × hierarchy + DenseCRF, 0.55): VOC-Sim 0.833, COCO-Cap 0.709; NERVE (NV × walk, VOC-sel. + DenseCRF): VOC-Sim 0.843, COCO-Cap 0.735.

### Thresholds selected on full VOC-Sim (with DenseCRF)

| Seed | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---:|---:|---:|---:|
| CAu | 0.400 | 0.450 | 0.525 | 0.375 |
| CAg | 0.400 | 0.425 | 0.550 | 0.375 |
| NV | 0.650 | 0.525 | 0.600 | 0.500 |
| SC | 0.300 | 0.400 | 0.575 | 0.325 |
| hA | 0.625 | 0.525 | 0.600 | 0.525 |
| TG | 0.525 | 0.500 | 0.575 | 0.475 |
| TC | 0.550 | 0.475 | 0.550 | 0.450 |
| PW0.5 | 0.625 | 0.525 | 0.600 | 0.500 |
| PW1.5 | 0.300 | 0.375 | 0.500 | 0.300 |
| PW2 | 0.300 | 0.325 | 0.475 | 0.300 |
| PW3 | 0.300 | 0.300 | 0.425 | 0.300 |
| FL50 | 0.300 | 0.300 | 0.450 | 0.300 |
| FL70 | 0.300 | 0.300 | 0.375 | 0.300 |
| FL80 | 0.300 | 0.300 | 0.325 | 0.300 |
| FL90 | 0.300 | 0.300 | 0.300 | 0.300 |
| TK5 | 0.300 | 0.300 | 0.300 | 0.300 |
| TK10 | 0.300 | 0.300 | 0.300 | 0.300 |
| TK20 | 0.300 | 0.300 | 0.375 | 0.300 |
| TK40 | 0.425 | 0.350 | 0.475 | 0.300 |
| FO | 0.300 | 0.300 | 0.400 | 0.300 |

### Registered contrasts

| ID | Contrast | VOC-Sim, VOC-sel. + CRF | COCO-Cap, VOC-sel. + CRF | VOC-Sim, 0.55 + CRF | COCO-Cap, 0.55 + CRF |
|---|---|---|---|---|---|
| F | fine-only − seed, CAu | +0.072 [+0.062, +0.084] | +0.042 [+0.027, +0.058] | +0.183 [+0.168, +0.197] | +0.210 [+0.189, +0.233] |
| F | fine-only − seed, CAg | +0.088 [+0.077, +0.101] | +0.043 [+0.028, +0.060] | +0.169 [+0.156, +0.183] | +0.171 [+0.152, +0.190] |
| F | fine-only − seed, NV | +0.071 [+0.061, +0.080] | +0.021 [+0.002, +0.039] | +0.080 [+0.071, +0.090] | +0.035 [+0.023, +0.049] |
| F | fine-only − seed, SC | +0.081 [+0.072, +0.090] | +0.093 [+0.077, +0.109] | +0.147 [+0.135, +0.161] | +0.172 [+0.153, +0.191] |
| H | hierarchy − fine-only, CAu | +0.008 [+0.001, +0.014] | +0.016 [+0.005, +0.028] | +0.044 [+0.032, +0.054] | +0.015 [-0.001, +0.030] |
| H | hierarchy − fine-only, CAg | +0.007 [+0.001, +0.013] | +0.036 [+0.025, +0.047] | +0.036 [+0.025, +0.046] | +0.010 [-0.004, +0.026] |
| H | hierarchy − fine-only, NV | +0.008 [-0.000, +0.016] | +0.006 [-0.003, +0.015] | +0.004 [-0.005, +0.014] | -0.012 [-0.025, +0.001] |
| H | hierarchy − fine-only, SC | +0.013 [+0.005, +0.022] | +0.022 [+0.012, +0.032] | +0.037 [+0.025, +0.049] | +0.066 [+0.047, +0.082] |
| G | CAg − CAu (GOCA weighting), seed alone | -0.011 [-0.017, -0.006] | +0.013 [+0.003, +0.023] | +0.025 [+0.016, +0.033] | +0.069 [+0.057, +0.079] |
| G | CAg − CAu (GOCA weighting), fine-only | +0.004 [-0.002, +0.011] | +0.015 [+0.004, +0.026] | +0.011 [+0.005, +0.017] | +0.030 [+0.016, +0.042] |
| G | CAg − CAu (GOCA weighting), BRACE hierarchy | +0.004 [-0.002, +0.010] | +0.035 [+0.023, +0.046] | +0.003 [-0.002, +0.009] | +0.025 [+0.014, +0.036] |
| C | NV − CAg, seed alone | +0.024 [+0.012, +0.037] | +0.063 [+0.041, +0.087] | +0.124 [+0.107, +0.138] | +0.157 [+0.135, +0.179] |
| C | NV − CAg, fine-only | +0.007 [-0.008, +0.022] | +0.040 [+0.017, +0.065] | +0.035 [+0.019, +0.051] | +0.021 [-0.001, +0.045] |
| C | NV − CAg, BRACE hierarchy | +0.008 [-0.006, +0.020] | +0.010 [-0.011, +0.034] | +0.004 [-0.010, +0.016] | -0.001 [-0.023, +0.022] |
| C | NV − CAg, NERVE walk | +0.012 [-0.001, +0.027] | +0.060 [+0.037, +0.084] | +0.081 [+0.064, +0.097] | +0.064 [+0.043, +0.090] |
| C | SC − CAg, seed alone | +0.021 [+0.006, +0.036] | -0.010 [-0.035, +0.017] | +0.040 [+0.021, +0.057] | -0.029 [-0.057, -0.001] |
| C | SC − CAg, fine-only | +0.014 [-0.001, +0.027] | +0.040 [+0.013, +0.065] | +0.018 [-0.001, +0.036] | -0.029 [-0.057, -0.001] |
| C | SC − CAg, BRACE hierarchy | +0.020 [+0.006, +0.033] | +0.025 [-0.001, +0.052] | +0.019 [+0.006, +0.031] | +0.026 [+0.001, +0.052] |
| C | SC − CAg, NERVE walk | +0.005 [-0.010, +0.020] | +0.039 [+0.013, +0.065] | +0.028 [+0.009, +0.046] | -0.042 [-0.070, -0.012] |
| R | DenseCRF − raw, hierarchy, CAu | +0.013 [+0.010, +0.016] | +0.003 [+0.001, +0.007] | +0.014 [+0.011, +0.017] | +0.013 [+0.012, +0.016] |
| R | DenseCRF − raw, hierarchy, CAg | +0.014 [+0.010, +0.018] | +0.016 [+0.014, +0.018] | +0.013 [+0.011, +0.016] | +0.016 [+0.014, +0.018] |
| R | DenseCRF − raw, hierarchy, NV | +0.011 [+0.008, +0.015] | +0.007 [+0.004, +0.011] | +0.016 [+0.014, +0.019] | +0.016 [+0.014, +0.019] |
| R | DenseCRF − raw, hierarchy, SC | +0.010 [+0.007, +0.012] | +0.009 [+0.006, +0.013] | +0.014 [+0.011, +0.017] | +0.012 [+0.010, +0.014] |
| N | NERVE (NV × walk) − BRACE (CAg × hierarchy) | +0.011 [-0.003, +0.026] | +0.027 [+0.002, +0.054] | +0.007 [-0.007, +0.022] | +0.035 [+0.010, +0.063] |
| N | BRACE hierarchy on NERVE's seed (NV × hierarchy) − NERVE | -0.003 [-0.012, +0.005] | -0.016 [-0.030, -0.004] | -0.004 [-0.013, +0.006] | -0.036 [-0.053, -0.019] |

Sensitivity, resolved images only (generated within 2 grey levels of the official PNG), VOC-sel. + CRF:

| Contrast | VOC-Sim | COCO-Cap |
|---|---|---|
| H: hierarchy − fine-only, CAu | +0.008 [+0.001, +0.015] | +0.016 [+0.005, +0.028] |
| H: hierarchy − fine-only, CAg | +0.007 [-0.000, +0.013] | +0.039 [+0.027, +0.050] |
| H: hierarchy − fine-only, NV | +0.008 [+0.000, +0.017] | +0.003 [-0.006, +0.013] |
| H: hierarchy − fine-only, SC | +0.014 [+0.006, +0.022] | +0.021 [+0.011, +0.032] |
| G: CAg − CAu (GOCA weighting), seed alone | -0.011 [-0.017, -0.006] | +0.014 [+0.003, +0.024] |
| G: CAg − CAu (GOCA weighting), fine-only | +0.005 [-0.001, +0.012] | +0.014 [+0.004, +0.024] |
| G: CAg − CAu (GOCA weighting), BRACE hierarchy | +0.004 [-0.002, +0.009] | +0.037 [+0.025, +0.048] |
| N: NERVE (NV × walk) − BRACE (CAg × hierarchy) | +0.011 [-0.004, +0.025] | +0.029 [+0.006, +0.056] |
| N: BRACE hierarchy on NERVE's seed (NV × hierarchy) − NERVE | -0.003 [-0.012, +0.006] | -0.021 [-0.034, -0.009] |

Component table on resolved images only (VOC-sel. + CRF):

| Seed | set | seed alone | fine-only | BRACE hierarchy | NERVE walk |
|---|---|---:|---:|---:|---:|
| CAu | VOC-Sim | 0.746 | 0.817 | 0.825 | 0.825 |
| CAg | VOC-Sim | 0.735 | 0.822 | 0.829 | 0.829 |
| NV | VOC-Sim | 0.758 | 0.828 | 0.836 | 0.839 |
| SC | VOC-Sim | 0.757 | 0.834 | 0.848 | 0.833 |
| CAu | COCO-Cap | 0.619 | 0.663 | 0.678 | 0.651 |
| CAg | COCO-Cap | 0.633 | 0.677 | 0.715 | 0.681 |
| NV | COCO-Cap | 0.698 | 0.721 | 0.724 | 0.745 |
| SC | COCO-Cap | 0.624 | 0.717 | 0.738 | 0.720 |

### Calibration arm (exploratory)

- PW: VOC-Sim OOF (hierarchy + CRF) PW0.5 0.829, PW1.5 0.826, PW2 0.822, PW3 0.811 → **PW0.5**
- FL: VOC-Sim OOF (hierarchy + CRF) FL50 0.794, FL70 0.782, FL80 0.773, FL90 0.757 → **FL50**
- TK: VOC-Sim OOF (hierarchy + CRF) TK5 0.753, TK10 0.769, TK20 0.783, TK40 0.802 → **TK40**

**fine-only, with DenseCRF**

| Seed | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | COCO-Cap 0.55 | COCO-Cap best thr. | Δ vs BRACE, VOC-Sim | Δ vs BRACE, COCO-Cap | Δ best-thr., COCO-Cap |
|---|---:|---:|---:|---:|---|---|---|
| CAg: cross-attention, GOCA-style weights (BRACE seed) | 0.824 | 0.673 | 0.698 | 0.702 | -0.007 [-0.013, -0.001] | -0.036 [-0.047, -0.025] | -0.016 |
| hA: CAg ranking + per-image NERVE CLIP values | 0.851 | 0.718 | 0.721 | 0.723 | +0.019 [+0.008, +0.030] | +0.009 [-0.007, +0.025] | +0.004 |
| TG: global VOC template (CLIP offline) | 0.770 | 0.560 | 0.589 | 0.623 | -0.061 [-0.073, -0.049] | -0.148 [-0.167, -0.131] | -0.096 |
| TC: per-class VOC template (CLIP offline) | 0.796 | 0.543 | 0.588 | 0.615 | -0.036 [-0.048, -0.023] | -0.165 [-0.184, -0.147] | -0.104 |
| PW0.5: power s^0.5 (SD only) | 0.823 | 0.665 | 0.678 | 0.694 | -0.009 [-0.016, -0.002] | -0.044 [-0.055, -0.033] | -0.025 |
| FL50: percentile floor p50 (SD only) | 0.768 | 0.649 | 0.632 | 0.673 | -0.063 [-0.075, -0.051] | -0.060 [-0.074, -0.045] | -0.046 |
| TK40: top 40% (SD only) | 0.783 | 0.641 | 0.666 | 0.677 | -0.048 [-0.058, -0.038] | -0.068 [-0.080, -0.055] | -0.042 |
| FO: Otsu floor (SD only) | 0.793 | 0.691 | 0.576 | 0.692 | -0.038 [-0.049, -0.027] | -0.018 [-0.034, -0.001] | -0.026 |

**BRACE hierarchy, with DenseCRF**

| Seed | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | COCO-Cap 0.55 | COCO-Cap best thr. | Δ vs BRACE, VOC-Sim | Δ vs BRACE, COCO-Cap | Δ best-thr., COCO-Cap |
|---|---:|---:|---:|---:|---|---|---|
| CAg: cross-attention, GOCA-style weights (BRACE seed) | 0.832 | 0.709 | 0.709 | 0.719 | — | — | — |
| hA: CAg ranking + per-image NERVE CLIP values | 0.851 | 0.735 | 0.726 | 0.735 | +0.020 [+0.011, +0.029] | +0.027 [+0.015, +0.039] | +0.017 |
| TG: global VOC template (CLIP offline) | 0.799 | 0.639 | 0.623 | 0.668 | -0.033 [-0.042, -0.023] | -0.069 [-0.084, -0.056] | -0.051 |
| TC: per-class VOC template (CLIP offline) | 0.818 | 0.622 | 0.622 | 0.664 | -0.014 [-0.024, -0.004] | -0.087 [-0.102, -0.072] | -0.054 |
| PW0.5: power s^0.5 (SD only) | 0.829 | 0.692 | 0.664 | 0.707 | -0.002 [-0.007, +0.003] | -0.017 [-0.023, -0.011] | -0.012 |
| FL50: percentile floor p50 (SD only) | 0.794 | 0.687 | 0.703 | 0.703 | -0.038 [-0.048, -0.027] | -0.022 [-0.033, -0.011] | -0.015 |
| TK40: top 40% (SD only) | 0.802 | 0.681 | 0.696 | 0.697 | -0.029 [-0.038, -0.020] | -0.027 [-0.036, -0.019] | -0.022 |
| FO: Otsu floor (SD only) | 0.808 | 0.702 | 0.708 | 0.715 | -0.023 [-0.031, -0.015] | -0.006 [-0.017, +0.006] | -0.004 |

**NERVE walk, with DenseCRF**

| Seed | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | COCO-Cap 0.55 | COCO-Cap best thr. | Δ vs BRACE, VOC-Sim | Δ vs BRACE, COCO-Cap | Δ best-thr., COCO-Cap |
|---|---:|---:|---:|---:|---|---|---|
| CAg: cross-attention, GOCA-style weights (BRACE seed) | 0.831 | 0.675 | 0.679 | 0.698 | -0.001 [-0.011, +0.009] | -0.033 [-0.045, -0.020] | -0.021 |
| hA: CAg ranking + per-image NERVE CLIP values | 0.853 | 0.727 | 0.728 | 0.728 | +0.021 [+0.008, +0.034] | +0.019 [-0.001, +0.039] | +0.009 |
| TG: global VOC template (CLIP offline) | 0.745 | 0.537 | 0.578 | 0.614 | -0.087 [-0.102, -0.071] | -0.172 [-0.192, -0.154] | -0.105 |
| TC: per-class VOC template (CLIP offline) | 0.784 | 0.525 | 0.577 | 0.609 | -0.048 [-0.063, -0.033] | -0.183 [-0.202, -0.164] | -0.110 |
| PW0.5: power s^0.5 (SD only) | 0.830 | 0.665 | 0.687 | 0.691 | -0.002 [-0.013, +0.008] | -0.043 [-0.055, -0.031] | -0.027 |
| FL50: percentile floor p50 (SD only) | 0.738 | 0.666 | 0.574 | 0.666 | -0.094 [-0.109, -0.077] | -0.042 [-0.060, -0.024] | -0.052 |
| TK40: top 40% (SD only) | 0.769 | 0.645 | 0.642 | 0.667 | -0.062 [-0.074, -0.049] | -0.063 [-0.077, -0.050] | -0.052 |
| FO: Otsu floor (SD only) | 0.747 | 0.670 | 0.489 | 0.670 | -0.085 [-0.097, -0.072] | -0.039 [-0.060, -0.019] | -0.049 |

**Reading (hierarchy + CRF vs BRACE; helps = COCO Δ CI > 0, VOC Δ ≥ 0, COCO best-threshold Δ > 0):**

- hA (CAg ranking + per-image NERVE CLIP values): COCO +0.027 (lower +0.015), VOC +0.020, COCO best-thr. +0.017 → **helps**
- TG (global VOC template (CLIP offline)): COCO -0.069 (lower -0.084), VOC -0.033, COCO best-thr. -0.051 → **does not help**
- TC (per-class VOC template (CLIP offline)): COCO -0.087 (lower -0.102), VOC -0.014, COCO best-thr. -0.054 → **does not help**
- PW0.5 (power s^0.5 (SD only)): COCO -0.017 (lower -0.023), VOC -0.002, COCO best-thr. -0.012 → **does not help**
- FL50 (percentile floor p50 (SD only)): COCO -0.022 (lower -0.033), VOC -0.038, COCO best-thr. -0.015 → **does not help**
- TK40 (top 40% (SD only)): COCO -0.027 (lower -0.036), VOC -0.029, COCO best-thr. -0.022 → **does not help**
- FO (Otsu floor (SD only)): COCO -0.006 (lower -0.017), VOC -0.023, COCO best-thr. -0.004 → **does not help**

**COCO-Cap failure modes, VOC-sel. + CRF (descriptive)**

| Seed | cell | mIoU | small | middle | large | < 0.3 IoU | FP area / image | precision | recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAg | BRACE hierarchy | 0.709 | 0.602 | 0.780 | 0.821 | 0.105 | 0.0673 | 0.763 | 0.915 |
| CAg | NERVE walk | 0.675 | 0.570 | 0.740 | 0.797 | 0.123 | 0.0885 | 0.722 | 0.922 |
| NV | BRACE hierarchy | 0.719 | 0.669 | 0.776 | 0.822 | 0.075 | 0.0527 | 0.782 | 0.894 |
| NV | NERVE walk | 0.735 | 0.709 | 0.792 | 0.813 | 0.062 | 0.0442 | 0.809 | 0.885 |
| SC | BRACE hierarchy | 0.734 | 0.644 | 0.771 | 0.799 | 0.092 | 0.0330 | 0.854 | 0.845 |
| SC | NERVE walk | 0.715 | 0.631 | 0.753 | 0.780 | 0.100 | 0.0408 | 0.832 | 0.843 |
| hA | BRACE hierarchy | 0.735 | 0.651 | 0.783 | 0.821 | 0.082 | 0.0515 | 0.804 | 0.898 |
| hA | NERVE walk | 0.727 | 0.661 | 0.776 | 0.792 | 0.069 | 0.0423 | 0.817 | 0.868 |
| TG | BRACE hierarchy | 0.639 | 0.473 | 0.747 | 0.810 | 0.153 | 0.0921 | 0.692 | 0.912 |
| TG | NERVE walk | 0.537 | 0.308 | 0.618 | 0.771 | 0.231 | 0.1421 | 0.589 | 0.892 |
| TC | BRACE hierarchy | 0.622 | 0.453 | 0.722 | 0.801 | 0.168 | 0.1058 | 0.670 | 0.918 |
| TC | NERVE walk | 0.525 | 0.313 | 0.585 | 0.743 | 0.253 | 0.1695 | 0.572 | 0.902 |
| PW0.5 | BRACE hierarchy | 0.692 | 0.575 | 0.772 | 0.821 | 0.117 | 0.0722 | 0.746 | 0.912 |
| PW0.5 | NERVE walk | 0.665 | 0.558 | 0.735 | 0.792 | 0.129 | 0.0872 | 0.718 | 0.911 |
| FL50 | BRACE hierarchy | 0.687 | 0.589 | 0.769 | 0.795 | 0.106 | 0.0710 | 0.753 | 0.900 |
| FL50 | NERVE walk | 0.666 | 0.598 | 0.774 | 0.739 | 0.115 | 0.0526 | 0.783 | 0.824 |
| TK40 | BRACE hierarchy | 0.681 | 0.562 | 0.765 | 0.810 | 0.122 | 0.0774 | 0.737 | 0.915 |
| TK40 | NERVE walk | 0.645 | 0.515 | 0.725 | 0.778 | 0.141 | 0.0855 | 0.713 | 0.884 |
| FO | BRACE hierarchy | 0.702 | 0.621 | 0.777 | 0.798 | 0.096 | 0.0674 | 0.766 | 0.905 |
| FO | NERVE walk | 0.670 | 0.660 | 0.762 | 0.708 | 0.103 | 0.0322 | 0.839 | 0.768 |

### Per class, VOC-sel. + CRF (CAg × hierarchy / NV × walk / SC × hierarchy)

VOC-Sim: aeroplane 0.88 / 0.86 / 0.91, bicycle 0.80 / 0.78 / 0.80, bird 0.92 / 0.92 / 0.93, boat 0.85 / 0.86 / 0.85, bottle 0.87 / 0.79 / 0.85, bus 0.92 / 0.88 / 0.91, car 0.85 / 0.82 / 0.86, cat 0.85 / 0.92 / 0.92, chair 0.79 / 0.84 / 0.80, cow 0.90 / 0.96 / 0.96, dining-table 0.43 / 0.43 / 0.50, dog 0.93 / 0.95 / 0.94, horse 0.90 / 0.93 / 0.93, motorbike 0.87 / 0.86 / 0.86, person 0.81 / 0.76 / 0.84, potted-plant 0.75 / 0.88 / 0.83, sheep 0.76 / 0.94 / 0.97, sofa 0.96 / 0.91 / 0.89, train 0.74 / 0.88 / 0.81, tv 0.85 / 0.67 / 0.67

COCO-Cap: aeroplane 0.80 / 0.64 / 0.73, bicycle 0.63 / 0.72 / 0.74, bird 0.78 / 0.83 / 0.82, boat 0.59 / 0.69 / 0.56, bottle 0.57 / 0.61 / 0.75, bus 0.92 / 0.84 / 0.85, car 0.76 / 0.76 / 0.73, cat 0.92 / 0.92 / 0.91, chair 0.48 / 0.69 / 0.67, cow 0.87 / 0.91 / 0.89, dining table 0.48 / 0.61 / 0.51, dog 0.81 / 0.84 / 0.83, horse 0.72 / 0.73 / 0.77, motorbike 0.71 / 0.74 / 0.75, person 0.28 / 0.71 / 0.66, potted plant 0.74 / 0.85 / 0.62, sheep 0.87 / 0.92 / 0.90, sofa 0.70 / 0.74 / 0.72, train 0.75 / 0.70 / 0.66, tv 0.79 / 0.25 / 0.60

