# Ideas to try

Written 2026-11-16. This is a planning list, not a result. Nothing here is pre-registered or run. BRACE stays frozen; any idea that passes a screen still needs a confirmation on fresh images.

The prior-work notes come from one quick search per idea. Each idea needs a proper literature check before any novelty claim.

## What the evidence says

Sorted by outcome, the experiments so far follow one rule.

| | what it did | outcome |
|---|---|---|
| A3 seed | keeps cross-attention layers that agree with self-attention structure | worked |
| Capped hierarchy (V3) | fine scale may not exceed what the coarse scale supports | worked |
| CLIP seed; CLIP × cross-attention seed | two independent models must agree | worked (0.845 / 0.717; product seed 0.741 on COCO-Cap, exploratory) |
| OVAM learned tokens | a better class embedding, learned from labels | worked (+0.05 on VOC-Sim, older pipeline), but needs annotations |
| Features, VAE decoder, SA values, noise variability, flips and shifts | a new signal from the same model and the same image | all closed |
| Head weighting, purification, synonyms, contrast seeds, one-step token refinement | BRACE refining itself | all closed or stopped |
| Knockout, dose–response, CFG, Jacobian, chrono | intervening on the generation | all stopped, far below BRACE |

**Three facts to build on:**
- **Wins are consistency constraints between independent views.** Losses are more signal from the same view; BRACE's own confidence always already contained it.
- **Bleed is systematic, not noise.** It is stable across noise draws, flips and shifts, and it follows the class word's associations (dining table → plates). No per-image test can see it.
- **The seed is usually right; the propagation is what spreads.** On the Cityscapes-style images the seeds sat on the right class, then every class spread across its side of the horizon.

**So the unused levers are:**
- constraints between classes;
- statistics across images, where a systematic error becomes visible;
- a second model as an independent view;
- the fact that we control the generator.

## Ranking

| # | idea | lever | my odds | cost | training-free? |
|---|---|---|---|---|---|
| 1 | Class competition inside the capped hierarchy | between classes | good on scenes, uncertain on VOC-Sim | low | yes |
| 2 | Absent-class calibration | across images | moderate | low–medium | yes (statistics only) |
| 3 | Cross-backbone agreement seed (SD1.5 × SD2.1) | second model | moderate–good | medium | yes |
| 4 | Dataset-level self-distilled probe | across images and classes | moderate | medium | no labels, but fits a linear probe |
| 5 | Label-free annotator model over views | across images | low–moderate | low | yes (EM on statistics) |
| 6 | Inpainting predictability as a mask verifier | generator as judge | low–moderate | high | yes |
| 7 | Generate-to-agree | we control the generator | moderate, hard to evaluate | high | yes |
| 8 | Confirm what already nearly works | — | good | low–medium | mixed |

## 1. Class competition inside the capped hierarchy

**The gap.** BRACE spreads each class alone, normalises each map to its own 0–1 range and thresholds it. Nothing says a pixel belongs to only one class.

**The idea.**
- Run every class present in the image, plus a background slot, through the hierarchy together.
- At each coarse cell the classes split the mass between them.
- The cap then applies per class: a class's fine-scale share cannot exceed its coarse-scale share. BRACE's one-class cap becomes a partition across classes.
- The output is one label map by argmax, with no threshold and no per-class normalisation.

**Why it might work.**
- The Cityscapes-style masks failed in exactly this way: *sidewalk* covered the road and *person* covered the buildings.
- COCO-Cap's main error is false positives on other objects, which the score files already count separately.

**Risks.**
- VOC-Sim has one object per image, so competition has little to do there.
- An earlier attempt subtracted generic-word maps from the finished map: the mechanism showed, the practical gain did not. That was subtraction after propagation, not a partition inside it.
- The background slot needs a seed. The start-of-text token is the obvious candidate, and contrast against it lost precision before.

**Prior work.** [Dataset Diffusion](https://arxiv.org/abs/2309.14303) and [iSeg](https://arxiv.org/abs/2409.03209) take an argmax over per-class attention maps. The claim here would have to be the cross-scale partition constraint, not the argmax.

**First test.** The seeds and maps of all 269 Cityscapes-style queries are saved, so the joint version runs without new captures. Scoring needs ground truth for about 20 of those images. COCO-Cap's other-object false positives give a second test on existing labels.

## 2. Absent-class calibration

**The gap.** Bleed is a property of the class word, so it should show up wherever the word is asked, even when the object is not there.

**The idea.**
- We know from the prompts which classes are absent from each image, at no labelling cost.
- For class c, read its cross-attention on images where c is absent. Whatever it lights up there is spurious by construction: context the word is attracted to.
- Collect the image queries at those spurious peaks and summarise them per class, as a low-rank subspace or a few prototypes, per cross-attention layer.
- At test time, discount the part of c's attention explained by that spurious profile.

**Why it might work.**
- It is the first idea that uses the level where the error is visible: across images, per class.
- It is purification with the right nuisance set. The purification that failed used 8 generic words in text space and did worse than random. This one is class-specific, learned from data, and lives in the image-query space where the attention is computed.

**Risks.**
- A context that only ever appears with the object (plates with tables) may be rare in the absent set.
- Discounting can cost recall when object and context look alike.

**Prior work.**
- [ELBO-T2IAlign](https://arxiv.org/html/2506.09740v1) calibrates pixel–text alignment in diffusion models with a likelihood-based correction, and should be read first.
- [REFAM](https://arxiv.org/pdf/2509.22650v1) adds stop words to absorb background attention.
- [GoCA](https://arxiv.org/pdf/2603.06178) rescales token scores per pixel.
- None of the three appears to build a per-class profile from images where the class is absent.
- The weakly supervised literature treats co-occurring context as a confounder (CONTA); this would be a training-free version of that adjustment.

**First test.** The replay harness already captures the cross-attention queries. Build the profiles on VOC-Sim development images, where each image is a negative for 19 classes, then run the usual screen.

## 3. Cross-backbone agreement seed (SD1.5 × SD2.1)

**The gap.** The independent second view that worked was CLIP. A second diffusion model is also independent in the ways that matter: SD2.1 has a different text encoder and different training data.

**The idea.**
- Read the class word's cross-attention in SD2.1 on the same image, by noising the SD1.5 image and making single UNet calls.
- Combine the two seeds by geometric mean, the same rule that gave +0.024 on COCO-Cap with CLIP.
- Propagate with BRACE unchanged.

**Why it might work.**
- Agreement between independent models is the one pattern that has paid off every time.
- Reading attention from a re-noised image works: it cost about 0.015 with BRACE's own model.
- It doubles as the SD2.1 port JC asked for.

**Risks.**
- Both models learned from similar data, so association errors may be shared.
- If SD2.1's seed is much weaker, the product only removes recall.

**First test.** A seed-quality check for SD2.1 alone on the 150 + 150 development images, then the product seed against BRACE in one capture.

## 4. Dataset-level self-distilled probe

**The gap.** A classifier trained inside one image on BRACE's core against its background could not separate bleed (+0.003). Inside one image, the bleed looks like the object's neighbourhood. Across images it does not: plates are background in every image that is not about tables.

**The idea.**
- Fit one linear probe on frozen UNet features over the whole unlabelled set, with all classes and a background output.
- Train it only on BRACE's confident cells: cores as positives, clear background as negatives.
- Its prediction then gates or reseeds BRACE.

**Why it might work.** A systematic error in one class is contradicted by the other classes' labels, which is the correction that no per-image method can reach.

**Risks.**
- It is label-free but not training-free, which changes the paper's framing.
- The rich features exist for only 300 images, so it needs a new capture.
- If a context never appears as confident background elsewhere, the probe inherits the error.

**Prior work.** Training a student on diffusion pseudo-labels is common (DiffuMask, Dataset Diffusion). A linear probe on frozen features used to correct the labeller itself is a narrower claim.

## 5. Label-free annotator model over views

**The idea.**
- Treat each cross-attention layer, timestep block and model as a noisy annotator.
- Estimate each annotator's reliability per class from agreement patterns alone, with the classic Dawid–Skene EM used for crowd labels.
- Combine by the posterior.
- A3's layer weighting is a hand-built special case, estimated per image and per step; this would estimate reliabilities per class over the whole set.

**Why it might work.** COCO-Cap *person* is a seed failure (0.28, against 0.64 from CLIP), and a class-level reliability model could learn which layers fail there.

**Risk.** The method assumes annotators err independently. Layers of one model share their bleed, so the model may converge to the consensus, bleed included. It needs CLIP or SD2.1 among the annotators to break that.

## 6. Inpainting predictability as a mask verifier

**The idea.**
- A region is an object if it cannot be predicted from the rest of the image, and the rest cannot be predicted from it ([Inpainting Error Maximization](https://arxiv.org/abs/2012.07287), later [MOVE](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2022/move_unsupervised_movable_object_segmentation_and_detection__b9964daf.txt)).
- SD is a strong inpainter. Score candidate masks (BRACE at several thresholds, with and without suspect regions) by inpainting error, and keep the best.
- The search found no diffusion version of this criterion.

**Why it might work.** It judges the mask with the generator's image prior, not with attention, so it is an independent view.

**Risks.**
- It is expensive: several inpainting runs per candidate.
- Background clutter is also unpredictable, so the criterion may prefer large masks.

## 7. Generate-to-agree

**The idea.** For synthetic data, only the quality of the (image, mask) pair matters, not the image itself.
- During generation, compute BRACE's mask at the middle steps.
- Steer the remaining steps so the class word's attention stays inside it.
- The final image is then pushed to agree with its own label.

**Why it might work.** It removes the cause of bleed instead of detecting it afterwards.

**Risks.**
- The image changes, so the existing ground truth cannot score it. It needs new annotation or a downstream test (train a student, evaluate on real VOC).
- Attention guidance for layout control exists (Attend-and-Excite, BoxDiff, self-guidance); the new part is using the method's own mask as the target.

## 8. Confirm what already nearly works

- **CLIP × cross-attention product seed.** 0.741 on COCO-Cap (+0.024 over the CLIP seed, CI +0.010…+0.040), but it was not the registered pick. It needs a confirmatory test on fresh images.
- **BRACE with OVAM's optimised tokens.** JC asked for this. On the older pipeline the tokens gave +0.051 on VOC-Sim. It is a few-shot setting, reported separately.
- **Plain SD2.1 / SDXL port of the hierarchy.** JC asked for this too. SD2.1 has 24/48/96 grids and SDXL 32/64.

## Closed: do not retry without a new argument

- **Per-image bleed detection:** UNet features, VAE decoder ownership, SA value vectors, SA row divergence, noise variability at any noise level, flip and shift equivariance.
- **Self-refinement:** SA head weighting (seed agreement or entropy), nuisance purification of the class embedding, synonym ensembles, contrast seeds, one-step key refinement.
- **Generation interventions:** token knockout, dose–response, CFG differences, Jacobians, timestep regrouping.
- **Map-level fixes:** clutter-adaptive cap, generic-word subtraction, reliability scores from CLIP agreement, re-noising ensembles, flip and noise averaging.
