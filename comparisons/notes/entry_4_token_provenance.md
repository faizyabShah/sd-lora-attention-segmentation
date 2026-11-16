### Entry 5 — N (Narmeen), 2026-11-16
**Item:** #4 token provenance
**Kind:** result
**What was done:** Matched every per-image token file released with OVAM against the VOC-Sim and COCO-Cap evaluation sets. A token file is named `sd-15_<class>_<seed>_<step>.npy` and was optimized on the generated image `sd-15_<class>_<seed>.png` in `voc_sim_train/images/`. VOC-Sim and COCO-Cap images are identified by class and generation seed (VOC-Sim from `ovam_voc_sim.csv`, COCO-Cap from the annotation file names). The same prompt and seed give the same image, so a shared (class, seed) would mean a shared image.
**Where:** `datasets/ovam_experiment_with_dataset/evaluation/ovam/{voc_sim_train/tokens, voc_sim_train/images, voc_sim/ovam_voc_sim.csv, coco_captions/annotations}` (public OVAM archive, git-ignored). Check: `comparisons/notes/entry_4_token_provenance.md` (this file); the seed and pixel comparison is reproducible from the archive file names.
**Result:**
- 57 per-image tokens, 2–3 per class, for all 20 classes. Each token's seed is one of the 63 training images in `voc_sim_train/images`.
- The released class token is the mean of its class's per-image tokens. `scripts/run_token_package.py:92-99` already asserts this.
- **Overlap with VOC-Sim evaluation (500 csv rows, 20 classes): none.** No token seed equals any VOC-Sim evaluation seed, for the same class or for any class.
- **Overlap with COCO-Cap (707 annotated images): none.** No token seed equals any COCO-Cap seed. COCO-Cap images also come from caption prompts, not "A photograph of a <class>", so they could not be identical images even if a seed matched.
- **Pixel check:** none of the 63 training PNGs is pixel-identical to any of the 68 VOC-Sim evaluation PNGs held locally.
- **Conclusion:** the optimized tokens did not see any VOC-Sim or COCO-Cap evaluation image. They were trained on 57 separate, annotated generated images. The #4 block is therefore "few-shot (57 annotated images, 2–3 per class)", not "training-free", but it has no evaluation leakage.
**Slides:** `slides/jc_meeting/sections/c4_N.tex` (to add)
**Open points:** The pixel-level check covers only the 68 VOC-Sim evaluation PNGs held locally; the seed-level check covers all 500. Annotation masks of the training images are used by OVAM's optimisation, which is the reason the block is not training-free.
