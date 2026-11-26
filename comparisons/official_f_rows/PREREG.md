# JC #3, remaining comparison rows (OVAM class word, OVAM optimized token, DAAM, SCLIP, SeeDiff): pre-registration

Side N, 2026-11-19, written before any capture of this run. Owner request: finish every experiment of JC's #3 today.
These rows were side F's in the workplan split (#3 "F" column); side N runs them because F has not started them.
No file of side F is changed: F's committed modules are imported unchanged; all new code is in
`comparisons/official_f_rows/`, all outputs in `~/TRDP/external/n_side/official_f_rows/` (git-ignored).
BRACE is not changed and not recomputed: its maps and counts are the N-side reproduction of runs/20
(`comparisons/official/`, `~/TRDP/external/n_side/brace_repro/`, job 253750).

## Images, pairing and query (workplan Section 3)

- 490 VOC-Sim + 707 COCO-Cap items, runs/20 corrected prompts (`configs/new_reruns_prompts.json`), manifest seeds,
  safety checker off, SD 1.5 fp32, 30 steps, CFG 7.5: the generation of `src/attention_capture/live.py`
  (`generate_attention_sample`) replicated call for call, with the same pipeline load as the frozen runtime.
- **Same images as BRACE**: one generation per item on a V100 (wald, the node of the BRACE reproduction), with
  all generation-time methods read from that single trajectory. The generated image is compared with the BRACE
  reproduction's stored image; the count of bit-identical images is reported. If an image differs, that item is
  flagged and a sensitivity row restricted to identical images is reported.
- Query: the class word only (corrected class word for the two VOC multi-word classes). Ground truth: the BRACE
  reproduction's mask for the same id (identical to the OVAM annotation PNG, checked in `comparisons/official/`).

## Methods

| Row | Source | Map | Own binarisation |
|---|---|---|---|
| OVAM, class word | OVAM's released method as mirrored in `src/attention_capture/memory.py` + `src/segmentation/ovam_full.py` (F's committed reproduction) | CA: unconditional query × keys of the separately encoded, unpadded attribution text (VOC: the prompt, target position len−2; COCO: the class name, len−2), softmax over that text, summed over heads, all non-mid cross-attention modules, all 30 steps, each resized to 64². SA: OVAM's all-ones-value self-attention reduction on the 64² self-attention modules. Fused = CA/max × SA scaled to [0.85, 1] at 512² (`fuse_ovam_maps`) | fused > 0.40, then DenseCRF (OVAM's pseudo-mask; `ovam_full_prediction`) |
| OVAM, optimized token (few-shot, **not training-free**) | as above, keys from [SoT output, OVAM's released VOC-Sim token] (`configs/ovam/tokens_VOC.npy`), target position 1; the same VOC tokens on COCO-Cap (as OVAM) | same fusion | fused > 0.40, then DenseCRF |
| DAAM | castorini/daam v0.2.0 (`daam/trace.py`, `heatmap.py`, `utils.py`), aggregation reimplemented in the same pass (its package pins diffusers 0.21.2, which would change the generation): conditional query × the 77 padded prompt keys, softmax, the class word's prompt token(s) (`compute_token_merge_indices`), every non-mid cross-attention module except the 8² ones (DAAM skips factor 8), summed over steps, each (module, head) bicubic to 64² and clamped at 0, mean over (module, head), mean over the word's tokens; `expand_as`: bicubic to 512², min-max | > 0.4 (DAAM's τ, also OVAM's DAAM evaluation) |
| SCLIP + DenseCRF | F's committed SCLIP port (`scripts/sg_clip_dense.py`, ViT-B/16), as used for the runs/20 SCLIP seed: VOC21 vocabulary, `postprocess(…, 65, 0.1, 0.1)`, target-class probability at 512² | — | SCLIP's argmax == target class |
| SeeDiff | public code search (paper, GitHub, project page) | go / no-go entry | — |

DAAM is defined only when the class word occurs in the prompt. VOC-Sim: all 490. COCO-Cap: the items whose caption
contains the class word (DAAM's tokenizer search); the DAAM COCO-Cap row and its paired Δ use that subset, with
BRACE on the same subset. OVAM's archive scored DAAM on all 707 with OVAM's own class-name maps; that is OVAM,
not DAAM, and is not reproduced as a DAAM row.

A DAAM equivalence check against the real `daam` package (its own environment, on a few VOC-Sim items) is run if
time allows; the map correlation is reported, not used for any choice.

## Scoring (unchanged from `comparisons/official/`)

- Soft maps min-max scaled (`sg_steps_test.mm`), thresholded on the 27-value grid `THR_EXT`; raw and DenseCRF R2
  (`apply_ovam_densecrf`, frozen OVAM parameters); counts with `scripts/sg_class_comp.counts`.
- Rules: VOC-sel. (5-fold OOF on VOC-Sim; the full-VOC-Sim-best threshold applied unchanged to COCO-Cap) —
  **primary**; fixed 0.5; the method's own binarisation; best on set (audit only).
- Paired Δ, method − BRACE (BRACE at VOC-sel. with DenseCRF, and raw), 2000-sample image bootstrap, 95%.
- Labels: everything exploratory; BRACE's VOC-Sim numbers are in-sample for its design; OVAM-token rows are
  few-shot.
- Runtime and peak GPU memory of the shared capture are recorded per item.

## Addendum 1 (before any capture was scored; same day)

1. **SeeDiff is a go.** Public code: https://github.com/BAIKLAB/SeeDiff (commit 28d26f4, AAAI 2025,
   arXiv 2507.19808). Its bulk script (`seediff_main_origin.py`) writes the image as the mask and its final mask
   uses the last token of a loop, so the cleaner demo path (`seediff_app.py`, `save_cross_attention`) is the
   reference: prompt-to-prompt AttentionStore (conditional softmax attention averaged over steps), its
   `aggregate_attention` at 8/16/32/64 with Otsu seeds and cosine region expansion, soft map = the 64-level map
   (bicubic to 512, min-max), own mask = PAMR (5000 iterations) of map > 0.3, then > 0.8. Deviations: SD 1.5
   instead of v1-4, 30 steps / CFG 7.5 (the benchmark generation) instead of 50 / 7; its class-name special
   cases (`pottedplant`, `tvmonitor`, `diningtable`) do not trigger with our class words; PAMR runs on the GPU.
   Like DAAM it needs the class word in the prompt (first sub-token), same subset rule.
2. **OVAM optimized token, two variants.** OVAM optimizes and evaluates a 2-row embedding ([<SoT>, class token],
   rows (0, -2) of `encode_text`, both optimized; `experiments/voc_simple/2_token_optimization.py`,
   `3_generate_voc_simple_dataset.py`). The released `tokens_VOC.npy` holds only the class mean of row 1.
   *Primary* (`ovam_t`): the class mean of the 2–3 per-image 2-row embeddings in OVAM's archive
   (`voc_sim_train/tokens/`; its row 1 equals `tokens_VOC.npy` exactly). *Secondary* (`ovam_tf`): the released
   token with the text encoder's <SoT> output (side F's runs/15/25 convention). Same VOC tokens on COCO-Cap.
3. **DenseCRF cost.** One DenseCRF R2 call takes about 4.7 s on one core here. DenseCRF rows therefore use a
   7-index window of the threshold grid around each method's raw full-VOC-Sim optimum (plus 0.5) on VOC-Sim,
   with OOF selection inside that window, and the selected index plus 0.5 on COCO-Cap. Raw rows use the full grid.
4. **Nodes.** Any V100 node (renyi's A100s excluded so the images match the V100 BRACE reproduction);
   at most 4 GPUs per user are shared with the other N jobs.

## Addendum 2 (after the 2-item smoke, before the full capture; nothing scored)

- Smoke (1 VOC-Sim + 1 COCO-Cap item, jobs 253925-6): generated images bit-identical to the BRACE reproduction.
- **Token row corrected.** OVAM's library applies a softmax over the given tokens (`token_softmax`) and its
  README states that the released `tokens_VOC.npy` (one row per class) is what all its evaluations use, so the
  released token is evaluated as [text-encoder <SoT> output, token] (`ovam_tf`; also side F's runs/15/25 form).
  That is now the primary and only scored token row. The class mean of the per-image 2-row embeddings (`ovam_t`)
  is not a released artefact; on the smoke item it gave IoU 0.27 vs 0.78; it stays in the capture files, unscored.
- **OVAM VOC-Sim index corrected** to OVAM's released `len(prompt.split())` (for "aeroplane", two sub-tokens,
  this is the first one); COCO-Cap keeps OVAM's −2 on the separately encoded class name.
- SCLIP moved to `sclip_f.py` (CPU, `goca` env), on the BRACE reproduction's stored images: F's port does not run
  under transformers 5.
