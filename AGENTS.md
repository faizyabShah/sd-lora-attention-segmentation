# BRACE project working context

Read `docs/NEW_MACHINE_SETUP.md` first. The opening Cityscapes material in
`README.md` describes the older LoRA project, not the current research direction.

## Research and collaboration rules

- BRACE is frozen: A3 cross-attention seed, V3 capped 16→32→64 self-attention
  hierarchy (one-sided cap only at 16→32), DenseCRF R2, threshold 0.55.
  Do not change the frozen method as part of setup or maintenance.
  Section 135 records this as an explicit owner override after the registered
  worst-class guard failed, not a registered promotion; preserve that distinction.
- Inference is label-free. Ground truth is for scoring only.
- Pre-register selection rules and commit them before scoring new experiments.
  VOC-Sim is for development; COCO-Cap is untuned transfer. Do not promote a
  method based on COCO-Cap tuning or post hoc results.
- Keep compared variants in the same capture. Cross-GPU regeneration is not
  pixel-identical; do not mix captures to construct a paired comparison.
- Report failed tests and distinguish exploratory, in-sample, OOF, fixed-threshold,
  and confirmatory results. Do not silently replace or promote a result.
- Coordinate scientific changes in
  `docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md`; consult the newer
  `outputs/segmentation/sg_real_atlas/analysis/` reports and pre-registrations too.
- Delete files only after an explicit yes from the user.
- Commit locally; push only when asked. Do not message other collaborators unless
  explicitly instructed.
- When providing paper .tex snippets, put them in chat and avoid internal names
  such as A3, V3, C16, and SGxx in manuscript prose.

## Where the active code lives

- `scripts/sg27_runtime.py`: frozen live generation and attention capture.
- `src/segmentation/sg_combined_v2.py`: locked hierarchy variants; BRACE uses V3.
- `scripts/sg_class_comp.py`: propagation, counts, OOF thresholds, bootstrap.
- `scripts/sg_gen_clip.py`, `scripts/sg_gen_capab.py`: generated-set CLIP seeds,
  cap ablations, and seed fusion.
- `scripts/sg_clip_dense.py`, `scripts/sg_voc21.py`: SCLIP and VOC val protocol.
- `paper/`: working manuscript, with older CVPR layout and narrative in places;
  the owner-supplied November handoff names ICPR 2027 as the intended target.

Large datasets, model weights, attention maps, and local environments are ignored.
Do not regenerate production captures merely to familiarize yourself with the repo.
