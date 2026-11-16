# Cityscapes LoRA 50: exploratory visual threshold comparison

Written 16 November 2026 before capture or mask inspection. Owner requests all
50 selected images, each major class, three thresholds, and BRACE / best
established SCLIP-seeded BRACE / early geometric fusion.

## Fixed inputs and comparisons

Use the exact 50 PNGs and provenance in
`datasets/cityscapes_lora_benchmark_20261116/manifest.json`. No image selection
or regeneration. Classes in order: road, sidewalk, building, vegetation,
terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle.
Query every class on every image, including absent classes. No ground truth.

Thresholds: **0.40, 0.55, 0.70**, fixed before looking at masks. The middle
threshold is the frozen BRACE reference. Each output is a class-given binary
mask, so masks of different classes may overlap. No pixel accuracy, optimal
threshold, mIoU, or winner will be inferred without annotations.

Use the existing selected A3 cross-attention seed and V3 one-sided-capped
16→32→64 hierarchy, all heads, SA calls 10–20, frozen R2 DenseCRF. Three seeds:
A3; min-max SCLIP probability; min-max sqrt(minmax(A3)*minmax(SCLIP)). Each
method uses the SAME captured self-attention for an image. Apply min-max to
the propagated map, bilinear upsample to the original PNG size, then each
threshold and the unchanged R2 binary-mask CRF. Keep pre-CRF masks too.

The SCLIP-seeded V3 configuration is the established VOC-selected choice.
Geometric early fusion is an exploratory requested comparator: it had the
strongest reported COCO fixed-threshold result, but the registered selector
chose arithmetic fusion and its confirmatory gain test failed. Do not call
geometric fusion a registered winner or promote it here.

## Cityscapes input adaptation

The selected PNGs were generated with a LoRA, negative prompts, batching and
mixed aspect ratios. Frozen capture assumes square 512px inputs. Use the
repository's Section 136/137 image-anchored renoise procedure: bicubic stretch
to 512², deterministic VAE posterior mean, one crc32(image_id)-seeded fixed
noise tensor, and replace every UNet input by sqrt(alpha_t)*z0 +
sqrt(1-alpha_t)*noise at the original PNDM 30-step schedule (31 UNet calls).
Condition on the original generation prompt and negative prompt. Enable the
supplied LoRA at each image's recorded scale, use float32 capture, preserve
original PNGs and map the outputs back to their original dimensions. This is
an exploratory Cityscapes/LoRA adaptation, not the validated VOC regeneration
protocol or an exact replay of the original generation trajectory.

Query each class separately using the padded 77-token text encoding of
`a photograph of a <class>`, independently of the generation conditioning.
The A3 layer weights depend on image features, not target tokens, so collect
all 14 class seeds in one shared trajectory. Check the first class against
the unchanged SG25 recorder. Batched V3 propagation must match the unchanged
`sg_combined_v2.aggregate(...)[0]['V3']` on representative seeds (max error
<=1e-5). Existing frozen code in src/ is not modified.

VOC21 cannot represent the six Cityscapes stuff classes separately: use an
explicit fixed Cityscapes vocabulary with background (wall/fence/pole/traffic
light/traffic sign/other object), then the 14 target classes. Synonyms:
road/street; sidewalk/pavement; building; tree/bush/hedge/vegetation;
grass/dirt/soil/terrain; sky; person/pedestrian;
cyclist/motorcyclist/person riding a bicycle; car; truck; bus; train/tram;
motorcycle/motorbike/scooter; bicycle/bike. Existing SCLIP ViT-B/16, 80 text
templates, sliding windows, logit scale 65, probability .1 and area .1 remain.
This vocabulary change is recorded and is not VOC21 score reproduction.

## Outputs and checks

Commit this plan before running. Save all seeds, all propagated maps, source
hashes and runtimes; export 50×14×3×3 = **6300** post-CRF binary masks plus
raw masks and a local HTML browser with comparison sheets. Report coverage,
empty/full masks and threshold sensitivity descriptively, not as quality
metrics. Inspect representative outputs from every class and foreground
family. Keep failures and full tracebacks. Use detached Screen with automatic
capture→CRF/render→report, checkpoints and a failure status; do not affect
other Screen jobs. No push requested.
