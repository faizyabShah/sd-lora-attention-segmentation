# Owner-selected Cityscapes joint segmentation — 16 November 2026

Owner replaces the earlier selection with these **65 candidates**, in this
exact order: 0, 1, 2, 4, 5, 14, 25, 27, 28, 34, 43, 45, 47, 57, 61, 65,
89, 112, 115, 124, 143, 144, 146, 149, 499, 497, 494, 490, 489, 486, 471,
468, 467, 456, 447, 437, 430, 422, 412, 402, 399, 394, 390, 381, 372, 365,
338, 335, 331, 324, 318, 306, 302, 298, 292, 289, 282, 283, 274, 276, 266,
257, 215, 204, 192. Use original candidate PNGs; no visual reselection.
Preserve the old selection and outputs separately.

## Taxonomy and joint queries

Output label 0 is background. Labels 1–10, in owner order:
road; sidewalk; building/wall; vegetation/terrain; sky; person/rider; car;
bus; train/tram; motorcycle/bicycle. Truck and minor/unlisted classes are
not foreground targets. SCLIP's background vocabulary includes them.

One shared attribution prompt lists all 15 atomic class words:
`road, sidewalk, building, wall, vegetation, terrain, sky, person, rider,
car, bus, train, tram, motorcycle, bicycle`. The frozen A3 machinery observes
all these tokens jointly, with one shared padded 77-token softmax. For slash
groups, take the pixelwise maximum of their constituent A3 seed maps (union).
SCLIP uses the same 10 output groups as synonym groups, jointly competing
against background, with the existing B/16, 80 templates, scale 65,
probability .1 and group-level area filter .1.

**No independent per-class min-max normalization.** Each image/method uses a
single common peak scaling over the entire class stack, retaining relative
class strengths and exactly-zero seeds. These are relative activation
scores, not calibrated probabilities. This distinction is needed because
independent normalization forces an absent nonconstant class to reach 1.

## Three methods and exclusive decoding

Use A3 seed → frozen V3 hierarchy (BRACE); SCLIP group probability seed → V3;
and early geometric seed sqrt(A3*SCLIP) → V3. Apply common peak scaling to each
seed stack, and again to each method's propagated score stack. All methods
share the same captured self-attention for each image. Geometric fusion is
the requested exploratory comparator, not a registered promotion.

At each native-resolution pixel, pick the highest-scoring of the 10 groups.
At thresholds **0.40, 0.55, 0.70**, assign background if the highest score is
below threshold; otherwise assign the winner. Ties use owner-listed order.
Output is a single mutually exclusive 0–10 label map, never overlapping
independent binary masks. Save these exact raw assignments.

Also apply the existing R2 pairwise parameters to the 11-class one-hot raw
assignment with SimpleCRF. To preserve the owner's highest-score/background
rule, the refined result retains the original winner only where CRF agrees;
otherwise it assigns background. Thus refinement can veto an assignment but
cannot switch a pixel to a lower-scoring class or fill subthreshold pixels.
Save raw and refined results, and provide both in the viewer. This joint
multiclass decoder is an exploratory adaptation; frozen src/ is unchanged.

## Input capture, checks and unattended execution

Use the original image-anchored renoise procedure (512-square bicubic VAE
posterior mean, fixed crc32(image_id) noise, original 30-step PNDM schedule,
31 calls), original generation conditioning and negative prompt, and the
supplied LoRA at that candidate's recorded scale. Native output dimensions.
Check LoRA/source hashes. A3 first-class parity and batched V3 parity must be
<=1e-5 against unchanged source. Test joint assignments, slash union,
background gating, and multiclass CRF labels before launch. No GT is used.

Run a smoke check, then three GPU shards in detached Screen. Automatically
wait for all shards, run CPU rendering/reporting, validate the **585** refined
and **585** raw label maps, and create the archive. Write completion/failure
status and preserve full tracebacks. Resume completed items by config and
source hashes. Stop after launch/readiness verification; the owner explicitly
requests no continuous monitoring. No stage needs the agent to trigger it.
No push requested.
