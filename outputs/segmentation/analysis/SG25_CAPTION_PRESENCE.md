# SG25 COCO caption-presence split — 24 October 2026

## Which prompt contains the queried class?

For **all 707 COCO-Cap images**, the image is generated from its natural
caption, while cross-attention attribution separately encodes the target
class name as a padded class-only prompt. Thus an *absent* class means absent
from the **generation caption**, not absent from the attention query. The
released `word_included` field marks **465/707 (65.8%) present** and
**242/707 (34.2%) absent**. Every one of the **490 VOC-Sim** generation
prompts contains its queried class name, and its attribution prompt is that
same caption.

The release flag is the primary grouping below. A stricter exact
class-phrase match finds 437 present and 270 absent; its 28 disagreements
are release-present cases such as plural class words (`birds`, `boats`).
The released category for each caption matches the queried mask class.

## Frozen SG25 comparison

Only previously frozen, full-VOC-selected thresholds are used:
FINE-P2 `.450`, 16→64 `.525`, FINE-P2+R2 `.425`, and 16→64+R2 `.500`.
These are the SG25 images and masks; no new generation or COCO tuning was
performed. Macro mIoU is pooled within class, then averaged over classes
available in the subgroup. Mean-image gain is paired per image.

| COCO subgroup | Images | Classes | Raw fine → 16→64 | Raw gain | R2 fine → 16→64 | R2 gain | Paired mean-image R2 gain (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Class word present | 465 | 19 | .6628 → .6791 | +.0163 | .6642 → .6881 | **+.0239** | +.0260 [.0178, .0346] |
| Class word absent | 242 | 20 | .6631 → .6979 | +.0348 | .6661 → .7140 | **+.0480** | +.0293 [.0160, .0435] |

Both groups gain. The larger absent-group **macro mIoU** gain is partly
sensitive to class composition: all **14 potted-plant** images are in the
absent group (its own gain is +.0828), leaving only 19 classes in the
present group. Restricting both groups to the **same 19 classes** gives
R2 macro gains **+.0239 present** and **+.0461 absent**.

The evidence for a genuine *interaction* between caption absence and the
16→64 method is weak. The difference between subgroup **paired mean-image
gains** is only **+.0033** in favor of absent, with an image-bootstrap 95%
interval **[−.0122, +.0196]**. Giving each of the 19 shared classes equal
weight, the absent-minus-present difference in **class-pooled IoU gains**
is **+.0222**, but a bootstrap over classes gives **[−.0066, +.0504]**.
Both intervals include zero. These are descriptive post-hoc subgroup
analyses, not a new selection criterion or a proof of absence-specific
behavior.

The class effects are mixed. **Person** improves +.0776 when the class
word is absent but falls −.0352 when present; **aeroplane** does the
reverse (+.1123 present, −.0045 absent). **Car** improves in both groups
(+.1194 present, +.1621 absent). Several absent-class samples are sparse:
only two each for chair and bottle, and three for TV. This limits
class-level claims.

The [per-image CSV](../controlled_attention_cv/sg25/caption_presence_per_image.csv)
contains the release flag, strict-phrase flag, and TP/FP/FN/TN and IoU for
all four compared cells at their frozen thresholds. It is enough to
reproduce the subgroup scores without images or attention tensors. The
original [SG25 handoff](SG25_PRESG26_CHATGPT_HANDOFF.md) has the full
experiment and Pre-SG26 results.
