# Run 45 pilot: BLIP-caption prompts for our method (pre-registration)

Committed after the smoke (identity check + 8 items, numbers not used) and **before the full capture or any scoring**.
Development pilot on **PASCAL Context train** photos; nothing is promoted, and the paper 2 method stays frozen. An arm
that passes goes to a separately pre-registered confirmation on the run 44 test columns. Driver: `scripts/jc_blip45.py`.

## 1. Question

Run 44 found our maps very weak on PASCAL Context stuff classes (val: sky 0.137, road 0.072, wall 0.195; NERVE 0.80 / 0.43 /
0.09). The query prompt is always "A photograph of a {word}". **Does an image-specific BLIP caption as the prompt improve
our maps, especially for stuff classes, without hurting objects?**

## 2. Data (development only)

- **1,000 photos** sampled (seed 45) from the 4,999 PASCAL Context **train** photos. Excluded: the 5,104-photo Context
  val list and VOC2012 segmentation val. Labels come from run 44's conversion (iSeg's released 459 → 59 converter).
- **Items:** every class present; primary = class area ≥ 1% → **4,587 primary items**: 1,975 stuff, 1,587 VOC-class,
  1,025 other thing items.
- **Class groups, fixed now:**
  - stuff (19) and other things (20) follow OVDiff's released Context stuff/thing table; the 20 VOC classes are things.
  - **Stuff:** bedclothes, ceiling, cloth, curtain, fence, floor, grass, ground, mountain, platform, road, rock, sidewalk,
    sky, snow, track, wall, water, wood.
- These photos were never used before.

## 3. Fixed for every arm (= run 44)

- **Pass:** one SD 1.5 pass per item at t = 100, noise seed 0; uniform CA seed.
- **Gate:** MK4 with **run 44's Context profiles** (built with the template prompt from COCO train captions); MK4-LR at
  γ = 1.
- **Read-out:** fine-only V0; min-max; **fixed 0.55**; raw. Also stored: no gate (u-) and MK4 (uM).
- **Word:** as run 44 (e.g. "table" for dining table).
- **Identity check (passed in the smoke, job 263670):** arm A through this driver reproduces run 44's stored maps on 3
  Context val items, with a max |difference| of 0.0 on all 6 maps. Smoke outputs are in `<CACHE>/jc_blip45_smoke` and
  `result_smoke.md` (1 photo; not used).

## 4. Arms

| Arm | Prompt | Word's cross-attention |
|---|---|---|
| **A** (reference) | "A photograph of a {word}" | in-prompt (the word's tokens) = run 44 |
| **At** | "A photograph of the {word}" | in-prompt (tests the grammar of "a sky") |
| **C** | BLIP-large caption **conditioned on** the prefix "a photograph of {word}" (DiffSegmenter's recipe, without "++") | in-prompt, first occurrence (the prefix) |
| **B** | BLIP-large **unconditional** caption of the photo (one per photo) | **out-of-prompt** (OVAM-style): image queries against the keys of the bare word prompt (`jc_ov42.word_attn`) |

- **BLIP model:** Salesforce/blip-image-captioning-large, the checkpoint in our DiffSegmenter driver. Greedy decoding at
  its default length. Captions are generated once and saved.
- **Captions are used exactly as generated.** BLIP-large's unconditional captions sometimes begin with its known junk token
  ("arafed", "araffe"; seen in the smoke); they are kept, and their frequency is reported. Conditional captions
  can be ungrammatical ("a photograph of wall of people…"); they are kept too.
- **Secondary maps from the same passes:**
  - **A|op**: the template prompt read out-of-prompt. Combined with B, this separates the prompt from the read-out.
  - **B|ip**: B read in-prompt, only on items whose caption contains the word's tokens.

## 5. Primary contrasts and pass rule (MK4-LR, V0, raw 0.55)

For each arm X in {At, C, B}: **X − A** on stuff (19), things (40) and all 59 (class-macro mIoU; 2,000-sample paired
photo bootstrap; 95% CI).

**An arm PASSES if its stuff Δ ≥ +0.02 with CI > 0 AND its things Δ has a CI lower bound ≥ −0.01.**

- If several arms pass, the one with the largest all-59 Δ goes to confirmation.
- If none passes, caption prompts are closed for this pipeline (with these profiles), and the reason is reported.

## 6. Gate-input check (decides the profile question; fixed now)

The profiles were built with the template prompt, while the gate reads the image-side cross-attention queries, which
depend on the prompt only indirectly. Each pass records the mean row cosine between its gate queries and arm A's (per
item, mean over modules).

- If the **5th percentile ≥ 0.98** for the selected arm, the existing profiles are taken as valid for it.
- Otherwise the confirmation also includes an arm with **caption-conditioned profiles**: the same recipe, with the
  profile photos' prompts built by the same caption rule.

## 7. Secondary (descriptive)

- Each arm − A with no gate.
- A|op − A|ip (read-out only).
- B − A|op (prompt only, both out-of-prompt).
- MK4-LR − none within each arm.
- B in-prompt − A on items whose caption names the word.
- The VOC-20 and other-thing groups.
- Per-item AUROC (target vs other valid pixels) of the seed, of V0 without the gate, and of MK4-LR V0.
- Every threshold from 0.30 to 0.70; per class; caption examples.

## 8. Next (not part of this pilot)

- **Confirmation (if an arm passes):** the selected arm on the run 44 test columns (PASCAL VOC val, Context val,
  COCO-Object). Pre-registered separately; the same captures and contrasts vs A.
- **Run 46, SCLIP + profiles (owner decision 2026-11-26: test both):**
  - Our positive/negative profiles are built on SCLIP's dense features (the ones it scores classes with), with an
    MK4-style gate (no low-resolution consensus).
  - Two arms: **(i) the gated SCLIP map alone** and **(ii) the gated SCLIP map propagated with SD self-attention** (our
    V0).
  - Compared with SCLIP, positive-only profiles and the multi-class protocol.
  - A literature check comes first (ReCo, OVDiff, FreeDA prototypes). Pre-registered separately, on development photos.

## 9. Cost

- Captions: about 5,600 generations, under 1 GPU-hour.
- Capture: 4 passes per item, about 18,300 passes, about 5 GPU-hours on 4 shards.
- Scoring: CPU.
