# Response to JC's review: what he asked, what we found, what we change

Draft, 19 Nov 2026. For Narmeen, as the source for the meeting slides and for the paper edits.

**How to read the references.**

- **"p. 4, Sec. 4 'Selection status'"** = the PDF JC read ("TRDP_Paper (2).pdf", 9 pages, attached with his email).
- **`sec/4_protocol.tex:10`** = the same text in the repo copy of the paper (`sd-lora-attention-segmentation/paper/`), where the edit will be made. The repo copy matches JC's PDF except for the equation numbering in Sec. 3.3 (see #6).

**Words used throughout.**

- **BRACE**: our full method (SA-guided CA seed → coarse-to-fine SA with the one-sided cap → DenseCRF → threshold 0.55).
- **Fine-only**: the simpler pipeline without the coarse-to-fine hierarchy (the seed is propagated twice through the fine 64-pixel SA only, then DenseCRF).
- **mIoU**: how well masks overlap the annotation, 0–1, averaged over the 20 classes.
- **VOC-Sim**: 490 simple images, one object ("a photograph of a cat"). **COCO-Cap**: 707 cluttered images from COCO captions.
- **Out-of-fold (OOF)**: VOC-Sim is split into 5 parts; each choice is made on 4 parts and scored on the 5th. **Forced OOF**: one fixed variant scored that way.
- **VOC-sel.**: the threshold is chosen on VOC-Sim and reused unchanged on COCO-Cap.
- **Paired**: two methods scored on exactly the same images; the difference (Δ) comes with a 95% interval in brackets. If the interval does not contain 0, the difference is real.
- **DenseCRF**: a standard post-processing step that snaps the mask to image edges.

---

## #1 Contribution and claim

> *JC:* "In Table 1, the two new components (SA-guided layer weighting and the capped hierarchy) add +0.006 each on VOC-Sim; most of the gain over OVAM comes from configuration choices (CA scales, SA passes, DenseCRF, call window). The strongest argument for the hierarchy is COCO-Cap (+0.034), i.e. cluttered scenes. I would frame the contribution around that, rather than around «+12.5 points over OVAM». Also note that both components were adopted after the selection rule had kept the simpler reference (Sec. 4): this is honest, but reviewers will read it against the contribution «fold-safe selection protocol», so the text must handle it carefully."

### 1.1 What he means, in plain words

**Part 1: the headline is the wrong number.** The paper sells BRACE by its margin over OVAM (+0.125 on VOC-Sim, +0.114 on COCO-Cap). Table 1 shows where that margin comes from:

| Table 1 row (p. 4) | VOC-Sim | Δ | What kind of change |
|---|---|---|---|
| CA only, all scales | .665 | | starting point |
| drop the 32-pixel CA layers | .691 | +.026 | configuration |
| add 64-pixel SA propagation | .789 | +.033ᵃ (really .691 → .789 = +.098) | configuration |
| two passes | .800 | +.012 | configuration |
| DenseCRF | .813 | +.014 | configuration |
| SA calls 10–20 | .817 | +.004 | configuration |
| **SA-guided layer weights (new)** | .823 | **+.006** | **our new component** |
| **coarse-to-fine + cap (new)** | .829 | **+.006** | **our new component** |

So on VOC-Sim, about 0.15 of the gain is configuration (choices anyone could make on top of OVAM) and only 0.012 is our two new ideas. A reviewer will say: "the novelty adds 1 point". **But on COCO-Cap the new components matter:** .644 → .662 (+0.018, layer weights) → .696 (+0.034, hierarchy). JC's advice: make the paper about *why the scale structure matters in cluttered scenes*, not about beating OVAM.

**Part 2: the selection rule and the "simpler reference".** You asked what these are.

- **The selection rule** is written in Sec. 4, "Fold-safe selection" (p. 4, `sec/4_protocol.tex:7`): a change is accepted only if it gains at least 0.010 mIoU on the training folds (or 0.005 with at least 12 of 20 classes better and **no class losing more than 0.05**), and wins at least 4 of 5 folds.
- **The simpler reference** is the pipeline *without* the new component. There are two:

| New component (the "complicated" choice) | Simpler reference the rule kept | Why the rule kept the simpler one |
|---|---|---|
| SA-guided CA layer weights | **uniform** layer weights (plain average over CA layers) | Won 4/5 folds (+0.006), but the multi-candidate screen kept uniform weights (supp. Tab. 10, p. 10 of the supplement: "selector kept uniform weights") |
| coarse-to-fine hierarchy with the cap (BRACE) | **fine-only** propagation + DenseCRF | Won 5/5 folds (+0.0063), but one class, **sheep, lost 0.057–0.104**, more than the 0.05 guard |

So the rule, applied as registered, would have given a paper whose final method is **fine-only with uniform weights** (0.823 on VOC-Sim). The owners overrode the rule and adopted both components. The paper says so in one sentence (p. 4, Sec. 4 "Selection status", `sec/4_protocol.tex:10`) and once more in Sec. 7 (p. 8, `sec/7_discussion.tex:7`).

### 1.2 The problem, exactly

1. **The paper contradicts itself.** Contribution 4 (p. 1, `sec/1_intro.tex:14`) says "every choice made on VOC-Sim folds", and the intro says "Every choice is made on VOC-Sim folds, with COCO-Cap used only after freezing" (p. 1, `sec/1_intro.tex:7`). But the two choices that *are* the novelty were not made by the rule.
2. **The override saw COCO-Cap.** The protocol log (Section 135, 31 Oct) shows the decision to adopt the hierarchy was taken **after** COCO-Cap was scored, and it cites the COCO-Cap +0.0341 as evidence. So COCO-Cap is not an untouched test set for that decision, yet the paper presents COCO-Cap as "frozen transfer" and JC himself calls it the strongest argument. A reviewer who notices this will read it as choosing the method on the test set.
3. **"Out-of-fold" is used for a number the procedure did not produce.** 0.829 is a *forced* OOF score of the owner-chosen variant (abstract, intro). The out-of-fold score of the registered procedure is 0.823 (fine-only).
4. **The abstract says "Selecting every choice on five class-stratified folds"** (p. 1, `sec/0_abstract.tex:2`), which has the same problem.

### 1.3 How to handle it carefully

1. **Say plainly what the rule chose and why** (sheep guard; uniform weights), next to our choice. Print both pipelines on both benchmarks (the rule's choice is not printed anywhere in the main paper today; see #2).
2. **Say when the override happened**: after COCO-Cap was seen. Treat the COCO-Cap gain as evidence for the calibration analysis (Sec. 5.5), not as held-out confirmation.
3. **Label 0.829 as forced OOF** everywhere.
4. **Reword contribution 4**: a protocol that *records* every decision, including the two owner decisions, rather than one that "made every choice".
5. **Give a clean test of the decision**, scored after it was frozen. We now have one: the **#5 backbone runs** (Sec. #5 below) were done after the hierarchy was frozen, on new backbones. The hierarchy holds on SD 2.1 (+0.054 / +0.060) and fails on SDXL (−0.022 / −0.011). That is honest, untouched evidence, positive and negative. (Caveat: that setting is one noising pass, not the paper's generation-time setting.)

### 1.4 The evidence behind each claim, and what happens to it

| Claim in the paper | Where (JC's PDF; repo file) | Evidence now | Action |
|---|---|---|---|
| "+0.125 / +0.114 over class-word OVAM" | p. 8, Sec. 6; `sec/6_comparison.tex:3`. Abstract and intro give 0.829 / 0.696 "against 0.704 and 0.582 for class-word OVAM" (p. 1) | Table 1 (p. 4): new components add 0.012 of it | A row in the comparison table, not the headline |
| "Most of the COCO-Cap margin over our fine-only pipeline comes from the capped hierarchy" (+0.034) | p. 8, Sec. 6; Table 1 last row (p. 4); Table 3 (p. 7) | F's trace (workplan Entry 9): the +0.034 compares the two maps at **two different thresholds** (0.55 vs 0.425). At one shared threshold: +0.012 [−0.002, +0.026]. With each map at its own COCO-best threshold: +0.018. The per-image oracle is a tie (0.896 vs 0.895) | Keep, restated as **cross-image calibration** (which is what Sec. 5.5 already says) |
| SA-guided layer weighting helps (+0.018 on COCO-Cap) | p. 4, Table 1 row 7; p. 5–6, Sec. 5.1 | Rerun (Entry 8): **+0.024 [+0.013, +0.036]** on COCO-Cap at 0.55 | Keep: the cleanest positive result in clutter |
| "DenseCRF gains six times more on capped maps in cluttered scenes" | p. 1, abstract; p. 6, Sec. 5.4; p. 7, Table 4 caption; `sec/0_abstract.tex:2`, `sec/5_analysis.tex:50`, `tables/tab_refine.tex:15` | Same two-threshold artefact: at matched thresholds DenseCRF adds about +0.014 to both maps (N audit) | **Drop** |
| One-sided cap beats no cap | p. 6, Sec. 5.3; p. 7, Table 3 | Rerun: +0.010 VOC-Sim; **−0.005 [−0.011, −0.001]** COCO-Cap at 0.55 | Weaken: keep the mass analysis (13% background vs 0.5% object removed) |
| "Every choice is made on VOC-Sim folds, with COCO-Cap used only after freezing" | p. 1, intro; `sec/1_intro.tex:7` | Override after COCO-Cap (protocol Section 135) | **Rewrite** (below) |
| Best training-free method (implied) | p. 8, Table 5 | Paired comparison (#3): NERVE (adds CLIP) beats BRACE on COCO-Cap, +0.032 [+0.007, +0.060] | Narrow to "best method that uses only the diffusion model" |

### 1.5 Replacement text, location by location

Numbers below are the paper's current ones (0.829 / 0.696). If the owners switch to the prompt-corrected capture (runs/20: 0.832 / 0.709; see 1.6), replace them in one pass.

**(a) Abstract** — p. 1; `sec/0_abstract.tex:2` (whole abstract).

*Now:* "… We show that how these signals are combined across spatial scales matters more than which graph operator is applied to them. … DenseCRF gains six times more on capped maps in cluttered scenes. Selecting every choice on five class-stratified folds of VOC-Sim, the pipeline reaches 0.829 out-of-fold mIoU and transfers frozen to COCO-Cap at 0.696, against 0.704 and 0.582 for class-word OVAM. …"

*Replace with:*

> Text-to-image diffusion models localize words in cross-attention (CA) and group pixels in self-attention (SA), so the images they generate can be segmented without training. We study how these signals should be combined across spatial scales, and find that the choice matters most in cluttered scenes. Our training-free pipeline, BRACE, weights CA layers by their agreement with SA, propagates the resulting class-word seed from coarse to fine SA, and lets each finer level redistribute evidence only within the mass its coarse parent assigns. All settings are selected on class-stratified folds of the single-object VOC-Sim, where the two new components add little (+0.006 each; 0.829 forced out-of-fold mIoU). On the cluttered COCO-Cap the frozen pipeline reaches 0.696, 0.034 above a fine-only pipeline selected the same way, and SA-guided layer weighting is the most robust single component. An oracle analysis shows why: the hierarchy does not rank pixels better within an image; it makes maps comparable across images, so a threshold chosen on simple scenes still works on cluttered ones. In a paired comparison on the same images, BRACE is the strongest method we evaluated that uses only the diffusion model; a method that adds a CLIP seed is stronger on COCO-Cap, and we report it separately. Our registered selection rule preferred the simpler fine-only pipeline; we adopted the hierarchy after observing COCO-Cap, and report both pipelines and every negative result.

**(b) Introduction, results paragraph** — p. 1, right column ("The resulting pipeline, BRACE …"); `sec/1_intro.tex:7`.

*Now:* "… On VOC-Sim it reaches 0.829 out-of-fold mIoU, and on COCO-Cap it reaches 0.696 in frozen transfer, against 0.704 and 0.582 for OVAM with class words. Every choice is made on VOC-Sim folds, with COCO-Cap used only after freezing."

*Replace with:*

> The resulting pipeline, BRACE (Bounded Refinement of Attention from Coarse Envelopes), uses no training, no optimized tokens and no tuned graph parameters. Its two new components add little on single objects (0.829 forced out-of-fold mIoU on VOC-Sim, +0.006 each) but matter in clutter: on COCO-Cap the hierarchy adds 0.034 over fine-only propagation, and we show that this gain is cross-image calibration (Sec. 5.5). Every setting is selected on VOC-Sim folds; the decision to adopt the hierarchy over the simpler pipeline preferred by our registered rule was made after COCO-Cap had been scored, so we report both pipelines throughout.

**(c) Contribution list** — p. 1–2; `sec/1_intro.tex:11–14`.

*Replace with:*

> 1. A diffusion-only, training-free pipeline that turns attention into class masks: SA-guided CA layer weighting, coarse-to-fine SA propagation, and a one-sided cap that bounds each finer level by the coarser one.
> 2. Evidence that the scale structure matters in clutter rather than on single objects: about +0.006 per component on VOC-Sim, +0.034 for the hierarchy and +0.018 for the layer weighting on COCO-Cap, with controlled ablations of each choice.
> 3. An oracle threshold analysis that explains the hierarchy's gain as cross-image calibration, not better ranking within an image, and its main failure (objects that fill the frame).
> 4. A fold-safe protocol that records every decision — including the two components adopted against the registered rule, and when — and reports all negative results, together with paired reruns of competing methods on the same images.

**(d) Sec. 4, "Selection status"** — p. 4; `sec/4_protocol.tex:10`.

*Now:* "… Two components were adopted after the rule had retained a simpler reference. The self-attention-guided layer weights won 4/5 folds and transferred, but the multi-candidate screen kept uniform weights. The capped hierarchy won 5/5 folds against both references, but one class (sheep, Sec. 5.5) lost more than the 0.05 guard. The supplementary material reports the registered outcomes in full."

*Replace with:*

> The CA scales, the SA module, the two-pass baseline, DenseCRF and the SA call window were promoted by this rule. The rule kept simpler references for the two new components. The SA-guided layer weights won 4/5 folds, but the multi-candidate screen kept uniform weights. The capped hierarchy won 5/5 folds against both references (+0.006 forced OOF), but one class (sheep, Sec. 5.5) lost more than the 0.05 guard, so the rule kept fine-only propagation. We adopted both components as an explicit decision after the COCO-Cap results had been computed. COCO-Cap is therefore not an untouched test set for these two decisions, although every setting inside them was selected on VOC-Sim. We report the rule's pipeline next to ours on both benchmarks (Table X), use the COCO-Cap gain as evidence for the calibration analysis of Sec. 5.5 rather than as held-out confirmation, and give a test scored after the decision was frozen (new backbones, Sec. 5.y). Section S8 and Tab. S10 list the registered outcomes.

**(e) Sec. 5.4 and Table 4 caption, "six times"** — p. 6 (Sec. 5.4) and p. 7 (Table 4); `sec/5_analysis.tex:50`, `tables/tab_refine.tex:15`.

*Now (both places):* "… On capped coarse-to-fine maps it raises precision, helps COCO-Cap six times more (+0.014, 18 of 20 classes), …"

*Replace with:*

> On capped coarse-to-fine maps it mainly raises precision and improves 427 COCO-Cap images against 105 that get worse (fine-only: 354 against 243).

and delete "DenseCRF gains six times more on capped maps in cluttered scenes." from the abstract (done in (a)).

**(f) Sec. 6 (will become Sec. 5.1, see #8)** — p. 8; `sec/6_comparison.tex:3`.

*Now:* "… With class words only and no training, it exceeds OVAM by 0.125 on VOC-Sim and 0.114 on COCO-Cap. It also exceeds DatasetDM, which trains an extra decoder, on both benchmarks. Most of the COCO-Cap margin over our fine-only pipeline comes from the capped hierarchy: cluttered scenes benefit most from a coarse envelope. …"

*Replace with:*

> Table 5 compares BRACE with competing methods on the same images (paired block) and lists the numbers published for these benchmarks (unpaired block). Among methods that use only the diffusion model, BRACE is the strongest on both benchmarks: our run of OVAM with the class word trails it by 0.13 / 0.14 on the same images, DAAM by 0.10 / 0.07, SeeDiff by 0.11 / 0.05, GoCA by 0.08 / 0.11 with SD 1.5 and by 0.03 / 0.10 with SDXL, and DiffSegmenter by 0.04 / 0.02. OVAM with its optimized token, which is few-shot, ties BRACE on VOC-Sim (−0.003 [−0.016, +0.010]) and trails it on COCO-Cap (−0.044). NERVE, which adds a CLIP seed, is on par on VOC-Sim and stronger on COCO-Cap (+0.032 [+0.007, +0.060]); with its own released thresholding rule it falls below BRACE (0.812 / 0.663). The published numbers come from OVAM's reruns of each method and use other images and post-processing, so they are indicative only.


**(g) Sec. 7, "Failure modes and scope", last sentence** — p. 8; `sec/7_discussion.tex:7`.

*Now:* "Two components were adopted after the registered rule kept a simpler reference (Sec. 4); their gains hold in 4/5 and 5/5 held-out folds and transfer, but they are not pre-registered promotions."

*Replace with:*

> Two components were adopted after the registered rule kept a simpler reference, and after COCO-Cap had been scored (Sec. 4). Their gains hold in 4/5 and 5/5 held-out folds; on new backbones, scored after the decision was frozen, the hierarchy helps on SD 2.1 but not on SDXL (Sec. 5.y). A further benchmark scored once with both pipelines frozen would test the decision cleanly.

**(h) Conclusion** — p. 8; `sec/7_discussion.tex:10`. Add one sentence:

> The division of labour matters most in cluttered scenes, where it makes maps comparable across images, and it needs at least three attention scales: on SDXL's two, the hierarchy does not help.

### 1.6 Open decisions for the owners

- **Which capture is canonical**: the original (0.829 / 0.696) or the prompt-corrected one (runs/19–20: some benchmark prompts did not match the stored images; BRACE 0.832 / 0.709). All new paired comparisons (#3, #5) use the corrected prompts.
- **Headline**: "best diffusion-only method + separate CLIP block" (recommended) or an analysis paper without a ranking claim.

**Slides:** `c1_N.tex` (5 frames: claim moves; claims kept; claims weakened or dropped; override disclosure; options).

---

## #2 Mapping contributions to evidence

> *JC:* "for each of the four contributions … which section, table and figure provide the evidence, and which number is the key result."

### 2.1 The mapping (current paper, JC's PDF numbering; "S" = supplementary)

| # | Contribution (Sec. 1, p. 1) | Section | Tables | Figures | Key number | Main or supp |
|---|---|---|---|---|---|---|
| 1 | **Pipeline**: SA-guided seed, coarse-to-fine SA with cap, DenseCRF | 3.2–3.5, 5, 6 | Tab. 1 (last row), Tab. 5; S6, S11 | Fig. 1, 2; S8, S9 | **0.829** forced OOF / **0.696** frozen; vs OVAM 0.704 / 0.582 (published, not paired) | Main |
| 2a | Ablation: CA scales and layer weighting | 3.2, 3.3, 5.1 | Tab. 2, Tab. 1 rows 1–2, 7; S1, S3 | none in main; S1, S4 | drop 32-px CA: 0.665 → 0.691 (4/5 folds); weighting 0.817 → 0.823 (4/5), COCO 0.644 → 0.662 | Numbers main; **pictures only in supp** |
| 2b | Ablation: coarse vs fine SA | 3.4, 5.2 | Tab. 3 (first rows); S4 | Fig. 3 left/middle | 16-px SA alone 0.72; hierarchy vs fine-only COCO 0.693 vs 0.662 | Main; **no picture of a fine-only map next to the coarse one** |
| 2c | Ablation: cap vs exact conservation vs none | 3.4, 5.3 | Tab. 3; S5 | Fig. 3 right, Fig. 4; S8 | cap 0.830 (5/5 folds); conservation −5 points raw; cap removes 13% background vs 0.5% object | Main |
| 2d | Ablation: DenseCRF on each map | 3.5, 5.4 | Tab. 4, Tab. 1 row 5 | Fig. 5; S5 | images up/down 427/105 (capped) vs 354/243 (fine-only) on COCO-Cap ("six times" to be dropped, #1) | Main |
| 3 | **Oracle threshold analysis** | 5.5, 7 | **none in main**; S7 | Fig. 6; S6, S7 | per-image oracle tie 0.896 vs 0.895; best global threshold 0.425 → 0.55; sheep −0.09 | Claim main; **support in supp** |
| 4 | **Fold-safe protocol** | 4, 7 | **none in main**; S2, S8, S9, S10 | **none** | rule (+0.010 / 0.005, 12/20, 0.05 guard, 4/5 folds); 45 registered comparisons; **the rule's own choice (fine-only, 0.823) is not printed in the main paper** | **Mostly supp** |

Full version with the committed source of every number: `comparisons/notes/entry_2_mapping.md`.

### 2.2 What the table shows

- **Well supported in the main paper**: contributions 1 and 2 (all four ablations).
- **Rely on the supplementary**: 3 (the oracle has a figure but its supporting table is S7) and **4** (no table or figure at all in the main paper).
- **Weak spots**: the qualitative claim of contribution 2 ("each is backed by quantitative and qualitative evidence") holds for the cap and DenseCRF, not for CA scales (pictures only in supp) or coarse vs fine SA (no fine-only picture).
- **Results that support no contribution** (shorten or move to the supplement): Sec. 5.1's untracked sentence on CA heads and calls; the COCO-Cap person column in Fig. 2; the GoCA-orientation footnote and Tab. S1; Tab. S9 (our adaptations of related methods, now superseded by real reruns, #3); Sec. S9 / Tab. S11 (merge into the new Table 5); Sec. S1 "Cost" (replaced by measured cost, #8).

### 2.3 Fixes that follow

- Contribution 4: add a small main-paper table, **the rule's choice next to ours** (fine-only + uniform weights vs BRACE, both benchmarks). This also handles #1.
- Contribution 3: move one row of Tab. S7 (the label-free per-image thresholds) into the text of Sec. 5.5.

**Slides:** `c2_N.tex`.

---

## #3 Comparison with the state of the art

> *JC:* "a paired block on the same images: our own OVAM reimplementation as the baseline (also reporting OVAM's optimized-token results, flagged as a different setting), GoCA, … and if possible NERVE … PEARL … cite it for positioning … The target Table 5 would have two blocks: published numbers (marked as not paired) and the paired comparison."

### 3.1 What we did and how (configuration)

| Setting | Value |
|---|---|
| Images | the same 490 VOC-Sim + 707 COCO-Cap images and annotations for every method (ground truth checked identical on 1,197/1,197) |
| Query | the class word only, for every method |
| Code | each method from its own public repository and settings (GoCA github.com/Darkbblue/goca; NERVE; DiffSegmenter); OVAM, DAAM and SeeDiff (github.com/BAIKLAB/SeeDiff) from their released code; DAAM's attention read in BRACE's own generation pass so the images are identical |
| BRACE | regenerated with the prompt-corrected capture (runs/20), F's code imported unchanged; reproduces runs/20 to 0.001 |
| Threshold | chosen on VOC-Sim (5-fold out-of-fold), reused unchanged on COCO-Cap; also fixed 0.5 and each method's own rule (in `result.md`) |
| Post-processing | the same DenseCRF (OVAM's parameters) for every method; raw numbers also reported |
| Statistics | paired Δ (method − BRACE), 2,000-sample image bootstrap, 95% interval |
| Backbone | SD 1.5 (GoCA also SD 2.1-base and SDXL) |
| Files | `comparisons/official/result.md` (pushed, workplan Entries 11 and 14) |

**Which benchmarks the methods report** (checked in each paper): GoCA, NERVE, DiffSegmenter, SPARK and PEARL report only real-image benchmarks; none reports VOC-Sim or COCO-Cap. So they enter only through our paired reruns. SPARK has no public code and takes no class query: no-go. PEARL (CLIP/DINO, real images, multi-class): related work only.

**The published block is OVAM's reruns.** DAAM, Attn2Mask, Grounded Diffusion and DatasetDM did not publish VOC-Sim / COCO-Cap numbers themselves; Table 5's rows are OVAM's runs of them (the Table 5 caption says so; supp. line 434 says otherwise and must be fixed). Rescoring Attn2Mask's released masks gives 0.603 on VOC-Sim against OVAM's printed .687 (unexplained, to check).

### 3.2 The new Table 5

VOC-Sim / COCO-Cap mIoU. Paired rows: DenseCRF, VOC-selected threshold. Δ = method − BRACE.

| Block | Method | Extra model | VOC-Sim | COCO-Cap | Δ VOC-Sim | Δ COCO-Cap |
|---|---|---|---|---|---|---|
| **Paired, diffusion only, class word** | **BRACE (ours)** | — | **0.831** | **0.708** | — | — |
| | fine-only (ours, the rule's choice; runs/25) | — | 0.824 | 0.673 | −0.007 [−0.013, −0.001] | −0.036 [−0.046, −0.024] |
| | OVAM, class word (our run) | — | 0.703 | 0.568 | −0.128 [−0.143, −0.114] | −0.140 [−0.160, −0.120] |
| | DAAM (our run) | — | 0.727 | 0.644ᶜ | −0.104 [−0.119, −0.091] | −0.067 [−0.093, −0.039]ᶜ |
| | SeeDiff | — | 0.723 | 0.665ᶜ | −0.108 [−0.122, −0.094] | −0.046 [−0.070, −0.018]ᶜ |
| | DiffSegmenter | — | 0.795 | 0.688 | −0.036 [−0.049, −0.023] | −0.020 [−0.041, +0.001] |
| | GoCA, SD 1.5 | — | 0.753 | 0.599 | −0.078 [−0.095, −0.060] | −0.110 [−0.134, −0.084] |
| | GoCA, SD 2.1-base | — | 0.740 | 0.597 | −0.091 [−0.110, −0.072] | −0.111 [−0.134, −0.086] |
| | GoCA, SDXL | — | 0.797 | 0.606 | −0.034 [−0.048, −0.021] | −0.102 [−0.124, −0.078] |
| **Paired, with an extra vision-language model** | NERVE, class word | CLIP ViT-L/14 | 0.844 | 0.740 | +0.012 [−0.002, +0.028] | **+0.032 [+0.007, +0.060]** |
| | NERVE, 20-class vocabulary | CLIP ViT-L/14 | 0.839 | 0.747 | +0.008 [−0.007, +0.023] | **+0.039 [+0.013, +0.068]** |
| | SCLIP + DenseCRF | CLIP ViT-B/16 | 0.764 | 0.652 | −0.067 [−0.084, −0.052] | −0.057 [−0.084, −0.031] |
| | SCLIP seed + BRACE hierarchy (F, runs/20) | CLIP ViT-B/16 | 0.852 | 0.734 | | |
| **Paired, few-shot (optimized token; not training-free)** | OVAM, optimized token (our run) | — | 0.828 | 0.664 | −0.003 [−0.016, +0.010] | −0.044 [−0.063, −0.026] |
| | BRACE, optimized token (#4; runs/25) | — | 0.864 | 0.693 | +0.032 [+0.023, +0.041] | −0.015 [−0.030, −0.000] |
| **Published, not paired** (OVAM's Table 1; other images and post-processing) | DAAM | — | .662 | .484 | | |
| | Attn2Mask | — | .687 | .550 | | |
| | OVAM, class word | — | .704 | .582 | | |
| | OVAM, optimized token | — | .825 | — | | |
| | Grounded Diffusion (trains a module) | — | .621 | .502 | | |
| | DatasetDM (trains a decoder) | — | .803 | .593 | | |

ᶜ DAAM and SeeDiff need the class word in the caption, so on COCO-Cap they are scored on the 437 such images; BRACE on the same 437 images: 0.711. Our OVAM run with its own rule gives 0.696 / 0.546 (published: 0.704 / 0.582). SeeDiff (AAAI 2025, github.com/BAIKLAB/SeeDiff) has public code and publishes no numbers on these benchmarks. SCLIP uses only CLIP (no diffusion model).

**Reading.** BRACE is the best method that uses only the diffusion model. NERVE, which adds CLIP, is tied on VOC-Sim and better on COCO-Cap. With its own released thresholding rule NERVE drops to 0.812 / 0.663, below BRACE. Without DenseCRF every sign is the same.

**Caveats to print in the caption.** BRACE reads attention while generating, so it regenerates each image; the other methods segment the stored image (the two differ by under one grey level on most images; every sign holds on the closest subset). DiffSegmenter's own evaluation tunes its background score on the test set, so it has no fair own-rule row. All rows are exploratory; BRACE's VOC-Sim number is in-sample for its design.

**LaTeX draft** (replaces `tables/tab_published.tex`):

```latex
\begin{table}[t]
\centering
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & VOC-Sim & COCO-Cap & $\Delta$ VOC & $\Delta$ COCO \\
\midrule
\multicolumn{5}{@{}l}{\emph{Paired, same images, diffusion only, class word}}\\
\textbf{BRACE (ours)} & \textbf{.831} & \textbf{.708} & -- & -- \\
OVAM, class word~\cite{marcosmanchon2024ovam} & .703 & .568 & $-$.128 & $-$.140 \\
DAAM~\cite{tang2023daam} & .727 & .644$^*$ & $-$.104 & $-$.067 \\
SeeDiff~\cite{park2025seediff} & .723 & .665$^*$ & $-$.108 & $-$.046 \\
DiffSegmenter~\cite{...} & .795 & .688 & $-$.036 & $-$.020 \\
GoCA, SD 1.5~\cite{meng2026making} & .753 & .599 & $-$.078 & $-$.110 \\
GoCA, SDXL~\cite{meng2026making} & .797 & .606 & $-$.034 & $-$.102 \\
\midrule
\multicolumn{5}{@{}l}{\emph{Paired, with a CLIP model}}\\
NERVE~\cite{mahatha2025nerve} & .844 & .740 & +.012 & \textbf{+.032} \\
SCLIP + DenseCRF & .764 & .652 & $-$.067 & $-$.057 \\
\midrule
\multicolumn{5}{@{}l}{\emph{Paired, few-shot (optimized token)}}\\
OVAM, optimized token & .828 & .664 & $-$.003 & $-$.044 \\
\midrule
\multicolumn{5}{@{}l}{\emph{Published (OVAM's runs; not paired)}}\\
DAAM~\cite{tang2023daam} & .662 & .484 & & \\
Attn2Mask~\cite{yoshihashi2023attention} & .687 & .550 & & \\
OVAM~\cite{marcosmanchon2024ovam} & .704 & .582 & & \\
Grounded Diffusion~\cite{li2023open} & .621 & .502 & & \\
DatasetDM~\cite{wu2023datasetdm} & .803 & .593 & & \\
\bottomrule
\end{tabular}
\caption{\textbf{Comparison with competing methods.} Paired rows:
  the same 490 VOC-Sim and 707 COCO-Cap images, class word, DenseCRF,
  threshold selected on VOC-Sim and reused on COCO-Cap;
  $\Delta$ = method $-$ BRACE, bold when the 95\% bootstrap interval
  excludes zero. Published rows are OVAM's reruns on other images
  and post-processing.}
\label{tab:published}
\end{table}
```

**Slides:** `c3_N.tex` (5 frames: methods and go/no-go; sanity check; the paired table; paired Δ; caveats and actions).

---

## #4 BRACE with optimized tokens

> *JC:* "Please run BRACE with OVAM's optimized tokens … report it as a separate «optimized token» block … Check on which images the released tokens were optimized … state clearly that this block is no longer training-free."

### 4.1 Which images the tokens saw (N, Entry 5) — done

- OVAM released 57 per-image tokens (2–3 per class) optimized on **63 separate generated training images** (`voc_sim_train/`); the class token is their mean.
- **No overlap** with VOC-Sim (no shared class + seed among 500 rows; no identical pixels) or COCO-Cap (no shared seed; different prompts).
- So the block is **few-shot (57 annotated images), not training-free**, with no test leakage.

### 4.2 BRACE with the token (F, Entry 8 — original prompts) — done

Configuration: one live generation per image, both the class word and OVAM's VOC-Sim token read from the same generation; threshold fixed at 0.55; DenseCRF; paired; pre-registered (`runs/15_token_ladder`).

| | VOC-Sim | COCO-Cap |
|---|---|---|
| BRACE, class word | 0.830 | 0.696 |
| BRACE, optimized token | **0.861** | 0.681 |
| token − word | **+0.031 [+0.022, +0.039]** | −0.015 [−0.030, +0.000] |
| seed alone: token vs word | 0.588 → 0.766 | 0.444 → 0.625 |

Do our components still help with the token? (Δ with the token; with the class word in brackets)

| Component | VOC-Sim | COCO-Cap |
|---|---|---|
| hierarchy vs fine-only | +0.010 (+0.034) | **−0.017 [−0.028, −0.007]** (+0.012) |
| SA-guided vs uniform weights | +0.003 (+0.004) | +0.004 (**+0.024**) |
| cap vs no cap | +0.012 (+0.010) | −0.002 (−0.005) |

**Reading.** The token makes the seed much better (+0.18). With such a strong seed the hierarchy is no longer needed; on COCO-Cap it even hurts. So JC's "stronger message" (BRACE helps regardless of the query) **does not hold**; the "why" is that BRACE fixes a weak class-word seed, and a strong seed leaves nothing to fix. The COCO-Cap drop is concentrated in a few classes (sofa 0.70 → 0.34, car 0.76 → 0.66; person rises 0.28 → 0.62).

### 4.3 Rerun on the corrected prompts (F, runs/25) — done

F ran the full pre-registered ladder on the prompt-corrected capture (`runs/25_token_ladder_nr/result.md`, commit a64aff6); N's copy of the same run is an independent reproduction only. DenseCRF, paired, 95% intervals:

| | VOC-Sim | COCO-Cap |
|---|---|---|
| BRACE, class word (0.55) | 0.832 | 0.708 |
| BRACE, optimized token (0.55) | **0.864** | 0.693 |
| token − word, BRACE (0.55) | +0.031 [+0.022, +0.040] | −0.015 [−0.030, −0.000] |
| hierarchy − fine-only, class word, VOC-sel. | +0.007 [+0.001, +0.013] | **+0.036 [+0.024, +0.046]** |
| hierarchy − fine-only, class word, 0.55 | +0.036 | +0.010 [−0.005, +0.026] |
| hierarchy − fine-only, token, VOC-sel. | −0.011 [−0.017, −0.005] | −0.007 [−0.014, −0.000] |
| hierarchy − fine-only, token, 0.55 | +0.009 | **−0.019 [−0.031, −0.009]** |

Same picture as Entry 8, now on the corrected data: the token helps on VOC-Sim (+0.032) and slightly hurts on COCO-Cap, and with the token the hierarchy no longer helps. With the class word and each map at its own VOC-selected threshold, the hierarchy's COCO-Cap gain is **+0.036 [+0.024, +0.046]**, the cleanest version of the #1 number (one threshold rule, corrected data). Under F's registered rule (judged at 0.55), "the hierarchy helps" does not hold for either query.

### 4.4 Paper

- New "optimized token (few-shot)" rows in Table 5 (see #3) and a short paragraph in Sec. 5.1 (new numbering).
- Limitation sentence for Sec. 7: *"BRACE is designed to repair a weak class-word seed; with OVAM's optimized token, which raises the seed by 0.18, the hierarchy no longer helps on COCO-Cap (−0.017). The tokens are optimized on 57 annotated images disjoint from both benchmarks, so this setting is few-shot, not training-free."*

**Slides:** `c4_N.tex` (provenance), F's `c4_F.tex` (results).

---

## #5 Diffusion backbones

> *JC:* "evaluate fine-only, BRACE and GoCA with SD 1.5, SD 2.1 and SDXL on the existing VOC-Sim/COCO-Cap images, adding noise and reading the attention in one denoising pass … SDXL has only two attention scales, so it tests the coarse-to-fine principle itself. SD3/FLUX … out of scope … If the gain does not hold, better to know it now."

### 5.1 Configuration (pre-registered before any capture: `comparisons/backbones/PREREG.md`)

| Setting | Value |
|---|---|
| Images | the stored benchmark images and annotations (no regeneration, so the annotations stay valid) |
| Noise | image → latent (VAE), Gaussian noise at timestep 100 of 1,000 (≈ 0.95 × image + 0.33 × noise), fixed seed 0; the same level GoCA uses |
| Pass | **one** network call per image, the image's caption as prompt, class word as the query; no classifier-free guidance |
| Layers | SD 1.5 and SD 2.1-base: the frozen BRACE layers (identical layout); hierarchy 16 → 32 → 64. SDXL: all 70 CA layers, GoCA's reference layer, hierarchy 32 → 64 (**one step**, its only two scales) |
| Weights | SD 1.5; SD 2.1-base from `sd2-community/stable-diffusion-2-1-base` (official weights removed; hash-identical to a clone of Stability's repo); SDXL base 1.0 |
| Scoring | raw maps: threshold chosen on VOC-Sim (primary) and fixed 0.55; DenseCRF only at 0.55 (the full grid would need 240 CPU-hours) |
| GoCA | same images and masks, GoCA's own code (`config_model_15/21/xl.py`, t = 100) |
| Files | `comparisons/backbones/result.md`, `comparisons/goca/PREREG_backbones.md` (workplan Entries 14, 15) |

### 5.2 Results (one-pass setting; not comparable with the paper's 0.829 / 0.696)

Fine-only vs BRACE, raw, threshold chosen on VOC-Sim (VOC-Sim / COCO-Cap):

| Backbone | Fine-only | BRACE | Δ BRACE − fine-only |
|---|---|---|---|
| SD 1.5 (reference) | 0.784 / 0.612 | 0.793 / 0.650 | +0.009 [+0.003, +0.016] / **+0.038 [+0.029, +0.047]** |
| SD 2.1-base | 0.749 / 0.599 | 0.803 / 0.658 | **+0.054 [+0.044, +0.064] / +0.060 [+0.049, +0.071]** |
| SDXL (two scales) | 0.761 / 0.506 | 0.739 / 0.496 | **−0.022 [−0.030, −0.015] / −0.011 [−0.017, −0.005]** |

BRACE vs GoCA on the same backbone (raw, VOC-sel.):

| Backbone | GoCA | Δ BRACE − GoCA |
|---|---|---|
| SD 1.5 | 0.738 / 0.610 | +0.055 / +0.040 |
| SD 2.1-base | 0.719 / 0.586 | +0.083 / +0.072 |
| SDXL | 0.787 / 0.602 | **−0.048 / −0.107** |

All intervals exclude zero. At one fixed threshold of 0.55, BRACE beats fine-only on all three backbones, but that scores fine-only away from its best threshold (the same artefact as the +0.034, #1), so we do not claim it.

**Reading.** The hierarchy holds on three-scale UNets (SD 1.5, SD 2.1) and fails on SDXL, where only one coarse-to-fine step is possible; there GoCA scales with the stronger model, as it claims. This is the answer to "why only SD 1.5" and an honest limitation.

### 5.3 Paper

- New Sec. 5.y "Other backbones" with the two tables above (or a merged one) and the setting.
- Scope sentence for Sec. 7: *"We study multi-resolution UNet diffusion models. SD3 and FLUX use one resolution and joint text–image attention, so they offer no coarse-to-fine hierarchy and are out of scope. On SDXL, with two attention scales, the hierarchy does not help."*
- Open: Faiz to confirm the SD 2.1 copy.

**Slides:** `c5_N.tex` (3 frames: GoCA by backbone; hierarchy by backbone; GoCA overtakes on SDXL).

---

## #6 Consistency checks — done (fix list ready)

> *JC:* "One estimate throughout (0.829 vs 0.830, 0.696 vs 0.697); no raw and DenseCRF values mixed …; «Eq. ??» in Fig. 2; citations for Attn2Mask, Grounded Diffusion and DatasetDM; every pointer to the supplementary landing on the right section."

162 numbers traced: 137 match a committed result, 22 differ, 3 untraced. Full list: `comparisons/notes/entry_6_consistency.md`.

| JC's item | Cause | Fix (location) |
|---|---|---|
| 0.829 vs 0.830, 0.696 vs 0.697 | three generations of the same pipeline (v2, v3, a third) | use **v3: 0.829 / 0.696** everywhere; change Sec. 5.3 (p. 6, `sec/5_analysis.tex:35`) and Table 3 (p. 7, `tables/tab_hierarchy.tex:20`); Table 4's .830 is in-sample, label it |
| raw and DenseCRF mixed | Sec. 5.2 (p. 5–6, `sec/5_analysis.tex:24`): COCO pair after DenseCRF, VOC pair raw; abstract "five points" is raw while the rest is after DenseCRF; Table 5 mixes raw and DenseCRF rows | one stage per comparison: "0.818 vs 0.823, both after DenseCRF"; "costs up to three points after DenseCRF (five before)"; new Table 5 uses DenseCRF for all (#3) |
| "Eq. ??" in Fig. 2 (p. 5) | in JC's PDF the seed equation is split into Eqs. (3) and (4); the caption's `\ref{eq:weights}` has no matching label in that build | on Overleaf, put `\label{eq:weights}` on the seed equation (Eq. 4 in JC's PDF) and run LaTeX twice; check with a search for "??" before sending |
| missing citations | no `references.bib` entries | Attn2Mask (Yoshihashi et al., arXiv 2309.01369), Grounded Diffusion (Li et al., ICCV 2023), DatasetDM (Wu et al., NeurIPS 2023); BibTeX in `comparisons/notes/entry_venue_and_published_block.md` |
| supplementary pointers | 10 pointers say only "(supplementary material)"; the supplement numbers its own Table 1, Fig. 2, Sec. 5, which collide with the main paper's | point each to its section/table (list in the note); number the supplement S1, S2, … |

**Slides:** `c6_N.tex`.

---

## #7 Table 1 — deltas

> *JC:* "a COCO-Cap Δ, the folds won (x/5) for each row … and make the reference of each Δ explicit"; "no COCO-Cap value at all in the first three rows"; row 3 "hides the largest jump (.691 → .789)".

### 7.1 What is wrong with the current Table 1 (p. 4, `tables/tab_progression.tex`)

- No COCO-Cap Δ; rows 1–3 have no COCO-Cap value.
- Δ is not the difference between consecutive rows (each comes from its own paired run), e.g. .789 → .800 is +.011, printed +.012.
- Row 3's Δ (+.033) is measured against a row that is not in the table (.756, one pass through all 16 SA modules), which hides the .691 → .789 jump.

### 7.2 New Table 1, rebuilt from one capture on both benchmarks — done

Configuration: all nine rows of the current Table 1 rebuilt from **one** live generation per image (prompt-corrected capture, frozen runtime), 490 VOC-Sim and 707 COCO-Cap images, pre-registered (`comparisons/drafts/table1_ladder/PREREG.md`). Each Δ is paired against the row named in "vs", with a 95% bootstrap interval. Threshold chosen on VOC-Sim (5-fold OOF) and reused on COCO-Cap. Rows 5–8 with DenseCRF.

| Row | Change | vs | VOC-Sim | Δ VOC-Sim | folds | COCO-Cap | Δ COCO-Cap |
|---|---|---|---|---|---|---|---|
| 1 | CA, all 16 modules (reference) | — | .707 | — | — | .574 | — |
| 2 | drop the five 32-px CA modules | 1 | .714 | +.008 [+.003, +.012] | 5/5 | .572 | −.002 [−.009, +.004] |
| 3a | one SA pass, all 16 SA modules (alternative) | 2 | .759 | +.044 [+.038, +.052] | 5/5 | .632 | +.060 [+.050, +.071] |
| 3 | one SA pass, 64-px module only | 2 | .790 | **+.076 [+.069, +.085]** | 5/5 | .629 | **+.057 [+.045, +.069]** |
| 4 | second SA pass | 3 | .803 | +.012 [+.009, +.017] | 5/5 | .636 | +.008 [+.004, +.012] |
| 5 | DenseCRF | 4 | .816 | +.013 [+.010, +.016] | 5/5 | .651 | +.014 [+.012, +.017] |
| 6 | SA calls 10–20 | 5 | .820 | +.004 [+.001, +.007] | 4/5 | .657 | +.007 [+.004, +.009] |
| 7 | SA-guided CA weights (= fine-only reference) | 6 | .825 | +.005 [−.001, +.011] | 4/5 | .673 | +.015 [+.004, +.026] |
| 8 | hierarchy with one-sided cap (BRACE) | 7 | .832 | +.007 [+.001, +.013] | 4/5 | .708 | **+.036 [+.024, +.046]** |

- **COCO-Cap now has a value and a Δ for every row**, and every Δ names its reference.
- **The hidden jump is visible**: adding SA propagation (row 3) is the largest step, +.076 on VOC-Sim and +.057 on COCO-Cap; the all-16-module alternative (row 3a) sits beside it.
- **Reproduces the paper**: VOC-Sim at the early-screen threshold 0.45 gives .667 / .696 / (.757) / .790 / .803 / .816 / .821 (paper: .665 / .691 / (.756) / .789 / .800 / .813 / .817).
- **Threshold caveat for row 8**: rows 7 and 8 use different VOC-selected thresholds (0.425 vs 0.55). At one shared threshold of 0.55 the row-8 COCO-Cap Δ is +.010 [−.005, +.026] (#1). The caption must say so; a shared-0.55 column is in the LaTeX draft.
- LaTeX (fits the LNCS width): `comparisons/drafts/table1_jc7.tex`. Putting it into `paper/` is F's.

**Slides:** F's `c7_F.tex`.

---

## #8 Paper structure and cost

> *JC:* "Rename Sec. 5 to «Experimental results» and move Sec. 6 … into it …; Sec. 4 should include the hardware and software details … and the cost of the method: runtime and memory per image."

### 8.1 Structure

| Now | Proposed |
|---|---|
| 4 Experimental setup | 4 Experimental setup: benchmarks, protocol (with the override, #1), metrics, **hardware, software, runtime and memory** |
| 5 What each stage contributes | **5 Experimental results** |
| 6 Comparison with published methods | **5.1 Comparison** (new Table 5, #3) — first, so the narrowed claim is the first result |
| | 5.2–5.6 the former 5.1–5.5 (seed, envelope, cap, DenseCRF, calibration) |
| | 5.7 Optimized tokens (#4); 5.8 Other backbones (#5) |
| 7 Discussion and limitations | 6 Discussion and limitations (+ token scope, CLIP-seeded methods, SDXL, SD3/FLUX out of scope) |
| Conclusion | 7 Conclusion |

### 8.2 Cost per image — done

Configuration: 100 benchmark images (50 VOC-Sim + 50 COCO-Cap), SD 1.5 in float32, 30 steps, guidance 7.5, 512 px, batch 1, F's frozen runtime unchanged; median per image, warm-up discarded. Checks: the recorded image is bit-identical to plain generation and the maps equal the frozen code's output on all 100 images.

| Stage | A100 (40 GB) | V100 |
|---|---|---|
| generation alone | 2.66 s | 12.11 s |
| generation with attention recording | 6.94 s | 12.02 s |
| fine-only map / BRACE map | 0.034 s / 0.038 s | 0.080 s / 0.098 s |
| DenseCRF (one CPU core) | 5.92 s | 5.07 s |
| **total, BRACE** (fine-only is the same) | **12.9 s** (7.0 s on the GPU) | **17.2 s** (12.1 s on the GPU) |
| peak GPU memory | 5.1 GiB | 5.1 GiB |

One-pass setting (#5), per image on A100: SD 1.5 0.21 s / 5.1 GiB, SD 2.1 0.19 s / 5.5 GiB, SDXL 1.49 s / 15.6 GiB.

**Paragraph for Sec. 4 (ready to paste):**
> *Implementation and cost.* All experiments use Stable Diffusion 1.5 in float32 with its default PNDM scheduler, 30 denoising steps and classifier-free guidance 7.5 at 512 × 512 (PyTorch 2.4.1, CUDA 12.1, diffusers 0.36). We measured cost on 100 benchmark images at batch size 1. On an NVIDIA A100 (40 GB), generating an image takes 2.7 s; recording the attention BRACE needs raises this to 6.9 s (on a V100 both take 12.0 s). Building the maps costs 0.04 s for BRACE and 0.03 s for fine-only propagation, and DenseCRF adds 5.9 s on one CPU core. One image therefore takes 12.9 s end to end on the A100 (17.2 s on a V100), of which 7.0 s is GPU time, with a peak of 5.1 GiB of GPU memory. The hierarchy adds no measurable time or memory over fine-only propagation, and the method trains nothing.

(JC worried about storing 64×64 SA maps over many calls: the maps are reduced on the fly, so the peak stays at 5.1 GiB.)

**Slides:** `c8_N.tex` (structure); cost slide in the response deck.



---

## #9 Target venue: ICPR 2027 — done

| Item | ICPR 2027 (ICPRv) |
|---|---|
| Registration (author list fixed) | **1 March 2027**, AoE |
| Submission | **8 March 2027**, AoE |
| Notification / camera-ready | 14 June / 28 June 2027 |
| Format | Springer LNCS, one column, **15 pages + references** |
| Review | double-blind; mandatory self-review pages incl. an LLM-use disclosure |
| Conference | online only, 4–6 and 14–15 Oct 2027 |
| Preprints | arXiv allowed |

**Page fit.** The current text, converted to the ICPR template unchanged, fills about **13.5 of 15 pages** before any of JC's additions (new Table 5, backbones, tokens, cost, Table 1 columns). Needed: the `llncs` class, table captions above tables, Table 3 restructured (6.5 cm too wide). Move to the supplement: Table 4 (one sentence instead), Fig. 5, one of Figs. 3 and 6, the top half of Table 2.

**To decide with JC:** author list and order (fixed on 1 March); online-only venue OK; long paper (15 pages).

**Slides:** `c9_N.tex`.

---

## Summary for the meeting

| # | Status | One-line answer |
|---|---|---|
| 1 | text drafted | Reframe around clutter and calibration; disclose the override and its timing; drop "six times"; narrow to diffusion-only |
| 2 | done | Contributions 1–2 well supported; 3–4 rely on the supplement; add the rule's choice next to ours |
| 3 | done | BRACE best among diffusion-only methods (OVAM −0.13 / −0.14 on the same images); NERVE (with CLIP) better on COCO-Cap |
| 4 | done | No test leakage; the token helps VOC-Sim, and with it the hierarchy no longer helps on COCO-Cap |
| 5 | done | Hierarchy holds on SD 1.5 and SD 2.1, fails on SDXL; GoCA wins on SDXL |
| 6 | done | 22 fixes listed |
| 7 | done | Rebuilt from one capture: COCO-Cap value and Δ for every row, folds won, explicit references; SA propagation (+.076 / +.057) is the biggest step |
| 8 | done | Structure proposal; 12.9 s (A100) / 17.2 s (V100) and 5.1 GiB per image; the hierarchy is free |
| 9 | done | ICPR confirmed; about 13.5 of 15 pages used before the additions |
