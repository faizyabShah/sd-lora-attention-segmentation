# Audit of side F's Entries 8, 9 and 10, and of the runs/20 headline (side N, draft, 2026-11-19)

Draft for the review lines of `docs/JC_REVIEW_WORKPLAN.md` Entries 8–10. Not yet pasted into the workplan.
Repository state audited: `93dd9c9` (main). Nothing here changes any existing file.

## What was checked, and how

| Check | Method | Coverage |
|---|---|---|
| Numbers in Entries 8–10 vs cited result files | Every number in each entry traced by hand to `runs/15_token_ladder/result.md`, `runs/16_trace_034/result.md` and `runs/17_seed_walk_swap/result.md` | All numbers in the three entries |
| Source of the draft's +0.034 | **Recomputed on CPU** from the committed per-image counts of the original run (`outputs/segmentation/controlled_attention_cv/sg_combined_v3/{voc,coco}/shards/*.npz`), own bootstrap (2000, different RNG) | Thresholds, the +0.034, same-threshold Δ, COCO-best Δ, DenseCRF effect at matched thresholds |
| NERVE fidelity claim | Read N's `sanity_summary.json`; **recomputed** class-macro IoU from N's `per_item_counts.csv` (pooled per class) | Variants A, V, W |
| Variant A identity | Read `scripts/jc_swap_nerve.py` against `comparisons/nerve/PREREG_sd15.md`, `nerve_classword.py` and `repo_patch.diff` | Settings and patch |
| PREREG before result | `git log` dates of PREREG/result, `git diff` of every code change between them, **SLURM accounting** (`sacct`, read-only) for submit/start times of smoke, capture and score jobs | runs/15, runs/17, runs/20; runs/16 is post hoc |
| Reading vs decision rule | Rule text in each PREREG applied to the printed intervals | H, L, C (runs/15); P-CLIP, P-CA, S-H, S-W (runs/17); N and calibration arm (runs/20) |
| Paper consequences | `grep` and reading of `paper/sec/*.tex`, `paper/tables/*.tex`, `paper/supplementary.tex` at `174fa7e` | Lines listed in the last section |

**Not checked** (not in git): per-image maps in F's cache, F's per-image counts behind the 40/490 and 94/707
identical-count figures, F's copy of the patched NERVE repo, and the bootstrap intervals of runs/15, 16, 17 and 20
(no per-image counts committed for those runs). No GPU job was run.

---

## Entry 8 (runs/15 token ladder)

**Numbers.** Every number in the entry matches `runs/15_token_ladder/result.md` to the printed precision:
0.830/0.696 reproduction; token 0.861/0.681; Q +0.031 [+0.022, +0.039] / −0.015 [−0.030, +0.000]; token H, L, C;
class-word H, L, C; best token COCO-Cap cell 0.698 (row 3c); seed 0.588→0.766, 0.444→0.625; 238/490 and 243/707
bit-identical, 8e-4 / 3e-3; +0.022 VOC-selected (0.696 − 0.674); sofa, car, person per class. **No mismatch.**

**Pre-registration.** PREREG committed `95288dc` 18:41:39 on 16 Nov. The smoke jobs (253140–253143) were submitted
at 18:41:40. The smoke report failed on empty folds, and the fix `95381a4` (18:45:18) only guards `mi()` against an
empty subset; I checked the diff, and scoring is not touched. Full jobs 253146–253153 were submitted at 18:45:18,
scoring ran 20:41–21:58, and the result was committed at 22:02. No other code change between the PREREG and the
result touches `scripts/`, `src/`, `configs/` or `jobs/` (the intervening merge `ece97c4` contains only side N's
files). **In order.**

**Reading vs rule.** The verdict for the token, "none of H, L, C holds", is the rule's verdict: H's COCO-Cap
interval lies below zero, and the L and C intervals include zero. VOC-Sim is never below −0.005. **Correct.**

**Notes.**
1. *Omission that matters for the paper.* The entry gives the class-word numbers "for reference" but not their
   verdicts. Under the same pre-registered rule at 0.55, **only L holds with the class word**. H is
   +0.012 [−0.002, +0.026] (does not hold). C, cap vs no cap (raw), is **−0.005 [−0.011, −0.001] on COCO-Cap**
   (does not hold, and the interval is below zero). `result.md` prints these verdicts. The entry's wording ("with
   the optimized token none of the three holds") can be read as if all three hold with the class word. They do not.
2. *The fixed 0.55 favours the hierarchy on VOC-Sim.* The class-word H on VOC-Sim at 0.55 is +0.034, but fine-only
   + DenseCRF is 0.796 at 0.55 and 0.824 at its own threshold. This is the mirror image of Entry 9. With the token,
   VOC-selected fine-only + DenseCRF (0.870) is above the full pipeline (0.861) on VOC-Sim as well.
3. "Best token cell on COCO-Cap is fine-only + DenseCRF, 0.698" holds at 0.55 only. At the VOC-selected threshold
   the best token cell is the full pipeline, 0.681 (fine-only + DenseCRF 0.678).
4. runs/15 scores DenseCRF on a 0.05 grid (`scripts/jc_ladder.py`, `cc.THR_CRF`), while runs/16/17/20 use 0.025.
   runs/15 "VOC-sel." values are therefore not comparable with runs/17/20 "VOC-sel." values (for example, fine-only
   picks 0.45 here and 0.425 there). This explains the +0.022 vs +0.034 in Entry 8's open point.
5. *Reporting item missing.* PREREG "Reporting" says OVAM's published optimized-token scores are quoted as an
   unpaired reference. Neither `result.md`, the entry nor `c4_F.tex` quotes them.
6. *Superseded data.* The capture uses the uncorrected manifest prompts (runs/19 addendum: 50 VOC-Sim and about 88
   COCO-Cap generations are not the annotated image, and 2 COCO-Cap images are black from the safety checker). The
   corrected-prompt rerun, runs/25, is pre-registered (`e624868`), but only a smoke result exists. Entry 8's numbers
   should not enter the paper until runs/25 reports. Nodes erdos, wald and neumann are V100, as stated.

**Review line (ready to paste):**

**Review — N, 2026-11-19:** confirmed with notes. I traced every number to `runs/15_token_ladder/result.md` and
found no mismatch. The PREREG (`95288dc`, 18:41) precedes every job: smoke at 18:41:40, full jobs 253146–253153 at
18:45:18 per `sacct`, scoring 20:41–21:58. The only later change (`95381a4`) guards empty folds in the report and
does not touch scoring (diff read). The token verdict ("none of H, L, C holds") is the registered rule's verdict.
Notes: (1) under the same rule the class-word rerun also fails H (+0.012 [−0.002, +0.026]) and C (COCO-Cap −0.005
[−0.011, −0.001]); only L holds, and the entry should say so. (2) At 0.55, fine-only is scored away from its
VOC-Sim optimum, so the class-word VOC-Sim H of +0.034 is partly threshold, the mirror of Entry 9. (3) "Best token
cell = fine-only + DenseCRF" holds at 0.55 only. (4) runs/15's DenseCRF grid is 0.05, so its VOC-sel. values are
not comparable with runs/17/20. (5) The PREREG asked for OVAM's published token scores as an unpaired reference;
they are missing. (6) The capture predates the prompt correction (runs/19 addendum), so these numbers wait for
runs/25 before going into the paper. Not checked: the bootstrap intervals and per-image maps (not committed).

---

## Entry 9 (runs/16, source of the +0.034)

**Numbers.** All values match `runs/16_trace_034/result.md`: +0.0341 [+0.0227, +0.0450] (0.6962 vs 0.6621);
0.55: +0.0121 [−0.0018, +0.0264]; 0.425: −0.0387 [−0.0502, −0.0265]; COCO-best 0.600/0.525: +0.0176
[+0.0063, +0.0294]; raw +0.0219, +0.0049, +0.0172 with their intervals; VOC-Sim +0.0057 and OOF +0.0074; threshold
moves 0.425→0.525 and 0.55→0.60; precision 0.703; per class car +0.132, aeroplane +0.082, tv +0.074; 8 of 20 classes
lose at the same threshold (aeroplane, boat, bottle, bus, chair, motorbike, potted plant, sofa); dining table +0.139,
tv +0.114. **No mismatch.**

**Independent recomputation of the source (original committed run, `sg_combined_v3` shards):**

| Quantity, COCO-Cap, + DenseCRF | Entry 9 / report.md | N's recomputation |
|---|---|---|
| Full-VOC-Sim-best thresholds, fine-only (R0+R2) / hierarchy (R1+R2) | 0.425 / 0.550 | 0.425 / 0.550 |
| Hierarchy at 0.55 − fine-only at 0.425 | +0.0341 (0.6964 − 0.6623) | +0.0341 [+0.0224, +0.0454] (0.6964 − 0.6623) |
| Same threshold 0.55 | (new capture: +0.0121) | +0.0123 [−0.0026, +0.0265] |
| Each at its COCO-best (0.600 / 0.525) | +0.0175 | +0.0175 (0.7069 − 0.6894) |

The attribution to two different thresholds is **confirmed on the original run itself**, not only on the new capture.

**Pre-registration.** None. The entry labels the work post hoc and diagnostic, and nothing is selected. The script
was committed with the result (`cc2d732`). Jobs 253170–253171 ran 22:33–22:59, before the commit at 22:59:53.
**Acceptable as labelled.**

**Notes.**
1. *Wording conflict.* The Result bullet says the +0.034 is "part localisation (+0.018 at each map's best
   threshold)", and the entry then says this "agrees with protocol Section 134". It does not. Section 134 and the
   paper's oracle analysis (`5_analysis.tex:63`: per-image best thresholds tie, 0.896 vs 0.895; AP unchanged) say
   the hierarchy does **not** localise better within an image. The +0.018 at each map's best *global* COCO-Cap
   threshold is still cross-image calibration (and it uses COCO-Cap labels). The decomposition should read: about
   +0.018 cross-image calibration (diagnostic, COCO-best global threshold) plus about +0.016 from transferring
   fine-only's VOC-Sim threshold, which is over-inclusive on COCO-Cap. Neither part is within-image localisation.
   The proposed wording in the entry's open points is fine.
2. *Same artefact in a second paper claim (N's recomputation, post hoc).* The paper's "DenseCRF helps COCO-Cap six
   times more on capped maps" (+0.002 vs +0.014; `tab_refine.tex:15`, abstract) compares fine-only raw at 0.45 with
   fine-only + DenseCRF at 0.425. At matched thresholds on the original run, DenseCRF adds **+0.0138
   [+0.0118, +0.0162] to fine-only at 0.45** and +0.0155 at 0.425, against +0.0145 [+0.0126, +0.0168] for the capped
   map at 0.55. At every common threshold the capped map gains somewhat more (0.45: +0.018 vs +0.014; 0.55: +0.014
   vs +0.007), but nowhere near six times more.
3. *Replicates on the corrected data (runs/20).* For the BRACE seed (CAg): H = +0.036 [+0.025, +0.047] at the
   VOC-selected thresholds (0.425 vs 0.55), +0.010 [−0.004, +0.026] at 0.55, and +0.017 at each map's COCO-best
   (0.719 − 0.702). The +0.034 pattern is not an artefact of the old prompts.

**Review line (ready to paste):**

**Review — N, 2026-11-19:** confirmed with notes. I traced every number to `runs/16_trace_034/result.md` and found
no mismatch, and I recomputed the source from the committed counts of the original run (`sg_combined_v3` shards).
That gives thresholds 0.425/0.55, +0.0341 [+0.0224, +0.0454] (0.6964 − 0.6623), +0.0123 [−0.0026, +0.0265] at a
common 0.55, and +0.0175 at each map's COCO-best, so the two-threshold attribution holds on the original run too.
The work is post hoc and labelled so, with no selection. Notes: (1) calling the +0.018 at each map's best threshold
"localisation" contradicts Section 134 and the paper's per-image oracle (tie, 0.896 vs 0.895), which the entry says
it agrees with. Both parts are cross-image calibration: about +0.018 at a COCO-chosen global threshold and about
+0.016 from transferring fine-only's VOC-Sim threshold. (2) The same convention produces the paper's "DenseCRF gains
six times more on capped maps" (+0.002 vs +0.014). At matched thresholds DenseCRF adds +0.014 [+0.012, +0.016] to
fine-only at 0.45, against +0.014 for the capped map at 0.55 (N's recomputation, post hoc). (3) runs/20 reproduces
the pattern on the corrected data: +0.036 at the VOC-selected thresholds, +0.010 [−0.004, +0.026] at 0.55, +0.017 at
COCO-best. Not checked: runs/16's own intervals (no per-image counts committed).

---

## Entry 10 (runs/17 seed × walk swap)

**Numbers.** The table (16 cells), all ten registered contrasts with intervals, the thresholds, the per-class values,
the fidelity values, the bit-identical sensitivity (−0.013 [−0.035, +0.008]) and "same signs" for the raw and
bit-identical columns all match `runs/17_seed_walk_swap/result.md`. **No mismatch.** One wording point: in
"person 0.28 → 0.64 / 0.71" the left value is BRACE (CA × hierarchy), not the CLIP seed alone.

**Pre-registration.** PREREG `6a445fc` was committed at 22:55:58 on 16 Nov. Smoke jobs 253173–253180 were submitted
at 22:55:58 and the device checks 253183–253189 ran at 23:02–23:04 (renyi). The fix `150586f` (23:05:16) adds an
empty-subset guard in the report and a `--walk-tag` output suffix; I checked the diff, and scoring is not touched.
Full jobs 253191–253203 were submitted at 23:05:16, scoring ran 23:59–02:05, and the result was committed at 02:08.
The seed and walk jobs ran on renyi (A100), as stated. **In order.**

**Reading vs rule.** S-H is +0.008 [−0.012, +0.029] (crosses zero), so "the seed explains it" fails. P-CA is −0.019
[−0.031, −0.005], so "the walk explains it" fails. The result is mixed, with the interaction +0.052
[+0.034, +0.070]. This is the rule's conclusion, and the extra sentences in the entry do not overstate it.

**Claims about N's work.**
- *Four decimals.* N's `nerve_sd15_A/sanity_summary.json` gives 0.830195 (VOC-Sim) and 0.729468 (COCO-Cap), which
  round to 0.8302 and 0.7295. Recomputing from N's `per_item_counts.csv` (tp/fp/fn at 0.5, pooled per class) gives
  0.83020 and 0.72946. The claim is **confirmed**. The per-image figures (identical counts on 40/490 and 94/707, max
  |ΔIoU| 0.0016/0.0021) cannot be checked, because F's per-image counts are not committed.
- *Variant A.* `scripts/jc_swap_nerve.py` uses the vocabulary {class word, hyphen → space}, NERVE's 25 VOC
  background words (`voc_datasets.py:69` split), `ovs_voc21.yaml` ViT-L/14, size 336, fused-attn, SFSA, `car_template`
  (15), time-step index 45, noise seed 0 before each refinement, and `sfsa_head_dim` set from the UNet (40, deviation
  D2), on the patched repo (D1, the K=1 guard). This **is N's variant A** as registered in
  `comparisons/nerve/PREREG_sd15.md`, which answers the entry's open point. A one-class vocabulary without D1 gives
  NaN, so the four-decimal match is itself evidence that the patch was applied. That F's repo copy is unchanged
  cannot be verified from git.
- *Possible lead on the residual (unverified).* F computes the CLIP seed in a separate CLIP-only pipeline
  (`refinement="none"`), saves it as float32, and passes it to `refinement()` later. N's driver passes the
  in-pipeline tensor in one `pipe(...)` call. A dtype or RNG-order difference here could explain the 1e-3 per-image
  residual.

**Issues.**
1. *Caveat (1) and the "bit-identical" sensitivity point at the wrong reference image.* `jc_swap.py:65` compares the
   regeneration with `stored["image"]` from `<CACHE>/{vocsim,cococap}_maps`, which is the stored **paper capture**.
   The CLIP seed and the walk read the manifest image (`image_local`), which is the **official OVAM PNG** on which the
   annotation was drawn. The runs/19 addendum shows that the stored captures themselves differ from the official PNG
   on 101/707 COCO-Cap and 55/490 VOC-Sim images (mean difference ≥ 5 grey levels). The 243-image subset therefore does not control the mismatch the
   PREREG meant it to control, and caveat (1) misdescribes it. (The PREREG made the same conflation, so this is an
   error in the design text, not a change after registration.)
2. *The headline gap is superseded.* "NERVE's COCO-Cap lead (0.737 vs 0.696)" is inflated by that mismatch. On the
   528 matched COCO-Cap images the two tie: −0.001 [−0.028, +0.043] (runs/19 addendum, post hoc). On the
   prompt-corrected rerun (runs/20) NERVE − BRACE is +0.027 [+0.002, +0.054] on COCO-Cap and +0.011 [−0.003, +0.026]
   on VOC-Sim. With the corrected data the rule's outcome is still mixed:

   | Contrast, COCO-Cap, VOC-sel. + DenseCRF | runs/17 | runs/20 (corrected) |
   |---|---|---|
   | P-CLIP, walk − hierarchy, CLIP seed | +0.033 [+0.019, +0.049] | +0.016 [+0.004, +0.030] |
   | P-CA, walk − hierarchy, CA seed | −0.019 [−0.031, −0.005] | −0.033 [−0.045, −0.020] |
   | S-H, CLIP − CA, hierarchy | +0.008 [−0.012, +0.029] | +0.010 [−0.011, +0.034] |
   | S-W, CLIP − CA, walk | +0.060 [+0.035, +0.083] | +0.060 [+0.037, +0.084] |

   (The runs/20 values come from its registered rows: P-CLIP is the negated "BRACE hierarchy on NERVE's seed −
   NERVE"; P-CA is the CAg walk row against BRACE; S-H and S-W are the C rows. In runs/20 the walk is the ported
   noised pass on our regeneration, not NERVE's code on the official PNG.)
3. The slides `c3_F.tex` (two slides) still carry 0.737 vs 0.696 and the 243-image sensitivity. They need a
   correction entry from F pointing to the runs/19 addendum and runs/20 before the meeting.

**Review line (ready to paste):**

**Review — N, 2026-11-19:** issues (numbers and rule confirmed; one caveat wrong; headline superseded). I traced
every number to `runs/17_seed_walk_swap/result.md` and found no mismatch. The PREREG (`6a445fc`, 22:55) precedes
the full jobs 253191–253203 (submitted 23:05:16 per `sacct`, scoring 23:59–02:05). The only change (`150586f`) is a
report guard and an output suffix, with no scoring change (diff read). "Mixed" is the registered rule's outcome. The
NERVE cell is N's variant A (class word + NERVE's 25 background words, d = 40, K=1 guard; checked in
`jc_swap_nerve.py` against `PREREG_sd15.md`). Its 0.8302/0.7295 equals N's `sanity_summary.json` and N's recomputed
counts (0.83020/0.72946) to four decimals; the per-image residuals were not checked (not committed). Issues: (1)
caveat (1) and the "bit-identical" sensitivity compare the regeneration with the stored paper capture
(`jc_swap.py:65`), not with the official PNG that the CLIP seed, the walk and the annotation use. Those differ on
101 COCO-Cap images (runs/19 addendum), so the subset does not control the mismatch. (2) The 0.737 vs 0.696 lead is
superseded. On corrected prompts (runs/20) NERVE − BRACE is +0.027 [+0.002, +0.054] on COCO-Cap and +0.011
[−0.003, +0.026] on VOC-Sim. The swap stays mixed (P-CLIP +0.016 [+0.004, +0.030], P-CA −0.033 [−0.045, −0.020],
S-H +0.010 [−0.011, +0.034], S-W +0.060 [+0.037, +0.084]). (3) `c3_F.tex` needs the corrected numbers; a correction
entry is requested.

---

## runs/20 headline claims (corrected prompts)

**Numbers.** BRACE 0.833/0.709 (CAg × hierarchy + DenseCRF, 0.55), NERVE 0.843/0.735 (NV × walk, VOC-sel. +
DenseCRF), SCLIP + hierarchy 0.852/0.734, and "only per-image CLIP values (hA) help; every template and SD-only remap
fails" all match the tables and the reading block of `runs/20_new_reruns/result.md`. **No mismatch.**

**Pre-registration.** PREREG `abb272c` (16:58:55, 17 Nov) precedes every job: smoke 253336 at 16:58:55, full jobs
253356–253384 at 17:14:30, scoring from 19:29, and the result `8baa528` at 22:14. One procedural point: the full
jobs (and the second smoke at 17:11) were submitted **before** `c4b88ad` (17:19:25) committed the `--ids` option
they ran with. I read the diff. It only filters rows by id and changes no registered computation, and no scoring had
started when it was committed.

**Notes on the headline.**
1. *Mixed conventions.* The headline sets BRACE at 0.55 against NERVE at the VOC-selected threshold. The registered
   like-for-like contrast N is +0.011 [−0.003, +0.026] on VOC-Sim and **+0.027 [+0.002, +0.054]** on COCO-Cap
   (VOC-sel. + DenseCRF), and +0.007 / +0.035 [+0.010, +0.063] at 0.55 + DenseCRF. On resolved images only it is
   +0.029 [+0.006, +0.056]. At each method's COCO-best threshold it is still +0.027 (0.746 − 0.719), so the lead is
   not a threshold-transfer effect. The COCO-Cap lead is consistent, but its lower bound at the primary convention is
   only +0.002. VOC-Sim is a tie.
2. *This is NERVE under our evaluation conventions* (DenseCRF R2, VOC-selected threshold), on our regeneration, with
   a ported noised pass. NERVE's own binarisation in N's variant A sanity run gives 0.790/0.645 (VOC-Sim/COCO-Cap,
   official PNG, no DenseCRF). The PREREG also registered N "both at their own conventions", which the result does
   not report as a separate contrast.
3. NERVE uses a second model (CLIP ViT-L/14), so the row is not SD-only. SCLIP + our hierarchy (0.734) and hA + our
   hierarchy (0.735) reach NERVE's COCO-Cap level. The gap to NERVE is the seed source, not our propagation: with
   NERVE's seed our hierarchy is 0.016 below its walk, and with our seed it is 0.033 above.
4. Data checks: 18 VOC-Sim and 44 COCO-Cap images remain ≥ 2 grey levels from the official PNG, against "≈ 21"
   expected in the PREREG. The sensitivity table excludes them and changes no conclusion. The NERVE seed is computed
   on the step-1 PNG, while the walk, DenseCRF and BRACE use the capture image. These are identical only on 245/490
   and 177/707 images (cross-GPU drift). This is disclosed in the PREREG and is small, but the NV cells are not
   strictly single-capture.
5. The fidelity of the ported walk to runs/17 on matched images is median max |Δ| 1.8e-2 / 1.9e-2 (min–max maps),
   which is close but not "equal".

---

## Consequences for the paper (files at `174fa7e`)

### (a) The COCO-Cap hierarchy gain of +0.034 comes from two thresholds

Facts to use: +0.034 holds only with each map at its own VOC-Sim threshold (fine-only 0.425, hierarchy 0.55). At a
common threshold it is +0.012 [−0.002, +0.026] (original run: +0.012 [−0.003, +0.027]), and at each map's COCO-best
it is +0.018. On corrected data the three values are +0.036 / +0.010 (n.s.) / +0.017. Neither part is within-image
localisation (per-image oracle tie).

| Location | Text | Problem |
|---|---|---|
| `paper/supplementary.tex:424` | "+0.006 OOF, COCO +0.034" (hierarchy row of the selection table) | Two thresholds, not stated |
| `paper/sec/6_comparison.tex:3` | "Most of the COCO-Cap margin over our fine-only pipeline comes from the capped hierarchy: cluttered scenes benefit most from a coarse envelope." | About half of that margin is threshold transfer; at a common threshold the margin is not significant |
| `paper/tables/tab_published.tex:18–19, 22` | fine-only .662 vs BRACE .696, "frozen COCO-Cap" | Rows at different thresholds; caption silent |
| `paper/tables/tab_progression.tex:14–15, 18` | COCO .662 → .696, "COCO-Cap: VOC-frozen transfer" | Same |
| `paper/tables/tab_hierarchy.tex:10, 20, 25` | COCO + CRF .6622 vs .6966, "COCO-Cap: VOC-frozen threshold" | Each row at its own threshold; the caption should say so (Entry 9) |
| `paper/tables/tab_refine.tex:10, 12, 15` | "barely helps COCO-Cap (+0.002) … six times more (+0.014)" | +0.002 compares raw at 0.45 with DenseCRF at 0.425. At matched 0.45 DenseCRF adds +0.014 to fine-only too |
| `paper/sec/0_abstract.tex:2` | "DenseCRF gains six times more on capped maps in cluttered scenes." | Same as above; remove or restate at matched thresholds |
| `paper/sec/5_analysis.tex:24` | "0.693 vs 0.662 after DenseCRF" (plain hierarchy vs fine-only, COCO-Cap) | Probably the same own-threshold convention (not recomputed: the plain hierarchy is not in the committed v3 counts) |
| `paper/sec/5_analysis.tex:63` | "best global threshold moves from 0.425 to 0.55 … peaks higher on both benchmarks" | Correct, and the natural place to state the +0.034 decomposition as calibration |
| N's #1 reframing text (not yet written) | JC's "cluttered scenes, hierarchy +0.034" | Must use the decomposed numbers, not +0.034 as a localisation gain |

### (b) With the optimized token the components do not hold (Entry 8; old prompts, runs/25 pending)

| Location | Text | Problem |
|---|---|---|
| `paper/sec/0_abstract.tex:2` | "Each stage has a measurable role." | At the registered fixed 0.55, COCO-Cap: H n.s. and C negative even with the class word; with the token none of H, L, C holds |
| `paper/sec/1_intro.tex:12` | contribution 2, "Controlled ablations that isolate each choice" | Same; the ablations' COCO-Cap support depends on the threshold convention |
| `paper/sec/5_analysis.tex:35` | cap "+0.012 over the plain hierarchy … with 0.697 on COCO-Cap" | runs/15 class word, cap vs no cap at 0.55 raw: VOC-Sim +0.010, COCO-Cap −0.005 [−0.011, −0.001] |
| `paper/sec/4_protocol.tex:10`, `paper/sec/7_discussion.tex:4, 7` | "…won 4/5 / 5/5 folds and transferred"; "largest change that survived refinement in every fold and transferred" | "Transferred" holds only under the own-threshold convention (a) |
| `paper/sec/1_intro.tex:7`; `paper/tables/tab_published.tex:22`; `paper/supplementary.tex:434` | "no optimized tokens"; "optimized-token rows are excluded"; "which we do not use" | JC #4 needs a separate few-shot block: token seed +0.18; full pipeline +0.031 VOC-Sim / −0.015 COCO-Cap; with the token the hierarchy lowers COCO-Cap (−0.017 [−0.028, −0.007]). Use runs/25 numbers when available |

### (c) NERVE (CLIP seed + its walk) beats BRACE on COCO-Cap

Corrected data: 0.735 vs 0.709, N = +0.027 [+0.002, +0.054]; VOC-Sim +0.011 [−0.003, +0.026] (tie).

| Location | Text | Problem |
|---|---|---|
| `paper/sec/0_abstract.tex:2` | "how these signals are combined across spatial scales matters more than which graph operator is applied to them" | The swap shows that the operator matters and that its sign depends on the seed: with NERVE's seed the walk beats our hierarchy (+0.016), and with ours the hierarchy wins (+0.033) |
| `paper/sec/2_related.tex:7`; `paper/supplementary.tex:371, 383–384` | "the head weighting and the random walk did not [help]"; walk −0.0094 VOC-Sim, +0.0073 COCO-Cap | True for our adaptation on the CA seed only; NERVE's released walk on its own seed is the best COCO-Cap cell |
| `paper/sec/1_intro.tex:5`; `paper/sec/7_discussion.tex:4` | "We tested many such operators … most did not help"; "45 comparisons of alternative operators, such as random walks" | Needs the qualifier "on the SD-only cross-attention seed" |
| `paper/sec/6_comparison.tex:1–5`; `paper/tables/tab_published.tex` | Comparison only with published class-word numbers; no NERVE row | JC #3: add a paired, separately labelled block (second model, not SD-only): NERVE 0.843/0.735 vs BRACE 0.832/0.709 (VOC-sel.) or 0.833/0.709 (0.55); also report SCLIP + hierarchy 0.852/0.734 |
| `paper/sec/0_abstract.tex:2`, `paper/sec/1_intro.tex:7`, all tables | 0.829 / 0.696 headline | On corrected prompts BRACE is 0.833/0.709 (0.55). Every generated-set number predates the prompt correction |
| `paper/sec/4_protocol.tex:4`; `paper/sec/7_discussion.tex:7` | "We regenerate each image … from its released prompt and seed" | The released manifest text differs from OVAM's actual prompts (hyphens, trailing periods), and the safety checker blacked 2 images; the correction must be stated |
| `paper/sec/7_discussion.tex:7` | "COCO-Cap person, our weakest class at 0.28 IoU" | Still 0.28 on corrected data; NERVE reaches 0.71, which is worth stating as the main failure that a CLIP seed fixes |
