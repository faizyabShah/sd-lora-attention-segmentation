Give Codex the following as the next locked experiment package.

## 1. SG12 + DenseCRF confirmation

**Goal:** test whether the already-selected SG12 observation-selection method improves the **actual final training-free pipeline**, rather than only the one-pass baseline.

The SG12 variant is already frozen:

\[
\boxed{\text{W/SG12/top50}}
\]

It selected the top 50% of the 248 SA observations by the registered leave-one-out observation-reliability score and won **5/5 VOC folds**, with OOF \(0.7873\rightarrow0.7967\). Its COCO transfer was also strong: \(0.6172\rightarrow0.6272\). Pasted markdown Pasted markdown

### Locked comparison

Do **not** retune SG12, DenseCRF, observation percentage, thresholds grid, folds, or any other hyperparameter.

Run:

\[
\textbf{A. Current final reference:}
\]

`W / ordinary two-pass SA / existing fixed DenseCRF`

versus

\[
\textbf{B. Candidate:}
\]

`W / SG12 top50 one-pass propagation / existing fixed DenseCRF`

Also report, diagnostically:

\[
\textbf{C.}
\]

`W / SG12 top50 without CRF`

so we can see what CRF itself changes.

Use the exact DenseCRF configuration already used in the current final FTF pipeline.

VOC-Sim remains the only selection dataset. COCO-Cap is frozen transfer only.

### What I need returned

**Selection / headline table**
- five VOC fold results;
- winner in each fold;
- OOF macro class mIoU:
  - final reference,
  - SG12+CRF candidate,
  - delta;
- candidate vote count;
- whether candidate passes the existing promotion rule.

**Full-dataset results**
For VOC and COCO:
- selected VOC threshold;
- macro class mIoU;
- mean image IoU;
- pixel IoU if already part of the standard package;
- recall / precision;
- all 20 per-class IoUs and delta versus the matching final reference;
- 20k paired bootstrap CI for mean-image-IoU difference;
- number of images improved / worsened / tied.

**Mechanism diagnostics**
Compare `SG12 raw` → `SG12+CRF`:
- TP added / FP added;
- TP removed / FP removed;
- boundary-local errors if existing tooling supports this;
- distant/separate-component FP counts from the AN1 diagnostic if available.

I specifically want to know whether CRF fixes SG12's residual local errors or whether SG12 and CRF interfere.

**Runtime**
Report:
- ordinary two-pass + CRF runtime;
- SG12 top50 + CRF runtime;
- observation-ranking overhead;
- propagation overhead;
- CRF overhead.

Do not claim SG12 is faster unless measured end-to-end.

### Success criterion

Primary:

\[
\boxed{\text{SG12+CRF beats two-pass+CRF under VOC OOF selection}}
\]

with at least 4/5 fold wins.

COCO does not select, but a practically interesting result would be:

\[
\boxed{\text{COCO}\ge 0.637}
\]

or at least a clear positive transfer relative to the current final pipeline.

---

# 2. SG14 — Selected-observation consensus propagation

**Goal:** test whether SG11 and SG12 are complementary.

SG12 decides **which whole SA observations are reliable**.

SG11 decides **how to robustly aggregate and accept pixelwise changes**.

Both independently validated strongly: SG11 `trim20/rho=.80` won 5/5 folds with OOF \(0.7984\), while SG12 `top50` won 5/5 with OOF \(0.7967\). Pasted markdown

Do **not** open a new grid.

Use exactly one preregistered combination:

\[
\boxed{
\text{top50 observations}
+
\text{20\% trimmed mean}
+
\rho=0.80
}
\]

## SG14 procedure

For each image/query, retain the same SG12-selected observation set:

\[
\mathcal I_{50}
=
\text{top 50\% observations by SG12 reliability}.
\]

There should be 124 retained observations, consistent with the existing SG12 implementation.

For each retained observation:

\[
z_i=A_i c.
\]

Compute directional agreement **only over the retained set**:

\[
r_\uparrow(p)
=
\frac{1}{|\mathcal I_{50}|}
\sum_{i\in\mathcal I_{50}}
\mathbf 1[z_i(p)>c(p)]
\]

\[
r_\downarrow(p)
=
\frac{1}{|\mathcal I_{50}|}
\sum_{i\in\mathcal I_{50}}
\mathbf 1[z_i(p)<c(p)].
\]

Compute the 20% trimmed mean over those same retained observations:

\[
m_{20}(p)
=
\operatorname{trim20}_{i\in\mathcal I_{50}} z_i(p).
\]

Final SG14 map:

\[
y(p)=
\begin{cases}
m_{20}(p),
&
\max(r_\uparrow,r_\downarrow)\ge0.80\\
c(p),
&
\text{otherwise}.
\end{cases}
\]

No DenseCRF in the initial SG14 selection experiment.

No alternative top-K values.

No alternative \(\rho\).

No median variant.

No two-pass variant.

This is explicitly a **complementarity test**, not another tuning screen.

### Required comparisons

Run against:

- `W/REF`
- frozen `W/SG11/trim20/rho0p80`
- frozen `W/SG12/top50`

### What I need returned

**Main selection**
- fold winner for all 5 VOC folds;
- OOF mIoU for REF, SG11, SG12, SG14;
- SG14 delta versus each;
- fold votes;
- frozen decision under the existing promotion rule.

**VOC and COCO**
- macro mIoU;
- VOC-selected threshold;
- recall / precision;
- mean image IoU;
- per-class results and deltas;
- paired image-IoU bootstrap CI;
- improved/worsened image counts.

**Complementarity diagnostics**
This is essential.

For each pixel, classify whether:

- SG12 changes REF but SG11 does not;
- SG11 changes REF but SG12 does not;
- both modify the pixel in the same direction;
- both modify it in opposite directions;
- neither modifies it.

Using GT only diagnostically, report for those groups:
- TP gains;
- FP gains;
- correct removals;
- harmful removals.

Also report:

- fraction of observations rejected by SG12;
- fraction of pixels rejected by SG11's consensus gate;
- fraction where SG12 filtering changes whether the \(\rho=.80\) consensus condition passes;
- correlation between SG12 observation reliability and per-observation agreement with the eventual useful/harmful pixel changes.

### Interpretation target

I want the result to answer:

> Does SG12 remove globally unreliable heads/calls while SG11 handles location-specific disagreement among the remaining observations?

If yes, that's a very clean mechanism.

### Success criterion

Strong success:

\[
\boxed{\text{SG14 OOF} > 0.7984}
\]

with ≥4/5 fold wins and positive COCO transfer versus both SG11 and SG12.

If SG14 does not beat either parent, report that clearly and treat them as largely redundant rather than continuing to stack them.

---

# 3. SG15 — Region refinement using a validated pixelwise prior

This is the revised SG13 direction.

**Goal:** SG4 showed that the fixed SPEC32 regions have strong oracle segmentation potential, but SG13 showed that directly assigning semantic labels to those regions from simple region statistics is poor.

So SG15 should **not ask the regions to recognize the target from scratch**.

Instead:

> use an already-good target segmentation as the semantic prior, and use SPEC32 only as a spatial regularizer/refinement structure.

Keep:

\[
\boxed{\text{fixed SPEC32 partition}}
\]

exactly as in SG13.

Do not retune the partition method or number of regions.

## Frozen semantic priors

Test these three already-established sources separately:

1. `W/P1` — ordinary two-pass SA.
2. `W/SG11/trim20/rho0p80`.
3. `W/SG12/top50`.

Do **not** select or alter the internal settings of these priors.

For each prior continuous map \(s(p)\), use its already-valid VOC threshold \(t\) to form:

\[
M(p)=\mathbf1[s(p)\ge t].
\]

For every SPEC32 region \(R_k\), compute:

\[
f_k=
\frac{1}{|R_k|}
\sum_{p\in R_k}M(p).
\]

This is the fraction of the region already predicted as foreground by the strong pixelwise prior.

Test two refinement families.

### SG15-A — majority region projection

Assign the entire region foreground if:

\[
f_k\ge q.
\]

Use only:

\[
q\in\{0.25,0.50,0.75\}.
\]

VOC folds may select \(q\).

This directly approximates the GT majority oracle but substitutes a validated prediction for GT.

### SG15-B — confidence snapping

Use a fixed rule:

\[
f_k\ge0.75
\Rightarrow
\text{set all pixels in }R_k\text{ foreground}
\]

\[
f_k\le0.25
\Rightarrow
\text{set all pixels in }R_k\text{ background}
\]

otherwise:

\[
0.25<f_k<0.75
\Rightarrow
\text{retain the original pixelwise prior inside }R_k.
\]

This family has **no tunable parameter**.

It is important because full region projection may be too destructive when a SPEC32 region crosses the true object boundary.

No DenseCRF in the initial SG15 screen.

No SG2 region growing.

No caption vocabulary.

No synonyms.

No learned region classifier.

No DiffSeg-style generic semantic assignment.

### Selection structure

Treat each prior separately:

- `P1 + SG15-A`
- `SG11 + SG15-A`
- `SG12 + SG15-A`
- `P1 + SG15-B`
- `SG11 + SG15-B`
- `SG12 + SG15-B`

Each candidate must be compared against **its own unrefined semantic prior**, not just W/REF.

This matters: SG15 succeeds only if the region prior adds value beyond the strong pixelwise method feeding it.

COCO remains transfer-only.

---

# What I need from SG15

### 1. Standard segmentation results

For every prior before/after SG15:

- five-fold VOC result;
- OOF mIoU;
- fold votes;
- full VOC mIoU;
- COCO mIoU with frozen VOC choice;
- threshold(s);
- recall / precision;
- per-class IoUs and deltas;
- paired image-IoU bootstrap CI;
- improved / worsened images.

### 2. Region-labelling quality

For each prior/refinement pair:

- foreground regions per image;
- region precision;
- region recall;
- region F1;
- number of oracle-positive regions missed;
- number of oracle-negative regions selected.

Keep the same diagnostic definition as SG13:

\[
\text{GT-positive region}
\iff
\frac{|R_k\cap GT|}{|R_k|}\ge0.5.
\]

### 3. Oracle-gap analysis

This is crucial.

For VOC and COCO report:

- SPEC32 majority-oracle mean image IoU;
- unrefined prior mean image IoU;
- SG15 refined mean image IoU;
- absolute oracle gap before;
- absolute oracle gap after;
- fraction of oracle gap closed.

Unlike SG13, I want to see a **positive** gap-closure fraction.

### 4. Changed-pixel accounting

For every SG15 variant report pixels modified by region refinement as:

- FN → TP
- FP → TN
- TP → FN
- TN → FP

and compute:

\[
\text{useful changes}
=
(\text{FN→TP})+(\text{FP→TN})
\]

versus

\[
\text{harmful changes}
=
(\text{TP→FN})+(\text{TN→FP}).
\]

Do this separately for:
- high-FG regions;
- high-BG regions;
- if relevant, full-majority projected regions.

This will tell us exactly why region snapping helps or hurts.

### 5. Class-specific analysis

Pay special attention to:

- person
- chair
- bird
- dining table
- aeroplane
- sheep
- train

Person is mandatory because SG13 catastrophically damaged it. I want to know whether using a strong prior prevents that failure.

### 6. Visual sheets

For representative examples show:

**RGB | SPEC32 regions | prior score map | prior binary mask | region foreground fraction \(f_k\) | SG15 result | GT | changed pixels**

Include at least:
- aeroplane
- bird
- person
- chair
- dining table
- one image SG15 helps strongly
- one image SG15 hurts strongly.

For changed pixels, distinguish beneficial and harmful changes in the visualization if the existing tooling supports it.

---

# What counts as success for SG15?

The primary comparison is **against the input prior**, not against plain REF.

For example:

\[
\text{SG12+SG15}
>
\text{SG12}
\]

under VOC OOF with ≥4/5 fold wins.

I would consider SG15 especially promising if it:

- gives a consistent positive OOF gain of at least ~+0.005;
- transfers positively to COCO;
- closes a meaningful positive fraction of the SPEC32 oracle gap;
- reduces boundary/noise errors without the catastrophic person-type failures seen in SG13.

If neither majority projection nor confidence snapping beats its source prior, stop this region-refinement direction despite the oracle headroom. That would tell us the oracle partition cannot currently be exploited reliably from our available semantic prediction.

---

## Important common protocol for all three

Tell Codex to keep these fixed across the package:

- class-word W track only;
- same 490 VOC-Sim images and five fixed folds;
- same 707 COCO-Cap images;
- VOC selects, COCO never selects;
- same primary macro class mIoU;
- same threshold-selection policy;
- same paired bootstrap procedure;
- no post-hoc configuration changes after COCO inspection;
- preserve the V100 recovery/audit distinction if these runs use the same environment as SG11/12;
- every new package must contain its own same-generation REF/control so cross-hardware historical masks are **not** used as an unconditional correctness gate.

And I would run them in this order:

**SG12 + DenseCRF → SG14 → SG15.**

The first tests the final system, the second tests the strongest current methodological hypothesis, and the third tests whether the huge SG4 oracle headroom is actually exploitable.