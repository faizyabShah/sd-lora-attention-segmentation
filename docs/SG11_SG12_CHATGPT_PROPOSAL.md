# User-supplied ChatGPT proposal: SG11, SG12 and SG-Diag

Source proposal preserved for review. The authoritative executable contract is protocol Section 84 and `configs/sg11_sg12_observation_reliability.json`. Numerical examples and hoped-for AUCs below are hypotheses, not results.

Given everything so far, I would **stop spending time on SG9/SG10-style graph engineering** and build directly on the only thing that showed a real signal: **variation across the 248 individual SA propagations**.

SG8 gave us two clues: directional consensus passed selection on W, while robust aggregation was consistently promising, especially on COCO. SG9 did not validate, and SG10's bottleneck-connectivity hypothesis was actively unsupported by its diagnostic AUC. Pasted text(20261019-151737)

I would run these next, in this order.

## Experiment 1 — Consensus-filtered robust propagation

Call it something like **SG11: Consensus + Robust Aggregation**.

Right now SG8-C says:

- compute all individual propagated maps \(z_i=A_i c\),
- calculate whether each observation moves a pixel up or down relative to CA seed \(c\),
- only trust the mean propagation where enough observations agree.

But C still uses the **ordinary mean** once the pixel passes consensus.

At the same time, SG8-A says that median/trimmed aggregation appears better than mean, although the W variants split the folds. That strongly suggests trying the two ideas together.

### Method

For every pixel \(p\):

\[
z_i(p)=(A_i c)(p)
\]

Compute:

\[
r_{\uparrow}(p)
=
\frac{1}{N}\sum_i
\mathbf{1}[z_i(p)>c(p)]
\]

and

\[
r_{\downarrow}(p)
=
\frac{1}{N}\sum_i
\mathbf{1}[z_i(p)<c(p)].
\]

Let

\[
r(p)=\max(r_{\uparrow},r_{\downarrow}).
\]

Then calculate a robust propagated value rather than the ordinary mean:

\[
m_{\text{robust}}(p)
=
\operatorname{median}_i z_i(p)
\]

or a trimmed mean.

Final map:

\[
y(p)=
\begin{cases}
m_{\text{robust}}(p), & r(p)\geq\rho\\
c(p), & \text{otherwise}.
\end{cases}
\]

I would test only a **small controlled grid**, not 40 variants:

- aggregator: `median`, `trim20`
- \(\rho\): `0.70`, `0.80`, `0.90`

Six candidates.

Why these? `rho=.80` already passed SG8 selection, and trim20/median were the strongest SG8-A variants. SG8-A showed W median at 0.7971 full VOC and trim20 at 0.7952, while trim20 reached 0.6294 COCO versus 0.6178 REF. Pasted text(20261019-151737)

### What I want Codex to report

I want exactly these results:

1. **5-fold VOC selection table**
   - winner per fold
   - OOF mIoU
   - REF OOF mIoU
   - delta
   - number of fold wins
   - whether it passes the same promotion criterion as SG8.

2. **Full VOC-Sim**
   - mIoU
   - selected threshold
   - recall / precision
   - per-class IoU deltas against W/REF.

3. **Frozen-threshold COCO-Cap transfer**
   - mIoU
   - delta vs W/REF
   - recall / precision
   - per-class deltas.

4. Compare against:
   - `W/REF`
   - `W/C/rho0p80`
   - `W/A/trim20`
   - `W/A/median`

5. Generate visual sheets for at least:
   - bird
   - aeroplane
   - person
   - chair
   - dining table
   - one class where C previously improved strongly, e.g. sheep/train.

For each image I want:

**image | CA seed | SG8-C | robust-only | SG11 | GT/prediction overlay**

### What would count as success?

The primary criterion:

\[
\boxed{\text{OOF mIoU} > 0.7970}
\]

with at least **4/5 fold wins**.

Even something around **0.800–0.802 OOF** would be meaningful if it also transfers positively.

For COCO, ideally:

\[
>0.625
\]

because that would start approaching the strong trim20 transfer result while retaining proper VOC selection.

The important scientific result would be:

> Consensus decides **where propagation is trustworthy**, while robust aggregation decides **what propagated value to use**.

That is a clean contribution.

---

# Experiment 2 — Observation selection instead of pixel selection

This is the one I find more interesting scientifically.

Call it **SG12: Reliability-weighted SA observations**.

Our current methods treat all 248 SA observations equally and then make decisions pixel-by-pixel.

But SG8-A's behaviour suggests something else:

> perhaps some heads/calls are systematically poor propagators.

Instead of throwing away extreme **values independently at every pixel**, identify unreliable **SA observations themselves**.

Suppose:

\[
z_i=A_i c,\quad i=1,\dots,248.
\]

For every observation \(i\), compute its agreement with the rest.

One simple training-free score is:

\[
s_i
=
\frac{1}{N-1}
\sum_{j\neq i}
\operatorname{corr}(z_i,z_j).
\]

Or preferably cosine similarity after flattening:

\[
s_i
=
\frac{1}{N-1}
\sum_{j\neq i}
\cos(z_i,z_j).
\]

Then either:

### Hard selection

Keep the best \(K\%\):

\[
\mu_K=
\frac{1}{|\mathcal K|}
\sum_{i\in\mathcal K}z_i
\]

with perhaps:

- top 50%
- top 70%
- top 80%
- top 90%.

### Or soft weighting

\[
w_i=
\frac{\exp(s_i/\tau)}
{\sum_j \exp(s_j/\tau)}
\]

and

\[
\mu_w=\sum_i w_i z_i.
\]

I would start with **hard selection**, because it is much easier to interpret.

### Why this is different from trimmed mean

Trimmed mean says:

> “For each pixel, throw away extreme propagated values.”

SG12 asks:

> “Is this entire head/call a trustworthy propagation observation?”

That is potentially much more interesting.

For example, in the bird failure cases, perhaps certain SA heads systematically bind the bird and branch together. If those observations are globally inconsistent with the majority, we could remove them before propagation.

### Very important diagnostic

Codex should additionally record the reliability score **by UNet location and head**.

I want a table like:

| block/call | head | mean reliability | selected frequency |
|---|---:|---:|---:|

Then determine whether particular heads/calls are repeatedly rejected.

If we find something like:

> late-block head 6 is consistently an outlier across images/classes,

that's a very useful finding.

### Results I want

Again:

- five-fold VOC winner voting,
- OOF mIoU,
- full VOC,
- frozen COCO transfer,
- per-class deltas,
- recall/precision.

But additionally:

**Observation analysis**
- histogram of \(s_i\),
- average \(s_i\) per head,
- average \(s_i\) per SA call/block,
- frequency each observation enters top-K,
- pairwise agreement matrix between observations.

And visual examples showing:

- REF propagation,
- top-50% propagation,
- top-70%,
- selected best variant.

### Success criterion

Anything that consistently beats SG8-C would be excellent:

\[
\text{OOF}>0.7970
\]

with ≥4 folds.

But even if segmentation gain is small, I would keep this experiment if it reveals **stable bad heads/calls**, because that gives us a principled direction for the next method.

---

# Experiment 3 — Diagnose useful vs harmful propagation before designing more methods

I would actually run this in parallel because it is cheap and could decide what SG13 should be.

This is not primarily another segmentation method. It is a **mechanism experiment**.

Call it:

**SG-Diag: What distinguishes a good SA correction from a bad one?**

We already learned from SG10 that graph connectivity does **not** distinguish true additions from false ones: bottleneck AUC was roughly 0.44 W and 0.47 T. Pasted text(20261019-151737)

Now test whether SG8's signals do.

For each pixel where REF changes CA appreciably:

\[
|\mu(p)-c(p)|>\epsilon
\]

label the change using GT as one of:

- TP addition,
- FP addition,
- correct foreground removal,
- incorrect foreground removal.

Then compute predictors:

\[
r_\uparrow,\quad r_\downarrow,\quad r=\max(r_\uparrow,r_\downarrow)
\]

plus:

- std across \(z_i\),
- MAD,
- interquartile range,
- median-minus-seed,
- mean-minus-seed,
- mean–median disagreement,
- fraction of observations very close to seed.

Then report how well each distinguishes:

### A
true additions vs false additions

and separately

### B
useful changes vs harmful changes overall.

Use ROC-AUC and PR-AUC.

### The result I specifically hope to see

Something like:

| Signal | TP add vs FP add AUC |
|---|---:|
| SA bottleneck | 0.44 |
| variance | ~0.5 |
| directional consensus | **0.65+** |
| mean–median agreement | **0.6+** |
| combination | **0.7+** |

If directional consensus actually gets a decent AUC, then suddenly SG8 has a mechanistic explanation rather than just an empirical +0.0095 result.

And if **mean–median disagreement** predicts mistakes, that directly motivates a third proper method:

\[
\text{trust propagation only when consensus is high AND mean/median agree.}
\]

That could become SG13.

---

# What I would tell Codex to do now

I would give Codex roughly this:

> Implement two new training-free experiments using exactly the existing SG8 evaluation protocol and do not modify the dataset splits, threshold-selection procedure, promotion rules, or W/REF definition.
>
> **SG11 — Consensus-filtered robust propagation:** combine SG8-C's directional consensus with robust aggregation. Test median and 20% trimmed mean at rho ∈ {0.70, 0.80, 0.90}. Where directional agreement is below rho retain the original CA seed; otherwise use the robust propagated value.
>
> **SG12 — Observation reliability selection:** compute all 248 propagated maps \(z_i=A_i c\). Score each observation by its mean cosine similarity with the remaining propagated maps. Test propagation using the top {50%,70%,80%,90%} most reliable observations. Keep this entirely training-free and calculate reliability independently per image/query.
>
> For both experiments run the existing 5-fold VOC-Sim selection procedure. Report fold winners, OOF mIoU, gain vs W/REF, full VOC mIoU, selected threshold, recall/precision, all 20 class IoUs and deltas, and COCO-Cap transfer using the frozen VOC-selected threshold. Compare directly to W/REF, W/C/rho0p80, W/A/trim20 and W/A/median. Do not select variants using COCO.
>
> Also run a diagnostic over REF propagation changes. Separate true-positive additions, false-positive additions, useful removals and harmful removals using GT. For each pixel record directional consensus, standard deviation, MAD, IQR, mean–median disagreement and propagation magnitude. Report ROC-AUC/PR-AUC for distinguishing true from false additions and useful from harmful changes.
>
> Produce representative visual sheets for bird, aeroplane, person, chair, dining table and sheep/train. For SG12 also report reliability by SA call and head and the frequency with which each observation survives top-K selection.
>
> Do not create further graph-connectivity or reciprocal-edge variants in this run.

That is what I would run next.

**My priority would be SG11 first, SG-Diag second, SG12 third.** SG11 is the most likely immediate improvement; the diagnostic is the most important for understanding *why* it works; SG12 is the more exploratory but potentially more novel direction.