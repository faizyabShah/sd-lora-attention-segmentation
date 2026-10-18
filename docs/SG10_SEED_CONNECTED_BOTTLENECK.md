# SG10 — Seed-Connected Bottleneck Propagation

> **Ownership and status (2026-10-19):** Claude owns implementation, smoke
> validation, production execution, finalization, and initial interpretation.
> Codex has not implemented or submitted SG10. The complete owner-supplied
> specification follows and must be preregistered before Claude launches it.

## Motivation

AN1 showed that the major failure of repeated SA propagation on cluttered scenes is not only local boundary bleeding.

A large fraction of new false positives:

- occur far from the target object;
- sometimes appear in entirely separate predicted components.

This suggests that self-attention can contain strong long-range semantic relationships that connect the target seed to spatially separate but visually or semantically related regions.

Ordinary propagation:

\[
y=A c
\]

or

\[
y=A^2c
\]

does not distinguish between:

1. expansion through a coherent chain of object-internal relationships;
2. a strong shortcut from the target to an unrelated distant region.

SG10 tests whether propagation can be made more reliable by requiring candidate foreground pixels to be **coherently connected back to the semantic CA seed**.

The core hypothesis is:

> True object completion should usually be reachable from the original semantic seed through a path of consistently strong SA relationships, whereas many distant false positives arise from isolated long-range shortcuts or weakly connected graph regions.

SG10 should use only the frozen diffusion attention and should remain training-free.

---

# 1. Inputs

Use the frozen setup:

- SD 1.5;
- frozen CA configuration;
- SA module 15 at 64×64;
- all calls 0–30;
- all 8 SA heads;
- genuine QK-derived SA;
- no RGB;
- no external segmenter;
- no learned classifier.

Run separately on:

1. ordinary class-word CA seed;
2. OVAM released optimized-token CA seed.

---

# 2. Graph used by SG10

SG10 requires a single graph on the 64×64 grid for connectivity analysis.

However, previous experiments showed that repeated propagation through a mean affinity can be inferior to observation-wise propagation.

Therefore SG10 should **not replace the normal observation-wise propagation output**.

Use the graph only to estimate connectivity.

Define:

\[
\bar A
=
\frac1N\sum_i A_i
\]

as a connectivity graph.

The ordinary propagated foreground evidence should remain:

\[
\mu
=
\frac1N\sum_i A_i c
\]

for one-pass analysis,

or the existing frozen observation-wise propagated result where appropriate.

Thus:

- observation-wise propagation provides semantic/spatial foreground evidence;
- the mean graph provides only a connectivity constraint.

Do not use:

\[
\bar A^2c
\]

as the SG10 prediction.

---

# 3. Seed set

SG10 needs a semantic core from which connectivity is measured.

Use a small fixed quantile-based seed definition based on the CA map \(c\).

Register:

\[
q_F \in \{0.90,0.95\}
\]

where the foreground seed set is:

\[
S_F
=
\{p:c(p)\ge Q_{q_F}(c)\}.
\]

This creates:

- top 10% CA pixels for \(q_F=0.90\);
- top 5% CA pixels for \(q_F=0.95\).

These are not clamped foreground predictions.

They are used only as graph sources.

This is important: SG10 must not repeat SG1's hard absorbing-anchor formulation.

---

# 4. Maximum-bottleneck connectivity

For every spatial pixel \(q\), define its connectivity to the seed set as:

\[
B(q)
=
\max_{\pi:S_F\rightarrow q}
\min_{(u,v)\in\pi}
W(u,v),
\]

where \(W\) is the graph affinity.

Interpretation:

- consider every path connecting a confident semantic seed pixel to \(q\);
- every path has a weakest edge;
- the score of the path is that weakest edge;
- choose the path whose weakest edge is as strong as possible.

Therefore \(B(q)\) answers:

> How strongly can this pixel be connected back to the semantic seed without crossing a weak graph bottleneck?

This is also known as a widest-path / maximum-capacity-path criterion.

It is not an absorbing random walk and does not iterate a foreground probability to convergence.

---

# 5. Graph direction

Because the frozen SA graph is directed, evaluate two clearly defined connectivity graphs.

## SG10-G1 — Forward directed graph

Use:

\[
W=\bar A.
\]

Connectivity follows the original SA transition direction.

## SG10-G2 — Symmetric graph

Construct:

\[
W(p,q)
=
\sqrt{
\bar A(p,q)\bar A(q,p)
}.
\]

Then use this graph for bottleneck connectivity.

This provides a cleaner undirected object-connectivity interpretation.

Do not introduce additional graph symmetrization schemes in the first screen.

The purpose is to determine whether directed versus reciprocal connectivity matters.

---

# 6. Normalization of connectivity score

The raw bottleneck score will likely be small because SA rows contain 4096 probabilities.

For each image, normalize only if necessary for combining with the propagated map.

A preferred scale-free option is rank/quantile normalization:

\[
\tilde B(q)
=
\text{empirical percentile rank of }B(q).
\]

Alternatively, if raw values are sufficiently stable across images, preserve the raw bottleneck score.

Before production runs:

1. inspect its numerical distribution on a small smoke set;
2. choose one normalization convention;
3. freeze it;
4. do not change it based on segmentation performance.

---

# 7. SG10 prediction variants

Test three conceptually distinct uses of connectivity.

## SG10-B0 — Reference

Ordinary propagated map:

\[
y=\mu.
\]

## SG10-B1 — Multiplicative connectivity

\[
y(q)
=
\mu(q)\tilde B(q).
\]

This requires both:

- high propagated foreground confidence;
- strong connection to the semantic core.

## SG10-B2 — Soft connectivity gate

Use:

\[
y(q)
=
\mu(q)
\left[
\alpha
+
(1-\alpha)\tilde B(q)
\right].
\]

Register:

\[
\alpha\in\{0.25,0.50\}.
\]

This prevents weak connectivity from completely eliminating foreground evidence.

## SG10-B3 — Expansion-only connectivity

This is the most important SG10 candidate.

If propagation increases a pixel relative to CA:

\[
\mu(q)>c(q),
\]

permit the increase only according to connectivity:

\[
y(q)
=
c(q)
+
\tilde B(q)
\max(0,\mu(q)-c(q)).
\]

If propagation decreases the pixel:

\[
\mu(q)\le c(q),
\]

keep:

\[
y(q)=\mu(q).
\]

Interpretation:

> CA remains the semantic core; SA may expand foreground confidence, but the expansion is attenuated when the candidate pixel is poorly connected to the core.

This directly targets AN1's long-range false-positive expansion.

---

# 8. Why B3 is particularly important

Suppose the CA seed detects a dog's head but misses its body.

If the body is connected through strong SA relationships:

\[
\text{head}
\rightarrow
\text{neck}
\rightarrow
\text{torso}
\rightarrow
\text{legs},
\]

then:

\[
\tilde B(q)
\]

should remain high and propagation can recover the body.

Suppose SA also strongly connects to another furry object in the background through one semantic shortcut.

If the rest of the path from the semantic core contains a weak bottleneck, then:

\[
\tilde B(q)
\]

should be low and the expansion is suppressed.

The goal is therefore not simply spatial closeness.

It is **graph-connected semantic continuity**.

---

# 9. Computing the bottleneck map efficiently

The bottleneck score can be computed without enumerating paths.

Use a widest-path algorithm starting from all seed pixels.

Initialize:

\[
B(s)=+\infty
\]

or the maximum graph weight for:

\[
s\in S_F.
\]

For every other node:

\[
B(q)=0.
\]

Using a max-priority queue, propagate:

\[
B(v)
\leftarrow
\max
\left(
B(v),
\min(B(u),W(u,v))
\right).
\]

This is analogous to Dijkstra's algorithm with:

- max instead of min over paths;
- min instead of sum along a path.

The graph has 4096 nodes.

Be careful with the dense graph.

If using all \(4096^2\) edges is too expensive, do not immediately use top-k pruning because prior experiments showed that top-k SA pruning was harmful.

First assess whether the widest-path computation can be implemented efficiently on the dense affinity.

If a sparse approximation becomes necessary, it must be documented as a separate implementation constraint rather than silently introduced.

---

# 10. Local-path diagnostic

In addition to the main dense SA graph, compute a diagnostic version restricted to local spatial neighbours:

- 4-neighbour;
or
- 8-neighbour.

Use the SA affinity only for those local edges.

This is diagnostic, not the primary SG10 candidate.

It answers:

> Is object completion supported by locally coherent SA paths, or are useful paths themselves dependent on long-range SA edges?

If local connectivity preserves most useful completion while eliminating leakage, this could motivate a later dedicated experiment.

Do not let this diagnostic select the SG10 winner in the first registered family unless it is explicitly registered beforehand.

---

# 11. Evaluation protocol

Use the existing five-fold VOC-Sim selection procedure.

Select only on VOC-Sim:

- seed quantile \(q_F\);
- graph type G1/G2;
- SG10 operator B1/B2/B3;
- \(\alpha\) where applicable;
- mask threshold.

COCO-Cap is transfer only.

Apply existing promotion rules.

Run class-word and optimized-token tracks independently.

Do not allow the token track to select a word setting or vice versa.

---

# 12. Required diagnostics

In addition to normal segmentation metrics, reuse AN1:

- newly added TP;
- newly added FP;
- removed TP;
- removed FP;
- FP/TP addition ratio;
- distant added FP;
- separate-component added FP.

For every newly added pixel, record or summarize:

\[
\tilde B(q).
\]

Compare the distributions for:

1. newly added TP;
2. newly added FP;
3. distant newly added FP;
4. separate-component newly added FP.

This is central.

The most important mechanistic test is:

> Do useful recovered object pixels have systematically stronger seed connectivity than harmful new false positives?

Report distribution summaries and, if useful, ROC/AP for using \(B(q)\) to distinguish newly added TP from newly added FP.

This diagnostic should not be used to tune a new threshold post hoc.

---

# 13. Component-level analysis

AN1 found some false positives in separate predicted regions.

For every connected predicted component after ordinary propagation, compute:

- whether it intersects the original CA foreground core;
- maximum bottleneck connection to the seed;
- mean bottleneck connection;
- component size;
- proportion of TP/FP pixels.

This should help determine whether low-connectivity components correspond to the separate-region leakage observed on COCO.

---

# 14. Visual diagnostics

For fixed images, visualize:

- CA seed;
- ordinary propagated map;
- bottleneck connectivity map \(B\);
- SG10 output;
- removed false-positive regions;
- true-positive regions lost by SG10.

Include examples where:

- propagation successfully expands into the target object;
- ordinary propagation leaks to a distant region;
- SG10 succeeds;
- SG10 incorrectly suppresses a valid disconnected object part.

Do not use these examples for setting selection.

---

# 15. Important constraints

Do not combine SG10 initially with:

- SG8 observation consensus;
- SG9 reciprocal propagation;
- DenseCRF;
- RGB boundaries;
- UNet feature compatibility;
- SG3 confidence locking;
- SG1 absorption;
- additional top-k graph pruning.

The first SG10 experiment must isolate whether seed connectivity itself contains useful information.

---

# 16. Scientific questions

SG10 should answer:

1. Are newly recovered true-positive pixels more strongly connected to the CA semantic core than newly introduced false positives?
2. Are distant false positives associated with weak graph bottlenecks?
3. Are separate predicted components particularly weakly connected?
4. Does a bottleneck criterion preserve object completion while reducing leakage?
5. Is directed connectivity sufficient, or is reciprocal/symmetric connectivity better?
6. Does SG10 help class-word and optimized-token seeds differently?
7. Does the optimized token already make most true foreground regions strongly connected?
8. Does SG10 improve precision without simply reverting toward the original CA seed?
9. Does it reduce the AN1 FP/TP addition ratio?
10. Does it transfer better to COCO-Cap than ordinary repeated propagation?

---

# 17. Expected interpretations

## SG10 improves VOC and COCO

Strong evidence that graph-connected semantic continuity is a useful structural prior for diffusion-attention segmentation.

## SG10 improves mainly COCO

Scientifically interesting and consistent with AN1:

> seed connectivity may specifically address clutter-driven long-range leakage.

This still must not override the registered VOC selection rule unless validated through the proper protocol.

## SG10 improves precision but loses too much recall

Connectivity is informative but the current gate is overly restrictive.

A future softer formulation may be justified.

## Bottleneck scores fail to separate TP and FP additions

Then the long-range leakage problem is probably not well characterized by graph connectivity, and SG10 should be stopped rather than expanded into a larger grid.

The central SG10 hypothesis is:

> A candidate foreground pixel should not only receive SA-propagated semantic evidence; it should be reachable from the original semantic CA core through a coherent high-affinity path.
