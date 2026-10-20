# Investigation: SG11/SG12 and SG2 + DenseCRF finalization failures

The GPU jobs finished **all 490 VOC-Sim and 707 COCO-Cap images**. Neither
failure was an out-of-memory error, interrupted generation, missing metric
shard or DenseCRF exception. Both stopped when their finalizers compared the
new V100 masks with historical A100/environment masks and required a maximum
per-image count discrepancy below 0.2% at **every** threshold.

## Evidence

- All 40 metric shards in each package are complete and disjoint, with the
  expected image IDs/folds/settings. Counts are nonnegative; image area and
  target area are constant across every method/threshold. Raw predictions
  are nested with threshold, and AP is finite for continuous maps.
- Independent SG11/SG12 and SG2+CRF runs used V100S/V100 GPUs on different
  nodes. Their REF confusion counts agree within **one pixel on VOC and two
  pixels on COCO**, across all 1,197 images and all 17 thresholds.
- A100 SG13 and historical A100 SG8 REF counts agree within **four pixels**
  across the same datasets. Thus each hardware family reproduces its own
  controls much more closely than the cross-hardware mask gate permits.
- The maximum SG11-versus-SG8 REF discrepancy is 8,185 pixels on VOC and
  39,917 on COCO. The maximum new-versus-old SG2-package REF discrepancy is
  26,282 pixels on VOC and 18,171 on COCO. These maxima are genuine measured
  differences and are retained in `count_audit.json`, including sample IDs,
  thresholds, quantiles and affected-image counts.
- Aggregate REF mIoU at .45 is much closer: SG11 V100 versus SG8 A100 is
  .788440 versus .788656 on VOC and .625872 versus .626588 on COCO.
  A small dataset-average change does not erase substantial individual
  image differences; the masks are **not** claimed to be identical.
- Every saved SG12 top-K selection agrees exactly with the ranking of its
  stored leave-one-out cosine scores. Selected counts are 124/174/199/224.
  Pairwise-cosine sums reproduce mean reliability, histograms account for
  the same eligible events for every feature, and all diagnostic image IDs
  match their metric shards.

## Independent live reproduction

Diagnostic job **234952** completed successfully on V100 in **1 min 3 s**.
It regenerated four discrepant samples: VOC sofa 441, VOC TV 482, COCO person
caption 472976 and COCO person caption 117816. These samples were chosen to
investigate failures, not to select methods or thresholds.

- Every SG11/SG12 count and every raw SG2-confirmation count reproduced the
  saved V100 measurements **exactly**, at every threshold on all four cases.
- Full post-DenseCRF counts were also reproduced **exactly** on the TV and
  person-472976 cases, for all three pipelines and all thresholds.
- The new locked SG2 operator was compared directly with Narmeen's original
  SG2 code on the **same captured Q/K**, using two identical class-word seeds
  rather than any learned token. Maximum raw differences for REF/P1/SG2 were
  <=2.99e-7 / 1.79e-6 / 2.99e-7; numerical assertions passed.
- Raw REF map ranges on these cases were approximately .215–.528. There is
  no evidence here that a nearly constant map or division by zero caused the
  failure. Precise attribution of hardware differences to generated RGB,
  native attention or their interaction is not established by these records.

A supplementary A100 audit (`234953`) was cancelled **before starting**:
the completed full-dataset SG13-versus-SG8 comparison already supplies the
A100 reference evidence. No full generation/evaluation rerun was required.

## Recovery and validity limits

The failures came from treating cross-hardware historical mask agreement as
an unconditional correctness gate. The numerical operators, current-run
reference consistency, selected-observation records and saved CRF outputs
passed their independent checks. This supports recovering **within-package
V100 comparisons**, with historical non-reproduction explicitly disclosed.
It does not establish bit-identical reproduction of earlier A100 runs.

The original default finalizers still enforce their historical gate. Recovery
requires an explicit `--recovery-evidence` file, created after the structural
and live audits. It binds the audit records and **all 120 original metric /
diagnostic shards by SHA256**. Changed measurements or changed audit evidence
are rejected. No metric shard was rewritten; no method, threshold grid, fold,
promotion guard or COCO selection rule was altered. Report headers and JSON
mark the audited recovery and require independent Claude review before any
paper or final-pipeline promotion.

## Recovered selection results

| Comparison | VOC OOF reference | VOC OOF candidate | Change | Candidate votes | COCO at VOC threshold |
|---|---:|---:|---:|---:|---:|
| SG11 trim20, rho=.80 vs one-pass REF | .787280 | .798400 | +.011120 | 5/5 | .621916 (REF .617163) |
| SG12 top50 vs one-pass REF | .787280 | .796658 | +.009378 | 5/5 | .627174 (REF .617163) |
| SG2 + DenseCRF vs two-pass + DenseCRF | .813209 | .811095 | -.002113 | 0/5 | .626111 (reference .637141) |

SG11 and SG12 pass their registered family selection in this V100 package.
SG11 meets ChatGPT's OOF >.7970 target, but its COCO score is below the hoped
.625. SG12 exceeds .625 on COCO, but its OOF score is below .7970. Neither
result alone establishes an improvement over the final system with DenseCRF.
The locked SG2+DenseCRF candidate does **not** replace the final pipeline.
Even at a common .45 threshold, its COCO score is .635039 versus .637141.

The SG2+DenseCRF row evaluates the locked candidate in every held-out fold;
the actual protocol-selected path retained the final reference in all five
folds. Full candidate/selection distinctions and paired intervals are in its
report. No additional method experiment has been submitted by this recovery.

## Files and reproducible commands

Audit records:
`outputs/segmentation/analysis/finalizer_reference_audit/`.
The small NPZ audit files retain only 64x64 score vectors/counts, not images
or full attention matrices. Live raw data were discarded after each sample.

Recovered reports:

- `outputs/segmentation/controlled_attention_cv/sg11_sg12_observation_reliability/report.md`
- `outputs/segmentation/controlled_attention_cv/sg2_densecrf_confirmation/report.md`

```bash
python scripts/investigate_attention_finalizers.py
python -c 'from pathlib import Path; from src.segmentation.reference_recovery import prepare_recovery_evidence; prepare_recovery_evidence(Path("outputs/segmentation/analysis/finalizer_reference_audit"))'
python scripts/finalize_sg11_sg12_observation_reliability.py --recovery-evidence outputs/segmentation/analysis/finalizer_reference_audit/recovery_evidence.json
python scripts/finalize_sg2_densecrf_confirmation.py --recovery-evidence outputs/segmentation/analysis/finalizer_reference_audit/recovery_evidence.json
```

The live audit JSON/arrays must already exist for those recovery commands;
CPU-only recovery cannot manufacture missing GPU audit evidence.
Precision/recall averaging differs between existing packages: SG11/SG12
reports pooled pixel rates, while the DenseCRF confirmation reports macro
class rates. mIoU and fold selection are macro class measures in both.
