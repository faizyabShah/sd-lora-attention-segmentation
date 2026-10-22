# Pre-SG19 score-control diagnostic for ChatGPT

**Question:** Does late-stage attention provide extra confidence for the middle-stage additions, or do raw semantic scores already explain the apparent late-support signal?

The CPU-only analysis uses saved SG18 CA seed, middle-call map (10–20), and late-call map (21–30), with no image generation. Within each image, all five scores are evaluated on exactly the same pixels: raw `M>c`, CA background at its fixed 0.45 normalized threshold, and middle foreground at its fixed 0.425 threshold. Scores are raw bilinearly resized values; no score cutoff or segmentation method was tested. ROC-AUC and average precision are exact and tie aware, pooled over pixels.

| Score | VOC TP/FP ROC | COCO TP/FP ROC | COCO TP vs >16 FP | COCO TP vs >64 FP | COCO TP vs separate FP |
|---|---:|---:|---:|---:|---:|
| CA `c` | 0.5590 | 0.6198 | 0.6342 | 0.6390 | 0.6215 |
| Middle `M` | **0.6081** | **0.6861** | **0.6991** | **0.7049** | **0.7011** |
| Late `L` | 0.5835 | 0.6733 | 0.6857 | 0.6891 | 0.6754 |
| `min(M,L)` | 0.6006 | 0.6784 | 0.6892 | 0.6924 | 0.6851 |
| `(M+L)/2` | 0.5973 | 0.6814 | 0.6941 | 0.6986 | 0.6900 |

For ordinary TP versus FP, average precision is:

| Score | VOC AP | COCO AP |
|---|---:|---:|
| CA `c` | 0.8002 | 0.6302 |
| Middle `M` | **0.8417** | **0.7099** |
| Late `L` | 0.8262 | 0.6931 |
| `min(M,L)` | 0.8369 | 0.7003 |
| `(M+L)/2` | 0.8357 | 0.7036 |

All four contrasts' ROC-AUC, PR-AUC, and positive/negative counts for both datasets are in `auc_table.csv` and `results.json`.

**Conclusion:** Middle score beats late score on both datasets and every requested contrast. The min and mean combinations do not beat middle. The apparent raw-late reliability trend can be explained by semantic score strength; this control offers no evidence of additional late-stage discrimination. The original late-veto SG19 is **not justified** by these diagnostics.

**Replay limitation:** The original Pre-SG19 candidate checkpoint was deleted after finalization. Reconstructing candidates from the saved SG18 maps yields 6,816,320 VOC and 12,665,100 COCO pixels, versus 6,837,692 and 12,671,265 in the slightly different GPU replay. The control's late AUC (0.5835 VOC / 0.6733 COCO) closely matches the replay (0.5857 / 0.6751). All five scores in the control are compared on precisely the same SG18-reconstructed pixels.
