# Pre-SG19 score controls

CPU-only analysis of saved SG18 maps. All five scores use the same candidate pixels within each image. The earlier Pre-SG19 replayed maps differ slightly from SG18 caches, so its reported candidate counts and late-score AUC may differ.

| Score | VOC TP/FP ROC | COCO TP/FP ROC | COCO TP vs >16 FP | COCO TP vs separate FP |
|---|---:|---:|---:|---:|
| ca | 0.5590 | 0.6198 | 0.6342 | 0.6215 |
| middle | 0.6081 | 0.6861 | 0.6991 | 0.7011 |
| late | 0.5835 | 0.6733 | 0.6857 | 0.6754 |
| min_middle_late | 0.6006 | 0.6784 | 0.6892 | 0.6851 |
| mean_middle_late | 0.5973 | 0.6814 | 0.6941 | 0.6900 |

Exact ROC-AUC, PR-AUC, all four contrasts and support counts are in `auc_table.csv` and `results.json`.

## Interpretation

Middle score is the best of the five on both datasets for all four ROC contrasts. For ordinary TP versus FP, its VOC/COCO ROC-AUC is **0.6081/0.6861**, compared with **0.5835/0.6733** for late. Its average precision is **0.8417/0.7099**, compared with **0.8262/0.6931** for late. Neither `min(M,L)` nor `(M+L)/2` improves on M. CA is weaker still.

This control gives **no support for the proposed SG19 late-veto mechanism**. The previous late-score reliability trend does not establish uniquely late information: raw middle score discriminates better, including against distant and separate false positives. This diagnostic did not select a cutoff or evaluate any segmentation mask.

The saved SG18 maps reconstructed **6,816,320 VOC** and **12,665,100 COCO** candidate pixels. The earlier Pre-SG19 GPU replay had 6,837,692 and 12,671,265 respectively, due to slight cross-run map drift. Each row of this table compares all five scores on exactly the same candidate pixels from the SG18 cache. The late-only ROC-AUC here (0.5835 VOC / 0.6733 COCO) closely matches the replayed result (0.5857 / 0.6751).
