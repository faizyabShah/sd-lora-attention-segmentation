# SG18/T1 + unchanged DenseCRF: results for ChatGPT

## Experiment and decision rule

All 490 VOC-Sim and 707 COCO-Cap images were regenerated and evaluated on the frozen W class-word track. The four cells share each generated trajectory: one-pass SG18/T1 (SA calls 10–20), T1 plus the existing unchanged OVAM DenseCRF (`R2`), ordinary two-pass `W/P1`, and P1 plus the same DenseCRF. No CRF parameters or time windows were tuned. The five fixed VOC folds select thresholds on four folds and score the fifth. Full-VOC thresholds are then frozen for COCO. The **pre-existing final-pipeline reference is W/P1+R2**, not unprocessed P1.

## Main results

| Method | VOC threshold | Full VOC mIoU | Frozen COCO mIoU | COCO precision | COCO recall |
|---|---:|---:|---:|---:|---:|
| W/T1 | 0.425 | 0.7960 | 0.6286 | 0.6754 | 0.8670 |
| W/T1+R2 | 0.425 | 0.8122 | **0.6465** | **0.6963** | 0.8604 |
| W/P1 | 0.450 | 0.8007 | 0.6249 | 0.6629 | **0.8762** |
| W/P1+R2 | 0.450 | **0.8135** | 0.6375 | 0.6790 | 0.8683 |

T1+R2 versus P1+R2: **−0.0013 VOC mIoU**, **+0.0089 COCO mIoU**. On COCO, its mean-image IoU gain is **+0.0050**, with 20,000-resample paired 95% CI **[+0.00085, +0.00891]** and **381 better / 323 worse / 3 tied** images. On VOC, the mean-image change is **−0.0014**, CI **[−0.00588, +0.00347]**, and **210 better / 280 worse** images.

DenseCRF helps both unprocessed maps. It raises T1 mIoU by **+0.0163 VOC / +0.0179 COCO**, versus **+0.0128 / +0.0127** for P1. T1+R2 gains COCO precision over P1+R2 by **+0.0173**, while recall drops **−0.0078**. This is a better precision–recall tradeoff for COCO mIoU, not a uniform gain on VOC.

## Locked VOC selection

The candidate T1+R2 received **0/5 VOC fold votes**; P1+R2 received 5/5. Forced candidate OOF mIoU is **0.8106** versus **0.8135** for P1+R2, a **−0.0029** difference. The forced OOF paired mean-image difference is **−0.00287**, CI **[−0.00759, +0.00218]**. Thus T1+R2 **fails the preregistered VOC promotion policy**, despite its positive frozen COCO result. The frozen system remains P1+R2 under that policy. COCO was used only for transfer evaluation.

## Class pattern versus P1+R2

- **VOC:** T1+R2 improves 13/20 classes. Largest gains: bird **+0.0265**, dining table **+0.0181**, cow **+0.0168**. Largest losses: person **−0.0610**, chair **−0.0565**, cat **−0.0196**. The person and chair losses explain much of the small aggregate VOC deficit.
- **COCO:** T1+R2 improves 14/20 classes. Largest gains: bicycle **+0.0450**, aeroplane **+0.0422**, boat **+0.0224**, bus **+0.0219**. Largest losses: dining table **−0.0250**, potted plant **−0.0160**, cat **−0.0112**, person **−0.0085**. Transfer is positive overall but not universal.

At fixed threshold **0.45**, T1+R2 reaches **0.6510 COCO mIoU**. This is a diagnostic score: its VOC-selected threshold is **0.425**, which gives the legitimate frozen-transfer score **0.6465**. Do not replace the frozen result with the fixed-0.45 diagnostic.

## Interpretation for ChatGPT

The temporal T1 map appears to interact particularly well with unchanged DenseCRF on COCO. That is an **inference from the paired results**, not a claim about a measured CRF mechanism. The preregistered VOC selection still rejects it. It is worth distinguishing “best frozen COCO transfer in this comparison” from “promoted final system.” If considering any follow-up, preserve that distinction and avoid selecting a threshold or model from COCO performance.

Machine-readable details: `results.json` contains fold decisions, forced OOF, precision/recall, pixel and mean-image IoU, paired intervals and all 20 class IoUs. `per_class_results.csv` has the full class table. Slurm job **235305** completed 0:0; `completed.json` confirms full coverage.
