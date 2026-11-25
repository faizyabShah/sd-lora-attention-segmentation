# Run 43 post hoc (descriptive): scores restricted to run 38's test / dev halves

Script: `/home/shafa/sg_comp/tmp/r43_sub38.py` (run via srun, cpu). Run 43's scores (one capture), class-macro mIoU,
subset = run 31's `split.json` (run 38 scored only the test halves: VOC-Sim 237, COCO-Cap 351). Not pre-registered;
explains why run 38 reported COCO-Cap ≈ 0.79–0.80 while run 43 reports 0.736.

| COCO-Cap, raw 0.55 | all 707 | run 38 test (351) | run 38 dev (356) |
|---|---:|---:|---:|
| no gate, uniform, V0 | 0.662 | 0.675 | 0.649 |
| no gate, SA weights, V0 | 0.690 | 0.691 | 0.688 |
| MK4 (H), V0 | 0.695 | 0.698 | 0.694 |
| **MK4-LR (H), uniform, V0** | **0.736** | **0.762** | **0.712** |
| MK4-LR (H), SA weights, V0 | 0.736 | 0.758 | 0.716 |
| MK4-LR (H), V0 + DenseCRF 0.55 | 0.744 | 0.773 | 0.719 |
| MK4-LR (H), SA weights, V3 | 0.738 | 0.760 | 0.713 |
| NERVE own | 0.702 | 0.712 | 0.698 |

Run 38 (regime X, SA weights, V0, MK4-LR) on its test half: raw 0.55 0.761, + CRF 0.55 0.768, + CRF at dev threshold
and dev γ 0.794 (primary), V3 0.801. Run 43 on the same images and read-out: raw 0.55 0.758, + CRF 0.55 (uniform) 0.773.

Reading: the gap is the image subset (+0.026 for MK4-LR; run 38's test half is easier for the gate) plus run 38's
dev-tuned threshold and γ with DenseCRF (+0.026). Same images and read-out agree across the two captures (0.761 vs 0.758).
