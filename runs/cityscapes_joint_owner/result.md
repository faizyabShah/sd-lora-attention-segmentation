# Owner-selected Cityscapes joint segmentation: completed run

All 65 requested images completed. Outputs: **585 raw + 585 refined exclusive label maps**, 130 comparison sheets, an offline browser and a verified ZIP. No inference traceback was found. Archive SHA-256 matches its recorded checksum.

## Main result

**The joint BRACE adaptation fails qualitatively through severe class bias.** At tau 0.55, it predicts only sidewalk, bus and train/tram anywhere in the 65 images. Sidewalk occupies **73.9% of mean image area**; person/rider, car, two-wheeler, building/wall, vegetation/terrain, road and sky receive zero pixels. Early geometric fusion inherits this bias: sidewalk occupies **58.0%** and person/rider receives zero pixels. These are prediction-area diagnostics, not accuracy scores.

The bias is already visible in the shared-query A3 seeds. For cs_0000, mean sidewalk seed activation is 0.463 versus person/rider 0.062; after propagation the corresponding normalized means are 0.880 and 0.123. Common peak scaling preserves these cross-class differences. The experiment does not isolate whether text-query context, token bias, propagation or their interaction is the main cause. This result concerns the new joint-query/decoder adaptation, not a re-evaluation of validated binary BRACE.

Raising the threshold removes low-scoring predictions but cannot change which class originally won at a pixel. Therefore threshold tuning alone does not correct this class bias.

## Threshold summary

Mean per-image **background percentage** after the specified R2 winner gate; these numbers do not measure segmentation quality.

| Threshold | BRACE | SCLIP-seeded | Geometric fusion |
|---|---:|---:|---:|
| 0.40 | 0.4% | 6.0% | 1.3% |
| 0.55 | 2.8% | 18.3% | 7.6% |
| 0.70 | 13.3% | 39.9% | 36.4% |

## Visual review

Seven completed comparison sheets were inspected (IDs 0, 89, 499, 447, 402, 394, 331), covering pedestrians, a rider, a car, a bus, a tram and a park street. This is not an exhaustive grading of all 65 images.

- **0 and 331:** BRACE labels almost everything sidewalk. SCLIP-seeded distinguishes vegetation from paving much better; fusion is dominated by sidewalk.
- **89:** SCLIP-seeded finds vegetation and paving, but leaves the two clear people as background at 0.55; BRACE/fusion mislabel them sidewalk.
- **499:** BRACE returns sidewalk across the image. SCLIP-seeded finds road but leaves the clear rider/bicycle as background at 0.55. Fusion recovers part of the two-wheeler while badly mislabelling much background.
- **447:** SCLIP-seeded gives a useful coarse car region at 0.55/0.70. BRACE returns sidewalk throughout. Fusion finds some car but has broad sidewalk spill at 0.55.
- **402:** all methods localize much of the bus. At 0.55 SCLIP-seeded/fusion reject more background than BRACE; 0.70 further trims the bus.
- **394:** all three overextend the train/tram label into the facade or track area. Higher thresholds remove some spill but do not produce a clean tram outline.

SCLIP-seeded is the most useful visually in these examples, but it also misses visible subjects. At 0.55 it produces person/rider pixels in only 4 images and no sky pixels anywhere. Without pixel ground truth, no mIoU, optimal threshold or scientific winning variant is claimed.

## Local artifacts

- Browser: `outputs/segmentation/cityscapes_joint_owner/index.html`.
- ZIP: `outputs/segmentation/cityscapes_joint_owner/cityscapes_joint_owner_results.zip`.
- All source selection hashes, masks, joint maps, config and logs are retained. The previous 50-image run is unchanged.
- No new run or parameter change was started during this review.
