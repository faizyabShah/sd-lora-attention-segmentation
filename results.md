# Stable Diffusion 1.5 Cityscapes results

## Experiment setup

- Base model: `stable-diffusion-v1-5/stable-diffusion-v1-5`.
- LoRA ranks: 8, 16, 32, 64, and 128.
- LoRA training seeds: 1–5.
- Reported LoRA weights: final weights after 5,000 training steps.
- Frozen baseline seeds: 1–4; frozen seed 5 was not included in this evaluation.
- Evaluation images: 5,000 per run, aligned with the 5,000 Cityscapes crops and captions.
- Metrics: FID, KID, CLIPScore, BLIP image-text matching probability, BLIP cosine similarity, and LPIPS diversity.
- Arrows indicate the usual interpretation: lower FID/KID is better; higher semantic scores are better. LPIPS measures diversity rather than quality.

## Aggregate results

Values are mean ± sample standard deviation across seeds.

| Model | Rank | Checkpoint | Seeds | FID ↓ | KID ↓ | CLIP ↑ | BLIP ITM ↑ | BLIP cosine ↑ | LPIPS diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen SD 1.5 | — | — | 1, 2, 3, 4 | 67.12725 ± 0.23653 | 0.07027 ± 0.00031 | 24.56950 ± 0.02746 | 0.70382 ± 0.00149 | 0.44783 ± 0.00038 | 0.69228 ± 0.00349 |
| LoRA | 8 | 5000 | 1, 2, 3, 4, 5 | 35.05206 ± 2.35390 | 0.03247 ± 0.00324 | 22.07507 ± 0.19742 | 0.61782 ± 0.01532 | 0.44382 ± 0.00177 | 0.51331 ± 0.00453 |
| LoRA | 16 | 5000 | 1, 2, 3, 4, 5 | 31.77270 ± 1.99277 | 0.02850 ± 0.00309 | 22.22122 ± 0.09212 | 0.63683 ± 0.01146 | 0.44503 ± 0.00073 | 0.51748 ± 0.00732 |
| LoRA | 32 | 5000 | 1, 2, 3, 4, 5 | 33.77007 ± 2.53859 | 0.03076 ± 0.00392 | 22.32323 ± 0.19290 | 0.66235 ± 0.01296 | 0.44621 ± 0.00285 | 0.50767 ± 0.00123 |
| LoRA | 64 | 5000 | 1, 2, 3, 4, 5 | 35.09159 ± 3.40479 | 0.03148 ± 0.00467 | 22.03956 ± 0.17906 | 0.67506 ± 0.01512 | 0.44522 ± 0.00081 | 0.50571 ± 0.00477 |
| LoRA | 128 | 5000 | 1, 2, 3, 4, 5 | 32.33793 ± 3.12154 | 0.02787 ± 0.00493 | 22.47810 ± 0.33928 | 0.68955 ± 0.01151 | 0.44803 ± 0.00140 | 0.51473 ± 0.00470 |

## Per-run results

| Model | Rank | Checkpoint | Seed | Images | Skipped | FID | KID | CLIP | BLIP ITM | BLIP cosine | LPIPS mean | LPIPS std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen SD 1.5 | — | — | 1 | 5000 | 0 | 67.02825 | 0.07047 | 24.60822 | 0.70550 | 0.44790 | 0.69639 | 0.04892 |
| Frozen SD 1.5 | — | — | 2 | 5000 | 0 | 66.96067 | 0.07022 | 24.56014 | 0.70189 | 0.44820 | 0.69055 | 0.04638 |
| Frozen SD 1.5 | — | — | 3 | 5000 | 0 | 67.47801 | 0.07054 | 24.54375 | 0.70375 | 0.44729 | 0.68846 | 0.05174 |
| Frozen SD 1.5 | — | — | 4 | 5000 | 0 | 67.04208 | 0.06985 | 24.56587 | 0.70414 | 0.44792 | 0.69372 | 0.05154 |
| LoRA | 8 | 5000 | 1 | 5000 | 0 | 34.11207 | 0.03120 | 22.34947 | 0.63073 | 0.44405 | 0.50934 | 0.05700 |
| LoRA | 8 | 5000 | 2 | 5000 | 0 | 33.02374 | 0.02932 | 21.87333 | 0.61920 | 0.44403 | 0.52105 | 0.05376 |
| LoRA | 8 | 5000 | 3 | 5000 | 0 | 36.57010 | 0.03317 | 21.89704 | 0.60761 | 0.44206 | 0.51275 | 0.05393 |
| LoRA | 8 | 5000 | 4 | 5000 | 0 | 33.14950 | 0.03094 | 22.09128 | 0.63393 | 0.44654 | 0.51103 | 0.05625 |
| LoRA | 8 | 5000 | 5 | 5000 | 0 | 38.40490 | 0.03772 | 22.16423 | 0.59765 | 0.44244 | 0.51240 | 0.05226 |
| LoRA | 16 | 5000 | 1 | 5000 | 0 | 33.19904 | 0.03098 | 22.38514 | 0.65447 | 0.44613 | 0.51037 | 0.05254 |
| LoRA | 16 | 5000 | 2 | 5000 | 0 | 32.24282 | 0.02804 | 22.17234 | 0.63461 | 0.44457 | 0.52967 | 0.05870 |
| LoRA | 16 | 5000 | 3 | 5000 | 0 | 28.89774 | 0.02453 | 22.17334 | 0.62258 | 0.44446 | 0.51629 | 0.05235 |
| LoRA | 16 | 5000 | 4 | 5000 | 0 | 30.69912 | 0.02682 | 22.19589 | 0.63792 | 0.44455 | 0.51734 | 0.05236 |
| LoRA | 16 | 5000 | 5 | 5000 | 0 | 33.82480 | 0.03213 | 22.17939 | 0.63456 | 0.44546 | 0.51374 | 0.05230 |
| LoRA | 32 | 5000 | 1 | 5000 | 0 | 35.27868 | 0.03342 | 22.26370 | 0.66549 | 0.44961 | 0.50585 | 0.05083 |
| LoRA | 32 | 5000 | 2 | 5000 | 0 | 34.35821 | 0.03026 | 22.44321 | 0.68124 | 0.44850 | 0.50867 | 0.04923 |
| LoRA | 32 | 5000 | 3 | 5000 | 0 | 36.38526 | 0.03545 | 22.36732 | 0.64607 | 0.44261 | 0.50697 | 0.05143 |
| LoRA | 32 | 5000 | 4 | 5000 | 0 | 29.79925 | 0.02527 | 22.02247 | 0.66297 | 0.44470 | 0.50823 | 0.04749 |
| LoRA | 32 | 5000 | 5 | 5000 | 0 | 33.02893 | 0.02940 | 22.51944 | 0.65600 | 0.44564 | 0.50863 | 0.05093 |
| LoRA | 64 | 5000 | 1 | 5000 | 0 | 38.09574 | 0.03526 | 21.96286 | 0.68990 | 0.44583 | 0.50655 | 0.04768 |
| LoRA | 64 | 5000 | 2 | 5000 | 0 | 30.54556 | 0.02436 | 22.17924 | 0.66649 | 0.44442 | 0.50946 | 0.05061 |
| LoRA | 64 | 5000 | 3 | 5000 | 0 | 32.51898 | 0.02941 | 22.07831 | 0.66303 | 0.44618 | 0.50836 | 0.05103 |
| LoRA | 64 | 5000 | 4 | 5000 | 0 | 38.04220 | 0.03545 | 21.76940 | 0.69310 | 0.44438 | 0.49746 | 0.05190 |
| LoRA | 64 | 5000 | 5 | 5000 | 0 | 36.25547 | 0.03294 | 22.20798 | 0.66278 | 0.44528 | 0.50673 | 0.05276 |
| LoRA | 128 | 5000 | 1 | 5000 | 0 | 29.76661 | 0.02328 | 22.83788 | 0.67582 | 0.44959 | 0.52068 | 0.05657 |
| LoRA | 128 | 5000 | 2 | 5000 | 0 | 28.30843 | 0.02186 | 22.62378 | 0.69827 | 0.44920 | 0.51830 | 0.05062 |
| LoRA | 128 | 5000 | 3 | 5000 | 0 | 33.58387 | 0.03009 | 22.11518 | 0.70031 | 0.44691 | 0.51104 | 0.05130 |
| LoRA | 128 | 5000 | 4 | 5000 | 0 | 34.67056 | 0.03227 | 22.69658 | 0.69480 | 0.44636 | 0.50962 | 0.05184 |
| LoRA | 128 | 5000 | 5 | 5000 | 0 | 35.36018 | 0.03186 | 22.11707 | 0.67852 | 0.44809 | 0.51403 | 0.05069 |

## Source files

- Aggregate machine-readable results: `outputs/lora/outputs-metrics/sd15-seeds/rank_summary.json`.
- All per-run machine-readable results: `outputs/lora/outputs-metrics/sd15-seeds/all_metrics.json`.
## Checkpoint comparison on 1,000-image subsets

Each row is the mean ± sample standard deviation across five seeds. Every run uses the combined caption offsets 0 and 1 (1,000 images).

| Model | Rank | Checkpoint | Seeds | FID ↓ | KID ↓ | CLIP ↑ | BLIP ITM ↑ | BLIP cosine ↑ | LPIPS diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen SD 1.5 | — | — | 1, 2, 3, 4, 5 | 88.41399 ± 1.04914 | 0.07057 ± 0.00039 | 24.59777 ± 0.07508 | 0.70476 ± 0.00997 | 0.44572 ± 0.00093 | 0.69441 ± 0.00258 |
| LoRA | 8 | 1200 | 1, 2, 3, 4, 5 | 51.66557 ± 6.42137 | 0.02724 ± 0.00763 | 22.63585 ± 0.43698 | 0.57833 ± 0.03451 | 0.44292 ± 0.00537 | 0.52706 ± 0.01027 |
| LoRA | 8 | 2400 | 1, 2, 3, 4, 5 | 52.71844 ± 1.71298 | 0.03097 ± 0.00231 | 22.16958 ± 0.38235 | 0.59435 ± 0.01703 | 0.44053 ± 0.00390 | 0.50612 ± 0.01125 |
| LoRA | 8 | 3600 | 1, 2, 3, 4, 5 | 53.56242 ± 0.86809 | 0.03256 ± 0.00092 | 22.19294 ± 0.21387 | 0.60596 ± 0.01079 | 0.44037 ± 0.00118 | 0.50658 ± 0.00355 |
| LoRA | 16 | 1200 | 1, 2, 3, 4, 5 | 54.38196 ± 0.77526 | 0.03189 ± 0.00159 | 22.28274 ± 0.26835 | 0.51977 ± 0.02345 | 0.43609 ± 0.00472 | 0.52123 ± 0.00927 |
| LoRA | 16 | 2400 | 1, 2, 3, 4, 5 | 51.27814 ± 3.64260 | 0.02874 ± 0.00606 | 22.46216 ± 0.30303 | 0.61995 ± 0.03473 | 0.44286 ± 0.00345 | 0.50288 ± 0.00576 |
| LoRA | 16 | 3600 | 1, 2, 3, 4, 5 | 49.93695 ± 3.73333 | 0.02735 ± 0.00499 | 22.49268 ± 0.47346 | 0.62509 ± 0.02335 | 0.44157 ± 0.00282 | 0.50472 ± 0.01046 |
| LoRA | 32 | 1200 | 1, 2, 3, 4, 5 | 50.81434 ± 3.61062 | 0.02725 ± 0.00607 | 22.33212 ± 0.32322 | 0.56502 ± 0.04684 | 0.43781 ± 0.00347 | 0.50007 ± 0.00669 |
| LoRA | 32 | 2400 | 1, 2, 3, 4, 5 | 54.21035 ± 4.37129 | 0.03367 ± 0.00742 | 22.51885 ± 0.60428 | 0.61723 ± 0.02792 | 0.44030 ± 0.00359 | 0.49637 ± 0.01256 |
| LoRA | 32 | 3600 | 1, 2, 3, 4, 5 | 53.12439 ± 3.16700 | 0.03148 ± 0.00474 | 22.50257 ± 0.37208 | 0.65906 ± 0.02814 | 0.44371 ± 0.00269 | 0.50368 ± 0.00643 |
| LoRA | 64 | 1200 | 1, 2, 3, 4, 5 | 52.49914 ± 4.11286 | 0.02797 ± 0.00421 | 22.22554 ± 0.44443 | 0.55756 ± 0.02642 | 0.43527 ± 0.00448 | 0.50675 ± 0.00448 |
| LoRA | 64 | 2400 | 1, 2, 3, 4, 5 | 53.22236 ± 5.39892 | 0.03104 ± 0.00763 | 22.22502 ± 0.59390 | 0.64572 ± 0.02887 | 0.44148 ± 0.00368 | 0.49636 ± 0.00612 |
| LoRA | 64 | 3600 | 1, 2, 3, 4, 5 | 56.31148 ± 2.40504 | 0.03467 ± 0.00315 | 21.99237 ± 0.49147 | 0.67530 ± 0.01890 | 0.44303 ± 0.00159 | 0.49724 ± 0.00620 |
| LoRA | 128 | 1200 | 1, 2, 3, 4, 5 | 52.55513 ± 3.99551 | 0.03108 ± 0.00515 | 22.51188 ± 0.41579 | 0.53638 ± 0.05147 | 0.43333 ± 0.00545 | 0.50490 ± 0.00730 |
| LoRA | 128 | 2400 | 1, 2, 3, 4, 5 | 52.72677 ± 6.01507 | 0.03055 ± 0.00931 | 22.65908 ± 0.35135 | 0.63551 ± 0.06611 | 0.44147 ± 0.00470 | 0.51293 ± 0.01130 |
| LoRA | 128 | 3600 | 1, 2, 3, 4, 5 | 51.39382 ± 3.22372 | 0.02808 ± 0.00599 | 22.74493 ± 0.24131 | 0.67289 ± 0.03015 | 0.44646 ± 0.00227 | 0.50897 ± 0.00692 |

Machine-readable files: `outputs/lora/outputs-metrics/sd15-checkpoints/checkpoint_summary.json` and `outputs/lora/outputs-metrics/sd15-checkpoints/all_metrics.json`.

## Attention segmentation ablations

These results use the stored Stable Diffusion 1.5 cross-attention probabilities for
the annotated VOC-Sim captures. Of the first 100 captures, 99 have masks and were
evaluated. Attention maps use the conditional classifier-free-guidance branch,
min-max normalization, and a mask threshold of 0.4. The reported mIoU is the macro
mean of the accumulated foreground IoU across the four target classes.

### Denoising timestep windows

| Window | Steps | mIoU | Aeroplane IoU | Bicycle IoU | Bird IoU | Boat IoU |
|---|---:|---:|---:|---:|---:|---:|
| All | 0–30 | **0.70947** | **0.71389** | 0.68308 | **0.67504** | 0.76588 |
| Early | 0–9 | 0.57323 | 0.55234 | 0.53643 | 0.61382 | 0.59032 |
| Middle | 10–20 | 0.69877 | 0.67401 | **0.69683** | 0.63378 | **0.79045** |
| Late | 21–30 | 0.60799 | 0.58811 | 0.52133 | 0.60956 | 0.71296 |

### Cross-attention resolutions

| Resolutions | mIoU | Aeroplane IoU | Bicycle IoU | Bird IoU | Boat IoU |
|---|---:|---:|---:|---:|---:|
| 16 | 0.68482 | 0.68677 | 0.64053 | 0.64273 | 0.76923 |
| 32 | 0.38103 | 0.31184 | 0.25880 | 0.47806 | 0.47543 |
| 64 | 0.33258 | 0.34251 | 0.39800 | 0.23752 | 0.35229 |
| 16 + 32 | 0.68119 | 0.69440 | 0.62315 | 0.65662 | 0.75061 |
| 16 + 64 | 0.70191 | 0.71255 | 0.66534 | 0.64966 | **0.78009** |
| 32 + 64 | 0.54899 | 0.44450 | 0.57253 | 0.59299 | 0.58595 |
| 16 + 32 + 64 | **0.70947** | **0.71389** | **0.68308** | **0.67504** | 0.76588 |

Machine-readable files: `outputs/segmentation/voc_sim_100_ablations/time_window.json`
and `outputs/segmentation/voc_sim_100_ablations/resolution.json`.

### Joint timestep-window and resolution ablation

| Time window | Resolutions | mIoU | Aeroplane IoU | Bicycle IoU | Bird IoU | Boat IoU |
|---|---|---:|---:|---:|---:|---:|
| All (0–30) | 16 | 0.68482 | 0.68677 | 0.64053 | 0.64273 | 0.76923 |
| All (0–30) | 16 + 64 | 0.70191 | 0.71255 | 0.66534 | 0.64966 | 0.78009 |
| All (0–30) | 16 + 32 + 64 | **0.70947** | **0.71389** | 0.68308 | **0.67504** | 0.76588 |
| 0–4 | 16 | 0.47155 | 0.44065 | 0.49349 | 0.47802 | 0.47405 |
| 0–4 | 16 + 64 | 0.46072 | 0.42654 | 0.48228 | 0.46840 | 0.46568 |
| 0–4 | 16 + 32 + 64 | 0.45443 | 0.42611 | 0.46857 | 0.47718 | 0.44585 |
| 5–9 | 16 | 0.63904 | 0.62655 | 0.59755 | 0.62049 | 0.71158 |
| 5–9 | 16 + 64 | 0.63921 | 0.62917 | 0.59864 | 0.61732 | 0.71171 |
| 5–9 | 16 + 32 + 64 | 0.63913 | 0.63697 | 0.58641 | 0.63891 | 0.69423 |
| 10–14 | 16 | 0.67186 | 0.67553 | 0.62393 | 0.60394 | 0.78403 |
| 10–14 | 16 + 64 | 0.66754 | 0.66399 | 0.62715 | 0.58798 | **0.79105** |
| 10–14 | 16 + 32 + 64 | 0.68644 | 0.66448 | 0.67106 | 0.62756 | 0.78265 |
| 15–20 | 16 | 0.62778 | 0.67338 | 0.52245 | 0.58209 | 0.73321 |
| 15–20 | 16 + 64 | 0.66552 | 0.64142 | 0.64181 | 0.59432 | 0.78452 |
| 15–20 | 16 + 32 + 64 | 0.68568 | 0.64524 | **0.69140** | 0.62839 | 0.77767 |
| 21–25 | 16 | 0.57924 | 0.63439 | 0.44108 | 0.56005 | 0.68144 |
| 21–25 | 16 + 64 | 0.64233 | 0.62737 | 0.61017 | 0.58496 | 0.74684 |
| 21–25 | 16 + 32 + 64 | 0.64217 | 0.62057 | 0.59776 | 0.61533 | 0.73502 |
| 26–30 | 16 | 0.54735 | 0.60148 | 0.42540 | 0.54051 | 0.62201 |
| 26–30 | 16 + 64 | 0.57162 | 0.52771 | 0.55003 | 0.54583 | 0.66289 |
| 26–30 | 16 + 32 + 64 | 0.54372 | 0.53080 | 0.39646 | 0.59171 | 0.65589 |

The best global configuration remains all timesteps with all three resolutions
(mIoU 0.70947). Among the narrow windows, steps 10–14 with all three resolutions
perform best (mIoU 0.68644). The class-specific winners are all/all for aeroplane
and bird, steps 15–20/all resolutions for bicycle, and steps 10–14/16+64 for boat.

Machine-readable files: `outputs/segmentation/voc_sim_100_joint_time_resolution/results.json`
and `outputs/segmentation/voc_sim_100_joint_time_resolution/results.csv`.

### Cross-attention U-Net layer ablation

All captured denoising steps were used. The previous baseline contains the 15
cross-attention modules at resolutions 16, 32, and 64; `all layers` additionally
includes the single 8-resolution mid-block module.

| Layer group | Modules | mIoU | Aeroplane IoU | Bicycle IoU | Bird IoU | Boat IoU |
|---|---:|---:|---:|---:|---:|---:|
| All layers | 16 | **0.72317** | 0.69937 | 0.66588 | 0.75948 | 0.76796 |
| Previous baseline, no 8-resolution mid layer | 15 | 0.70947 | **0.71389** | **0.68308** | 0.67504 | 0.76588 |
| Down only | 6 | 0.64605 | 0.66468 | 0.64079 | 0.62113 | 0.65761 |
| Mid only | 1 | 0.55856 | 0.48285 | 0.51499 | 0.56438 | 0.67201 |
| Up only | 9 | 0.59571 | 0.57627 | 0.38634 | 0.60992 | **0.81031** |
| Down + mid | 7 | 0.66911 | 0.64302 | 0.62882 | 0.69905 | 0.70557 |
| Mid + up | 10 | 0.70634 | 0.68851 | 0.54955 | **0.78947** | 0.79781 |
| Down + up | 15 | 0.70947 | **0.71389** | **0.68308** | 0.67504 | 0.76588 |

| Individual cross-attention module | Resolution | mIoU |
|---|---:|---:|
| `down_blocks.0.attentions.0.transformer_blocks.0.attn2` | 64 | 0.07711 |
| `down_blocks.0.attentions.1.transformer_blocks.0.attn2` | 64 | 0.14483 |
| `down_blocks.1.attentions.0.transformer_blocks.0.attn2` | 32 | 0.33308 |
| `down_blocks.1.attentions.1.transformer_blocks.0.attn2` | 32 | 0.11706 |
| `down_blocks.2.attentions.0.transformer_blocks.0.attn2` | 16 | **0.56074** |
| `down_blocks.2.attentions.1.transformer_blocks.0.attn2` | 16 | 0.54681 |
| `mid_block.attentions.0.transformer_blocks.0.attn2` | 8 | 0.55856 |
| `up_blocks.1.attentions.0.transformer_blocks.0.attn2` | 16 | 0.49392 |
| `up_blocks.1.attentions.1.transformer_blocks.0.attn2` | 16 | 0.25942 |
| `up_blocks.1.attentions.2.transformer_blocks.0.attn2` | 16 | 0.32257 |
| `up_blocks.2.attentions.0.transformer_blocks.0.attn2` | 32 | 0.02893 |
| `up_blocks.2.attentions.1.transformer_blocks.0.attn2` | 32 | 0.15203 |
| `up_blocks.2.attentions.2.transformer_blocks.0.attn2` | 32 | 0.19168 |
| `up_blocks.3.attentions.0.transformer_blocks.0.attn2` | 64 | 0.16168 |
| `up_blocks.3.attentions.1.transformer_blocks.0.attn2` | 64 | 0.04842 |
| `up_blocks.3.attentions.2.transformer_blocks.0.attn2` | 64 | 0.03300 |

Machine-readable files: `outputs/segmentation/voc_sim_100_cross_attention_layers/results.json`
and `outputs/segmentation/voc_sim_100_cross_attention_layers/results.csv`.

### Exhaustive cross-attention layer-subset search

All 65,535 nonempty subsets of the 16 captured cross-attention modules were
evaluated using all denoising steps. Layer indices use the mapping below.

| Index | Module | Resolution |
|---:|---|---:|
| 0 | `down_blocks.0.attentions.0.transformer_blocks.0.attn2` | 64 |
| 1 | `down_blocks.0.attentions.1.transformer_blocks.0.attn2` | 64 |
| 2 | `down_blocks.1.attentions.0.transformer_blocks.0.attn2` | 32 |
| 3 | `down_blocks.1.attentions.1.transformer_blocks.0.attn2` | 32 |
| 4 | `down_blocks.2.attentions.0.transformer_blocks.0.attn2` | 16 |
| 5 | `down_blocks.2.attentions.1.transformer_blocks.0.attn2` | 16 |
| 6 | `mid_block.attentions.0.transformer_blocks.0.attn2` | 8 |
| 7 | `up_blocks.1.attentions.0.transformer_blocks.0.attn2` | 16 |
| 8 | `up_blocks.1.attentions.1.transformer_blocks.0.attn2` | 16 |
| 9 | `up_blocks.1.attentions.2.transformer_blocks.0.attn2` | 16 |
| 10 | `up_blocks.2.attentions.0.transformer_blocks.0.attn2` | 32 |
| 11 | `up_blocks.2.attentions.1.transformer_blocks.0.attn2` | 32 |
| 12 | `up_blocks.2.attentions.2.transformer_blocks.0.attn2` | 32 |
| 13 | `up_blocks.3.attentions.0.transformer_blocks.0.attn2` | 64 |
| 14 | `up_blocks.3.attentions.1.transformer_blocks.0.attn2` | 64 |
| 15 | `up_blocks.3.attentions.2.transformer_blocks.0.attn2` | 64 |

| Configuration | Selected layers | Omitted layers | mIoU | Aeroplane | Bicycle | Bird | Boat |
|---|---|---|---:|---:|---:|---:|---:|
| Best overall | 0,2,5,6,7,9,10,11,12,13,14,15 | 1,3,4,8 | **0.74090** | 0.71002 | 0.68653 | 0.78501 | 0.78203 |
| Best leave two out | All except 3,4 | 3,4 | 0.73384 | 0.69789 | 0.66332 | 0.78688 | 0.78726 |
| Best leave one out | All except 3 | 3 | 0.72834 | 0.71232 | 0.66660 | 0.76331 | 0.77115 |
| All 16 layers | 0–15 | None | 0.72317 | 0.69937 | 0.66588 | 0.75948 | 0.76796 |
| Previous 15-layer baseline | All except 6 | 6 | 0.70947 | 0.71389 | 0.68308 | 0.67504 | 0.76588 |

| Selected layer count | Best mIoU |
|---:|---:|
| 1 | 0.56074 |
| 2 | 0.65380 |
| 3 | 0.68244 |
| 4 | 0.69868 |
| 5 | 0.71060 |
| 6 | 0.71970 |
| 7 | 0.72583 |
| 8 | 0.73139 |
| 9 | 0.73517 |
| 10 | 0.73808 |
| 11 | 0.73946 |
| 12 | **0.74090** |
| 13 | 0.73853 |
| 14 | 0.73384 |
| 15 | 0.72834 |
| 16 | 0.72317 |

The exhaustive winner was selected and measured on the same 99 images, so its
reported value is exploratory and must be confirmed on held-out data.

Machine-readable files: `outputs/segmentation/voc_sim_100_exhaustive_layer_subsets/summary.json`
and `outputs/segmentation/voc_sim_100_exhaustive_layer_subsets/all_subsets.csv.gz`.

### Threshold and normalization ablation

The best 12-layer subset, all 16 layers, and the previous 15-layer baseline were
evaluated with thresholds from 0.10 through 0.90 at intervals of 0.05. Percentile
normalization clips each map to its 1st and 99th percentiles and scales that range
to [0,1].

| Layer configuration | Normalization | Best threshold | mIoU | Aeroplane | Bicycle | Bird | Boat |
|---|---|---:|---:|---:|---:|---:|---:|
| Best 12 layers | Percentile 1–99 | 0.45 | **0.74394** | 0.70610 | 0.69611 | **0.78859** | 0.78496 |
| Best 12 layers | Min-max | 0.40 | 0.74090 | 0.71002 | 0.68653 | 0.78501 | 0.78203 |
| Best 12 layers | Maximum | 0.50 | 0.73113 | 0.70807 | 0.67411 | 0.75506 | 0.78729 |
| All 16 layers | Percentile 1–99 | 0.50 | 0.73883 | 0.71355 | 0.69507 | 0.74339 | **0.80330** |
| All 16 layers | Min-max | 0.45 | 0.73031 | 0.71561 | 0.67061 | 0.73692 | 0.79811 |
| All 16 layers | Maximum | 0.55 | 0.72393 | **0.72090** | 0.65651 | 0.71820 | 0.80011 |
| Previous 15 layers | Percentile 1–99 | 0.45 | 0.71712 | 0.70987 | **0.69809** | 0.68045 | 0.78007 |
| Previous 15 layers | Min-max | 0.40 | 0.70947 | 0.71389 | 0.68308 | 0.67504 | 0.76588 |
| Previous 15 layers | Maximum | 0.50 | 0.68996 | 0.69785 | 0.65536 | 0.64064 | 0.76597 |

The calibrated global configuration is the best 12-layer subset with percentile
normalization and threshold 0.45. Its mIoU is 0.00305 above the same subset with
the original min-max/0.40 setting and 0.02077 above the uncalibrated all-layer
baseline. These settings were selected on the same 99 images and require held-out
confirmation.

Machine-readable files: `outputs/segmentation/voc_sim_100_threshold_normalization/results.json`
and `outputs/segmentation/voc_sim_100_threshold_normalization/results.csv`.

## Comprehensive Stage 1 and Stage 2 attention ablations

All experiments use the 99 annotated captures among VOC-Sim indices 0--99. The
calibrated cross-attention reference is the best 12-layer subset, all timesteps,
1st--99th percentile normalization, and threshold 0.45 (mIoU 0.74394).

### Stage 1: cross-attention ablations

| Family | Best configuration | mIoU | Aeroplane | Bicycle | Bird | Boat |
|---|---|---:|---:|---:|---:|---:|
| Class-adaptive layers | Per-class subsets, threshold 0.45 | **0.76908** | 0.72853 | 0.73465 | 0.80576 | 0.80737 |
| Timestep selection | Best 12 layers, steps 5--30 | 0.76246 | 0.72436 | 0.72543 | 0.78904 | 0.81099 |
| Head selection | Remove global head 4 | 0.74559 | 0.70873 | 0.70495 | 0.79423 | 0.77444 |
| Postprocessing | Remove components below 0.1 percent of image area | 0.74424 | 0.70653 | 0.69686 | 0.78867 | 0.78492 |
| Fusion weighting | Inverse-entropy head weights | 0.74396 | 0.70843 | 0.70002 | 0.78140 | 0.78597 |
| Calibrated CA reference | Uniform best 12 layers | 0.74394 | 0.70610 | 0.69611 | 0.78859 | 0.78496 |

The first five denoising invocations are detrimental for the selected layers:
dropping steps 0--4 raises mIoU by 0.01851. Class-dependent layer selection gives
the strongest Stage-1 result. Head weighting and morphology provide only small
changes.

### Stage 2: single-resolution self-attention refinement

| SA resolution | Best configuration | mIoU | Aeroplane | Bicycle | Bird | Boat |
|---:|---|---:|---:|---:|---:|---:|
| 8 | No SA refinement | 0.74394 | 0.70609 | 0.69610 | 0.78860 | 0.78496 |
| 16 | Steps 26--30, 4 iterations, continuous seed, residual 0.25 | 0.74809 | 0.70626 | 0.69815 | 0.79907 | 0.78889 |
| 32 | All modules/steps, 4 iterations, seed 0.70, background ratio | 0.80144 | 0.79445 | 0.75441 | 0.81222 | 0.84467 |
| 64 | All modules/steps, 4 iterations, seed 0.60, foreground-background difference | **0.83551** | **0.85340** | 0.74980 | **0.87619** | **0.86265** |

The 64-resolution result is the overall winner, improving over the calibrated CA
reference by 0.09157 mIoU. The 8-resolution SA does not help, while the benefit
becomes substantial at 32 and 64 resolution.

### Stage 2: progressive self-attention refinement

| Resolution sequence | Best mIoU | Best strategy |
|---|---:|---|
| 8 -> 16 | 0.74394 | No SA refinement |
| 16 -> 32 | 0.77418 | 4 iterations, residual 0.75, seed 0.70, background ratio |
| 32 -> 64 | **0.81808** | 4 iterations, residual 0.75, seed 0.60, background ratio |
| 8 -> 16 -> 32 | 0.74394 | No SA refinement |
| 16 -> 32 -> 64 | 0.80482 | 4 iterations, residual 0.75, seed 0.60, foreground |
| 8 -> 16 -> 32 -> 64 | 0.77437 | 4 iterations, residual 0.75, seed 0.70, foreground |
| 64 -> 32 | 0.80650 | 4 iterations, residual 0.75, seed 0.60, background ratio |
| 32 -> 16 | 0.74960 | 4 iterations, residual 0.50, seed 0.40, background difference |
| 16 -> 8 | 0.74394 | No SA refinement |
| 64 -> 32 -> 16 | 0.76051 | 4 iterations, residual 0.50, seed 0.60, background ratio |
| 32 -> 16 -> 8 | 0.74394 | No SA refinement |
| 64 -> 32 -> 16 -> 8 | 0.74394 | No SA refinement |

Progressive refinement performs best in the 32 -> 64 direction, but it remains
below direct 64-resolution refinement. Adding 8-resolution refinement consistently
removes useful detail.

All configurations were selected and measured on these same 99 images; their
values are exploratory until evaluated without retuning on the other datasets.

Machine-readable summary: `outputs/segmentation/voc_sim_100_stage1_stage2_summary/summary.json`.

<!-- FULL_VOC_SIM_RESULTS_START -->
## All dataset: full VOC-Sim

Evaluated **490 annotated images across 20 classes**. All entries were generated and scored in memory; generated images and attention tensors were not saved.

**Unoptimized OVAM baseline:** 0.65404 foreground mIoU.
**Best regular ablation:** 0.77773 foreground mIoU, `self_attention/r64/module__up_blocks.3.attentions.2.transformer_blocks.0.attn1__iter4__continuous__non_residual` (+0.12369 versus OVAM).

The primary metric is the macro mean over 20 foreground class IoUs; each class IoU pools pixel counts over that class's images. All listed winners were selected on these same 490 images, so comparisons among configurations are exploratory until confirmed on another dataset.

Integrity checks: every one of the 6,261 regular configurations contains 490 images and 20 classes; all 20 resumable checkpoints contain the expected unique manifest IDs; and the OVAM reference agrees with four equivalent aggregation paths within 0.00000017 mIoU (at most 24 differing pixel decisions across the full evaluation, from floating point reduction order).

The released OVAM images and five local reproductions are not byte-identical, but their mean absolute pixel errors are only 0.12--0.53 intensity levels on the 0--255 scale. This supports using the released masks with the reproduced attention maps: the image composition and mask geometry are aligned, with small numerical differences from the generation environment.

### Best configuration in each experiment family

| Experiment | Best configuration | mIoU | Δ vs. OVAM |
|---|---|---:|---:|
| baseline | `ovam_unoptimized` | 0.65404 | +0.00000 |
| layers | `mid_up` | 0.68279 | +0.02875 |
| resolution | `r8_r16` | 0.71219 | +0.05815 |
| self_attention/progressive_16-32 | `iter4__non_residual__seed_t0.30__foreground` | 0.74129 | +0.08725 |
| self_attention/progressive_16-32-64 | `iter2__non_residual__seed_t0.30__foreground` | 0.75767 | +0.10363 |
| self_attention/progressive_16-8 | `iter1__residual_a0.50__seed_t0.30__background_difference` | 0.69135 | +0.03730 |
| self_attention/progressive_32-16 | `iter2__residual_a0.50__seed_t0.30__background_difference` | 0.72013 | +0.06608 |
| self_attention/progressive_32-16-8 | `iter2__residual_a0.50__seed_t0.30__background_difference` | 0.69439 | +0.04035 |
| self_attention/progressive_32-64 | `iter4__non_residual__seed_t0.30__foreground` | 0.76835 | +0.11431 |
| self_attention/progressive_64-32 | `iter4__residual_a0.75__seed_t0.70__background_difference` | 0.75575 | +0.10171 |
| self_attention/progressive_64-32-16 | `iter4__residual_a0.50__seed_t0.70__background_difference` | 0.72825 | +0.07420 |
| self_attention/progressive_64-32-16-8 | `iter4__residual_a0.50__seed_t0.30__background_difference` | 0.70001 | +0.04597 |
| self_attention/progressive_8-16 | `iter1__non_residual__seed_t0.80__background_ratio` | 0.70052 | +0.04647 |
| self_attention/progressive_8-16-32 | `iter1__non_residual__seed_t0.80__background_ratio` | 0.72643 | +0.07238 |
| self_attention/progressive_8-16-32-64 | `iter4__residual_a0.75__seed_t0.30__background_difference` | 0.74738 | +0.09334 |
| self_attention/r16 | `all_modules_all_steps__iter1__seed_t0.30__foreground_residual_a0.50` | 0.71612 | +0.06208 |
| self_attention/r32 | `module__up_blocks.2.attentions.2.transformer_blocks.0.attn1__iter4__continuous__non_residual` | 0.76409 | +0.11005 |
| self_attention/r64 | `module__up_blocks.3.attentions.2.transformer_blocks.0.attn1__iter4__continuous__non_residual` | 0.77773 | +0.12369 |
| self_attention/r8 | `window__steps_26_30__iter1__continuous__residual_a0.75` | 0.70336 | +0.04932 |
| stage1/fusion | `resolution_weights__16-1_32-0.5_64-2_8-0.5` | 0.67703 | +0.02299 |
| stage1/heads | `per_layer_top_4_heads` | 0.68301 | +0.02897 |
| stage1/postprocessing | `fill_holes` | 0.66606 | +0.01202 |
| stage1/time_layer | `all16__cumulative_00_25` | 0.68441 | +0.03037 |
| threshold_normalization | `all_16_layers__percentile_01_99__t0.40` | 0.69752 | +0.04348 |
| time_resolution | `all_00_30__r8_r16` | 0.71219 | +0.05815 |
| time_window | `steps_15_20` | 0.67525 | +0.02121 |

### Exhaustive cross-attention layer subsets

- Best of 65,535 nonempty subsets: **0.68674 mIoU**, mask `0x23d6`, layers `[1, 2, 4, 6, 7, 8, 9, 13]`.
- All 16 layers: **0.65015 mIoU**.

### Per-class IoU: OVAM versus the best configuration

| Class | OVAM | Best SA | Δ | Images |
|---|---:|---:|---:|---:|
| aeroplane | 0.49721 | 0.86095 | +0.36374 | 24 |
| bicycle | 0.52805 | 0.74647 | +0.21842 | 25 |
| bird | 0.72035 | 0.81246 | +0.09211 | 25 |
| boat | 0.58945 | 0.78567 | +0.19622 | 25 |
| bottle | 0.68057 | 0.88907 | +0.20850 | 25 |
| bus | 0.86662 | 0.86321 | -0.00341 | 25 |
| car | 0.79254 | 0.80738 | +0.01484 | 23 |
| cat | 0.78902 | 0.83387 | +0.04485 | 25 |
| chair | 0.34130 | 0.72033 | +0.37903 | 24 |
| cow | 0.85443 | 0.89011 | +0.03568 | 25 |
| dining-table | 0.24405 | 0.39109 | +0.14704 | 25 |
| dog | 0.81841 | 0.84228 | +0.02387 | 25 |
| horse | 0.81496 | 0.87006 | +0.05510 | 23 |
| motorbike | 0.69996 | 0.83764 | +0.13768 | 25 |
| person | 0.53205 | 0.73493 | +0.20287 | 24 |
| potted-plant | 0.54146 | 0.56208 | +0.02062 | 25 |
| sheep | 0.86186 | 0.79826 | -0.06360 | 25 |
| sofa | 0.64722 | 0.92951 | +0.28229 | 25 |
| train | 0.52013 | 0.65424 | +0.13412 | 25 |
| tv | 0.74122 | 0.72502 | -0.01620 | 22 |

<!-- FULL_VOC_SIM_RESULTS_END -->

<!-- FULL_VOC_FOCUSED_SA_RESULTS_START -->
## Full VOC-Sim focused self-attention follow-up

Evaluated **490 images across 20 classes** for 1,632 configurations covering output calibration, cross-attention seeds, self-attention heads, and self-attention timestep weighting.

- Unoptimized OVAM baseline: **0.65404 mIoU**.
- Previous best: **0.77773 mIoU**.
- New best: **0.79222 mIoU**, `focused/output_calibration/top4_heads_percentile__r64__iter2__percentile_01_99__t0.45`.
- Gain over OVAM: **+0.13818**; gain over the previous best: **+0.01449**.

All 1,632 configurations contain 490 images and 20 classes. The previous winning path was reproduced within 0.00002 mIoU; the small difference comes from GPU floating point variation across regenerated samples.

### Focused family winners

| Experiment | Best configuration | mIoU | Δ vs. OVAM |
|---|---|---:|---:|
| focused/cross_attention_seeds | `top4_heads_percentile` | 0.68307 | +0.02902 |
| focused/output_calibration | `top4_heads_percentile__r64__iter2__percentile_01_99__t0.45` | 0.79222 | +0.13818 |
| focused/postprocessing | `best12_percentile__r64__iter4__percentile_01_99__t0.45__remove_fraction_0.01` | 0.77800 | +0.12396 |
| focused/sa_heads | `iter3__leave_head_6` | 0.77970 | +0.12566 |
| focused/sa_time | `iter3__trailing_10_30` | 0.78006 | +0.12602 |
| focused/seed_operator | `top4_heads_percentile__r64__iter2` | 0.79222 | +0.13818 |

### Per-class IoU comparison

| Class | OVAM | Previous best | New best | Δ new vs. previous |
|---|---:|---:|---:|---:|
| aeroplane | 0.49721 | 0.86095 | 0.83032 | -0.03063 |
| bicycle | 0.52805 | 0.74647 | 0.76556 | +0.01909 |
| bird | 0.72035 | 0.81246 | 0.86519 | +0.05274 |
| boat | 0.58945 | 0.78567 | 0.83884 | +0.05318 |
| bottle | 0.68057 | 0.88907 | 0.89059 | +0.00152 |
| bus | 0.86662 | 0.86321 | 0.88153 | +0.01832 |
| car | 0.79254 | 0.80738 | 0.79519 | -0.01219 |
| cat | 0.78902 | 0.83387 | 0.78380 | -0.05007 |
| chair | 0.34130 | 0.72033 | 0.69026 | -0.03006 |
| cow | 0.85443 | 0.89011 | 0.90688 | +0.01677 |
| dining-table | 0.24405 | 0.39109 | 0.40376 | +0.01267 |
| dog | 0.81841 | 0.84228 | 0.87601 | +0.03373 |
| horse | 0.81496 | 0.87006 | 0.88278 | +0.01272 |
| motorbike | 0.69996 | 0.83764 | 0.85595 | +0.01832 |
| person | 0.53205 | 0.73493 | 0.68768 | -0.04725 |
| potted-plant | 0.54146 | 0.56208 | 0.59027 | +0.02819 |
| sheep | 0.86186 | 0.79826 | 0.82667 | +0.02842 |
| sofa | 0.64722 | 0.92951 | 0.92494 | -0.00458 |
| train | 0.52013 | 0.65424 | 0.76349 | +0.10925 |
| tv | 0.74122 | 0.72502 | 0.78471 | +0.05969 |

<!-- FULL_VOC_FOCUSED_SA_RESULTS_END -->
