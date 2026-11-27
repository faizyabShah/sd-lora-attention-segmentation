# Run 49 execution notes

iSeg class token table at commit `57e9a17`. BLIP-large used the pinned safetensors snapshot `353689b859fcf0523410b1806dace5fb46ecdf41`. Downloads were staged locally and GPU captures ran offline.

## Round 0

GPU: NVIDIA A40, 46068 MiB, 520.61.05. The existing CUDA 11.8 PyTorch environment was used in GNU Screen.
The sample matched the registered counts of 30 photos, 63 items and 31 classes.
Every reference includes 63 scored items and all 104 map cells are complete.
All per-item gate miss values are zero on both backbones and in iSeg captures.
BLIP returned 63 captions. 0 raw captions differ from the requested literal prefix.
Original captions were retained.
- profiles: /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_ov42/fit.pt (model SD 1.5)
- First SD 15 photo 13675: 6 passes in 9.7 s (0.62 passes/s).
- profiles: /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_sd21_47/ov42/fit.pt (model sd2-community/stable-diffusion-2-1-base)
- First SD 21 photo 13675: 7 passes in 12.5 s (0.56 passes/s).

The first publication attempt stopped at a Git whitespace check for a trailing blank line in these notes. The error handler also used a relative log path after changing worktree. Both issues were corrected before retrying publication. Capture and scores were reused.
