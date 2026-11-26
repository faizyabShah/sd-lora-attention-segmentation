# JC #8: runtime and memory per image, BRACE and the fine-only reference (side N)

`comparisons/cost/measure_cost.py` (summary `summarize.py` → `result.json`; raw per-image records
`~/TRDP/external/n_side/cost/{a100,v100}.json`). Side F's frozen runtime, used unchanged: `scripts/sg27_runtime.py`
(SD 1.5 fp32, PNDM scheduler, the manifest's 30 steps, CFG 7.5, 512 × 512), runs/20 corrected prompts, safety checker
off. Fixed subset: `jc_newrerun.rows(ds, max_images=50)`, 50 VOC-Sim + 50 COCO-Cap images; one warm-up image is
discarded. CUDA-synchronised wall clock per stage; GPU peak = `torch.cuda.max_memory_allocated`. Batch size 1.

Checks on all 100 images, both GPUs: the image from the recording run is bit-identical to plain generation; the
stage-wise fine-only and hierarchy maps equal `sg_combined_v2.aggregate`'s V0 / V3 exactly (max relative
difference 0).

## Median seconds per image (10th–90th percentile)

| Stage | A100-PCIE-40GB (renyi) | V100-PCIE-16GB (erdos) |
|---|---:|---:|
| Generation alone, no recording (GPU) | 2.66 (2.59–2.68) | 12.11 (11.66–12.64) |
| Generation with attention recording and seed (GPU) | 6.94 (6.91–6.98) | 12.02 (11.26–12.75) |
| ↳ recording overhead (difference of medians) | +4.27 | −0.09 (none measurable) |
| Fine-only map, 2 SA passes × 11 calls (GPU) | 0.034 | 0.080 |
| BRACE map, 16→32→64 hierarchy with cap (GPU) | 0.038 | 0.098 |
| DenseCRF R2 on the 512 × 512 mask (CPU, one thread) | 5.92 | 5.07 |
| **Total, BRACE** | **12.90** (GPU 6.98) | **17.20** (GPU 12.14) |
| **Total, fine-only** | **12.89** (GPU 6.97) | **17.20** (GPU 12.13) |
| *Reference: committed `aggregate()`, all seven variants* | 0.119 | 0.402 |
| *Runtime diagnostics not used by BRACE (`sg27_fusion`)* | 0.049 | 0.092 |

## Peak GPU memory (max over the 100 images)

| Stage | A100 | V100 |
|---|---:|---:|
| Generation alone | 4.95 GiB | 4.76 GiB |
| Generation with recording | 5.14 GiB | 5.15 GiB |
| Fine-only / BRACE map | 5.11 / 5.14 GiB | 5.11 / 5.14 GiB |

The SA maps of BRACE are built from the Q/K already recorded; the hierarchy adds no measurable memory over fine-only.

## Hardware and software

| | A100 node | V100 node |
|---|---|---|
| GPU | NVIDIA A100-PCIE-40GB, driver 580.95.05 | Tesla V100-PCIE-16GB, driver 560.35.05 |
| CPU (DenseCRF) | AMD EPYC 7F72 | Intel Xeon Gold 5118 @ 2.30 GHz |
| Software (both) | Python 3.11.15, PyTorch 2.4.1+cu121 (CUDA 12.1, cuDNN 9.1.0), diffusers 0.36.0, transformers 5.6.2, SimpleCRF 0.2.1.1 (OVAM's DenseCRF wrapper) | |

## Cross-check with the full benchmark runs

- Official N-side BRACE capture (`comparisons/official/`, job 253750, all 1,197 images, V100 on wald): generation +
  recording + committed `aggregate()` took a median of 17.3 s (p10–p90 16.4–18.2 s), peak 5.3 GiB. The same stages
  on erdos here take 12.4 s. Same GPU model and environment; the outputs were bit-identical (runs/25 fidelity
  check, `comparisons/token_ladder_nr/`), so the gap is node throughput, not the method (the cause on wald was
  not established). Report the subset numbers above as the per-image cost; quote the wald figure only
  as an upper range.
- runs/25 (4 seeds and their maps per image) took about 10 s per image on erdos V100s.

## Ready-to-paste paragraph for Sec. 4 (implementation details)

> **Implementation and cost.** All experiments use Stable Diffusion 1.5 in float32 with its default PNDM scheduler,
> 30 denoising steps and classifier-free guidance 7.5 at 512 × 512 (PyTorch 2.4.1, CUDA 12.1, diffusers 0.36). We
> measured cost on 100 benchmark images (50 VOC-Sim, 50 COCO-Cap) at batch size 1. On an NVIDIA A100 (40 GB),
> generating an image takes 2.7 s; recording the attention BRACE needs raises this to 6.9 s, because the recording
> runs inside every attention call (on a V100 the two take the same 12.0 s). Building the maps from the recorded attention costs
> 0.04 s for BRACE and 0.03 s for the fine-only reference, and DenseCRF adds 5.9 s on one CPU core (AMD EPYC 7F72).
> One image therefore takes 12.9 s end to end on the A100 (17.2 s on a V100), of which 7.0 s is GPU time. Peak GPU
> memory is 5.1 GiB in both cases, and the hierarchy adds no measurable time or memory over fine-only
> propagation. The method trains nothing: there is no fitting cost beyond the threshold selection on VOC-Sim.

Note for the owners: the paragraph states what was measured; F's #8 item may prefer the frozen pipeline's own
timing hooks or a different GPU. The DenseCRF figure is per mask at one threshold; the benchmark scoring ran 17
thresholds per map, which is an evaluation cost, not a method cost.
