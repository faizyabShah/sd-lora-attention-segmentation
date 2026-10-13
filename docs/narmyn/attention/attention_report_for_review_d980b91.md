> **Snapshot.** Written against `main` at `d980b91` (2026-10-12), before the
> pushes of 2026-10-13. Two items it lists as missing are now available in the
> repository: the fixed top-4 CA head rankings
> (`outputs/segmentation/voc_sim_100_stage1_suite/results.json`) and the
> environment versions (`requirements.txt`). Line numbers refer to `d980b91`.

# VOC-Sim attention segmentation: review report

Repo state: `main` at `d980b91` (2026-10-12), in sync with `origin/main`.
All paths are relative to the repo root. All line numbers refer to `d980b91`.
"Faiz's outputs" means files under `/home/shafa/sd-lora-attention-segmentation/outputs/…`. Every job cds into that directory (for example `jobs/run_joint_sa_head_time.slurm`, `jobs/run_attention_analysis_implemented_ovam.slurm`). Those files are **not readable from this account** (permission denied), and none of them are tracked in git.

---

## 1. Commit history

The repo has **9 commits in total**, not 10. All of them are listed below.

| # | Hash | Date | Author | Message | Files changed |
|---|---|---|---|---|---|
| 1 | `d980b91` | 2026-10-12 16:26 +0200 | Shah Faizyab | `artifacts` | 12 files (+866 / −181) |
| 2 | `076119c` | 2026-10-12 00:30 +0200 | Shah Faizyab | `code` | 23 files (+1842 / −32) |
| 3 | `415981b` | 2026-10-11 17:07 +0200 | Shah Faizyab | `code push` | 51 files (+79177 / −18) |
| 4 | `4b5e4a2` | 2026-10-07 22:22 +0200 | Faizyab Ali Shah | `Merge pull request #2 …` | merge |
| 5 | `9c69581` | 2026-10-07 22:10 +0200 | NARMYN | `Add Cityscapes LoRA rank-sweep pipeline` | 23 files (LoRA only, out of scope here) |
| 6 | `63c0c9c` | 2026-10-06 20:22 +0200 | Shah Faizyab | `update: add code for narmeen` | 58 files (+2566 / −178) |
| 7 | `c9183c6` | 2026-09-22 10:49 +0200 | Shah Faizyab | `repo refactoring` | 31 files (LoRA / legacy) |
| 8 | `b047d29` | 2026-04-27 10:30 +0200 | Narmyn | `Added notebooks for Florence2 …` | 11 files (notebooks) |
| 9 | `df0886d` | 2026-04-22 17:40 +0200 | Faizyab Ali Shah | `Repository setup` | 50 files |

### Attention-segmentation content per commit

**`63c0c9c`: first attention code; 100-capture dev experiments**
- Added:
  - `scripts/capture_ovam_attention.py`
  - `jobs/capture_ovam_voc_sim_100.slurm` (`--count 100 --start-index 0 --storage-dtype float16`)
  - `src/attention_capture/{recorder,io}.py`
  - `src/segmentation/{attention_aggregation,captured_dataset,ablation_runner,metrics}.py`
  - `scripts/run_attention_ablations.py`
  - `scripts/reproduce_ovam_voc_sim.py`
- Added `results.md`. Its first tables cover timestep windows, resolutions and layers on the 99 annotated captures among VOC-Sim indices 0–99.

**`415981b`: full-VOC-Sim streaming ablations, Stage 1/2 suites, focused SA**
- Added:
  - `src/attention_capture/{live,memory}.py`: in-memory capture
  - `src/segmentation/{stage1_suite,focused_self_attention,self_attention_refinement,exhaustive_layer_search,ovam_baseline,postprocessing,in_memory_dataset}.py`
  - all `jobs/run_*` / `jobs/merge_*` for the streaming, focused and joint suites
  - `ablations.md` (73,638 lines)
  - `docs/streaming_voc_ablations.md`
- `results.md` grew by 378 lines. It gained the sections "All dataset: full VOC-Sim" (`results.md:353`) and "Full VOC-Sim focused self-attention follow-up" (`results.md:430`).

**`076119c`: attention analysis, plus scoring of the released OVAM artifacts**
- Added:
  - `src/segmentation/attention_analysis.py`
  - `scripts/run_attention_analysis.py`
  - `scripts/merge_attention_analysis.py`
  - `scripts/evaluate_released_ovam_{masks,heatmaps}.py`
  - `src/segmentation/ovam_reference.py`
  - `attention_analysis.md` (first version)
  - `ovam.pdf`
- Modified `focused_self_attention.py` (+42 lines) and `ovam_baseline.py` (+23 lines).

**`d980b91`: implemented full OVAM (CA×SA + DenseCRF) as the baseline**
- Added:
  - `src/segmentation/ovam_full.py` (146 lines)
  - `jobs/{run,merge,smoke}_attention_analysis_implemented_ovam.slurm`
- `memory.py` (+26) now computes the OVAM all-ones SA hook (`memory.py:73-99`).
- `attention_analysis.md` was rewritten (+408 / −181). The final 0.7997 is now compared with the implemented OVAM (0.6934).

### Result tables (verbatim from the repo)

#### `results.md:366-394`: full VOC-Sim, best configuration per family (490 images, 6,261 configurations)

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

Exhaustive CA layer subsets (`results.md:397-400`): the best of 65,535 subsets scores 0.68674 (mask `0x23d6`, layers `[1, 2, 4, 6, 7, 8, 9, 13]`). All 16 layers score 0.65015.

#### `results.md:430-451`: focused SA follow-up (490 images, 1,632 configurations)

| Experiment | Best configuration | mIoU | Δ vs. OVAM |
|---|---|---:|---:|
| focused/cross_attention_seeds | `top4_heads_percentile` | 0.68307 | +0.02902 |
| focused/output_calibration | `top4_heads_percentile__r64__iter2__percentile_01_99__t0.45` | 0.79222 | +0.13818 |
| focused/postprocessing | `best12_percentile__r64__iter4__percentile_01_99__t0.45__remove_fraction_0.01` | 0.77800 | +0.12396 |
| focused/sa_heads | `iter3__leave_head_6` | 0.77970 | +0.12566 |
| focused/sa_time | `iter3__trailing_10_30` | 0.78006 | +0.12602 |
| focused/seed_operator | `top4_heads_percentile__r64__iter2` | 0.79222 | +0.13818 |

#### `attention_analysis.md:40-53`: performance progression

| Method | Configuration | mIoU | Absolute gain over OVAM |
|---|---|---:|---:|
| Authors' OVAM CA precursor | released heatmaps | 0.6616 | -0.0318 |
| Our regenerated CA precursor | `ovam_cross_attention_only` | 0.6540 | -0.0393 |
| Implemented OVAM, unoptimized | CA + SA fusion + DenseCRF | 0.6934 | 0.0000 |
| Authors' released OVAM artifacts | Released CA + SA + mask | 0.7036 | +0.0103 |
| Earlier SA winner | best-12 CA, final 64×64 SA, 4 iterations | 0.7777 | 0.0844 |
| Focused winner | `top4_heads_percentile__r64__iter2__percentile_01_99__t0.45` | 0.7922 | 0.0989 |
| Joint winner, no morphology | `iter3__leave_3_7__steps_15_20__t0.40` | 0.7991 | 0.1058 |
| Final winner | `iter4__leave_3_7__trailing_15_30__t0.45__remove_fraction_0.01` | 0.7997 | 0.1064 |

The joint suite (`run_joint_sa_head_time_suite`, `focused_self_attention.py:556-613`) has no results table in `results.md`. Its numbers appear only in `attention_analysis.md`. Its merged output is Faiz's `outputs/segmentation/voc_sim_full_joint_sa_head_time` (default in `scripts/run_joint_sa_head_time.py`).

#### `attention_analysis.md:71-83`: matched (one-factor) comparisons

| Controlled change | Before | After | mIoU change |
|---|---:|---:|---:|
| All 16 CA heads/layers → selected top-4-head seed, before SA | 0.6830 | 0.6831 | +0.0001 |
| Top-4 CA seed → two iterations through final 64×64 SA | 0.6831 | 0.7922 | +0.1092 |
| One → four SA propagation iterations | 0.7799 | 0.7991 | +0.0192 |
| All timesteps → steps 15–30 | 0.7937 | 0.7991 | +0.0054 |
| All SA heads → omit heads 3 and 7 | 0.7929 | 0.7991 | +0.0062 |
| Min-max → percentile 1–99 normalization | 0.7904 | 0.7922 | +0.0018 |
| No morphology → remove components below 1% | 0.7991 | 0.7997 | +0.0006 |

#### `attention_analysis.md:97-120`: pooled class IoU, implemented OVAM vs. final

| Class | Implemented OVAM | Final | Difference |
|---|---:|---:|---:|
| aeroplane | 0.709 | 0.838 | +0.129 |
| bicycle | 0.597 | 0.760 | +0.162 |
| bird | 0.776 | 0.859 | +0.083 |
| boat | 0.708 | 0.858 | +0.151 |
| bottle | 0.788 | 0.882 | +0.094 |
| bus | 0.906 | 0.877 | -0.029 |
| car | 0.835 | 0.785 | -0.050 |
| cat | 0.722 | 0.811 | +0.090 |
| chair | 0.403 | 0.723 | +0.321 |
| cow | 0.878 | 0.915 | +0.037 |
| dining-table | 0.250 | 0.407 | +0.157 |
| dog | 0.753 | 0.887 | +0.134 |
| horse | 0.805 | 0.887 | +0.082 |
| motorbike | 0.774 | 0.857 | +0.084 |
| person | 0.564 | 0.685 | +0.121 |
| potted-plant | 0.575 | 0.603 | +0.027 |
| sheep | 0.792 | 0.873 | +0.081 |
| sofa | 0.734 | 0.928 | +0.194 |
| train | 0.596 | 0.777 | +0.181 |
| tv | 0.702 | 0.782 | +0.080 |

Per-class tables comparing the CA-only baseline with the 0.77773 and 0.79222 winners are at `results.md:402-427` and `results.md:452-475`. A precision/recall/area table by class is at `attention_analysis.md:122-149`. These are not repeated here.

---

## 2. The final method (0.7997)

The authoritative definition is `run_joint_sa_head_time_suite`, `src/segmentation/focused_self_attention.py:556-613`. The config key that produced 0.7997 is `joint/joint_postprocessing/iter4__leave_3_7__trailing_15_30__t0.45__remove_fraction_0.01`. The analysis run reproduces it at `src/segmentation/attention_analysis.py:261-267`:

```python
joint4 = propagated.per_step_head[4].reshape(len(sample.timesteps), 8, 64, 64)
kept_heads = (0, 1, 2, 4, 5, 6)
winner_map = joint4[15:, list(kept_heads)].mean(dim=(0, 1))
winner_pred = attention_to_mask(winner_map, target.shape, 0.45, "percentile_01_99")
winner_pred = torch.from_numpy(
    postprocess_mask(winner_pred.numpy(), "remove_fraction_0.01")
)
```

### 2.1 Generation (live, nothing saved)

- `src/attention_capture/live.py:109-120` calls `pipe(prompt_embeds=…, negative_prompt_embeds=…, latents=initial_noise, height=512, width=512, num_inference_steps=steps, guidance_scale=7.5, …)`.
- `steps = int(row["num_inferece_steps"])`, which is 30 for every row (`live.py:58`). The column name is misspelled in the manifest.
- Seed: `_seed_everything(seed)` (`live.py:83`). Initial noise is `torch.randn((1, 4, 64, 64))` (`live.py:92-98`).
- Model: `stable-diffusion-v1-5/stable-diffusion-v1-5`, loaded with `--model-dtype float32`. Captured Q/K are stored with `--storage-dtype float16`.

### 2.2 Step numbering

- The scheduler is `PNDMScheduler` with `skip_prk_steps: True` and `timestep_spacing: "leading"`. `num_inference_steps=30` produces **31 UNet calls**.
- `sample.timesteps` has length 31. Step index `i` ∈ **0–30** is the UNet call counter (`memory.py:68`, `live.py:101-106`).
- Index → scheduler timestep (from `outputs/attention/voc_sim/ovam-voc-sim-0/metadata.json`):
  - 0–14: `958, 925, 925, 892, 859, 826, 793, 760, 727, 694, 661, 628, 595, 562, 529`. Timestep 925 is used twice, at indices 1 and 2 (PLMS warm-up).
  - 15–30: `496, 463, 430, 397, 364, 331, 298, 265, 232, 199, 166, 133, 100, 67, 34, 1`.
- The name `trailing_15_30` is `range(15, count)` with `count = 31` (`focused_self_attention.py:524`). That is **indices 15 through 30 inclusive, 16 steps**, i.e. timesteps 496 → 1.
- Weights are uniform, `1/16` each (`_selection_weights`, `focused_self_attention.py:506-512`).
- For comparison, `steps_15_20` (the 0.7991 variant) is `range(15, 21)`: indices 15–20 inclusive, 6 steps.

### 2.3 Cross-attention seed (`top4_heads_percentile`)

**Layers.** These are the `BEST_12_INDICES = (0, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)` (`stage1_suite.py:24`), indexed into the **sorted** list of 16 `attn2` modules (`focused_self_attention.py:80`):

| Index | Module | Res | In seed? |
|---:|---|---:|:---:|
| 0 | `down_blocks.0.attentions.0.transformer_blocks.0.attn2` | 64 | yes |
| 1 | `down_blocks.0.attentions.1.transformer_blocks.0.attn2` | 64 | no |
| 2 | `down_blocks.1.attentions.0.transformer_blocks.0.attn2` | 32 | yes |
| 3 | `down_blocks.1.attentions.1.transformer_blocks.0.attn2` | 32 | no |
| 4 | `down_blocks.2.attentions.0.transformer_blocks.0.attn2` | 16 | no |
| 5 | `down_blocks.2.attentions.1.transformer_blocks.0.attn2` | 16 | yes |
| 6 | `mid_block.attentions.0.transformer_blocks.0.attn2` | 8 | yes |
| 7 | `up_blocks.1.attentions.0.transformer_blocks.0.attn2` | 16 | yes |
| 8 | `up_blocks.1.attentions.1.transformer_blocks.0.attn2` | 16 | no |
| 9 | `up_blocks.1.attentions.2.transformer_blocks.0.attn2` | 16 | yes |
| 10 | `up_blocks.2.attentions.0.transformer_blocks.0.attn2` | 32 | yes |
| 11 | `up_blocks.2.attentions.1.transformer_blocks.0.attn2` | 32 | yes |
| 12 | `up_blocks.2.attentions.2.transformer_blocks.0.attn2` | 32 | yes |
| 13 | `up_blocks.3.attentions.0.transformer_blocks.0.attn2` | 64 | yes |
| 14 | `up_blocks.3.attentions.1.transformer_blocks.0.attn2` | 64 | yes |
| 15 | `up_blocks.3.attentions.2.transformer_blocks.0.attn2` | 64 | yes |

`results.md:232` confirms that indices 1, 3, 4 and 8 are excluded ("Best overall | 0,2,5,6,7,9,10,11,12,13,14,15 | 1,3,4,8").

**The 4 head indices per layer: not available from this account.**
- They are read at run time from `json.load(open("outputs/segmentation/voc_sim_100_stage1_suite/results.json"))["settings"]["head_rankings"]` (`run_joint_sa_head_time.py:66-67`; `run_attention_analysis.py:135-137`).
- They are used as `ranking[:4]` (`focused_self_attention.py:126-131`).
- Only Faiz's copy of that file exists. To extract them, run this in `/home/shafa/sd-lora-attention-segmentation`:
  ```bash
  python -c "import json;r=json.load(open('outputs/segmentation/voc_sim_100_stage1_suite/results.json'))['settings']['head_rankings'];[print(k, v[:4]) for k,v in sorted(r.items(), key=lambda x:int(x[0]))]"
  ```

**How the per-head CA map is computed (live path):**
- Function: `memory.py:114-123`, key `{module}.ovam_target_map`.
  ```python
  ovam_scores = torch.matmul(
      query[0].float(),
      key[conditional, :, : self.prompt_token_count].float().transpose(-1, -2),
  ) * scale
  bucket[f"{module_name}.ovam_target_map"] = self._cpu(
      ovam_scores.softmax(dim=-1)[:, :, self.ovam_target_position]
  )
  ```
- **Query:** the CFG batch is ordered `[unconditional, conditional]` (`metadata.json: cfg_batch_order`), so `query[0]` is the **unconditional** query.
- **Key:** the **conditional** key (index 1), truncated to the unpadded prompt length (`prompt_token_count` = length of the prompt tokenized with `padding=False`, `live.py:73-74`).
- **Softmax:** taken over those unpadded tokens only.
- **Token:** the single position `len(prompt.split())` (`live.py:79`). All prompts are `"A photograph of a <class>"`, so this is position 5 for every class. For classes that tokenize into several pieces, it is only the first piece: aeroplane → `aerop`, dining-table → `dining`, potted-plant → `potted`.
- **The `branch=1` argument has no effect.** `build_cross_attention_layer_heads` passes `branch=1` (`focused_self_attention.py:94`). But `load_step_module_head_maps` finds the compact key `{module}.{sample.cross_map_key}` (`attention_aggregation.py:111-115`) and never reads `branch`. In the live path `cross_map_key = "ovam_target_map"` (`live.py:142`, default `cross_map_kind="ovam"`). The query is therefore unconditional despite `branch=1`. The run signature records this as `"cross_map_kind": "ovam_unconditional_query_prompt_softmax"` (`run_joint_sa_head_time.py:76`).

**Aggregation** (`focused_self_attention.py:93-103`, `126-149`):
1. For every step `i ∈ 0–30` (**all 31 steps; the seed has no step window**) and every module, the per-head map is reshaped to `[8, r, r]` and resized to 64×64 with bilinear interpolation, `align_corners=False` (`resize_map`, `attention_aggregation.py:125-130`).
2. The resized maps are averaged over the 31 steps, giving `layer_heads[16, 8, 64, 64]`.
3. For each of the 12 layers, the 4 ranked heads are averaged. The 12 layer maps are then averaged with equal weight (`torch.stack(top4_maps).mean(dim=0)`, line 138).
4. The result is normalized with `percentile_01_99` (line 146; `attention_aggregation.py:163-171`): `torch.quantile` at [0.01, 0.99], then `(x − q01)/(q99 − q01)`, then `clamp_(0, 1)`.
5. The output is the seed `s₀ ∈ [0,1]^{4096}`.

### 2.4 Self-attention propagation

**Module.** `up_blocks.3.attentions.2.transformer_blocks.0.attn1` (`focused_self_attention.py:568`; `attention_analysis.py:30`). It operates at 64×64, i.e. 4096 tokens, with 8 heads of head_dim 40.

**Q/K source.** Conditional branch only. `memory.py:101-102` stores `query[conditional:conditional+1]` and `key[conditional:conditional+1]` as float16. `focused_self_attention.py:198` then selects branch 0 of that singleton, which is the conditional branch. The tensors are cast to float32 for the matmul (lines 229-230).

**Propagation**, per step `t` and head `h` (`focused_self_attention.py:227-240`):
```python
scale = 1.0 / math.sqrt(query.shape[-1])          # 1/sqrt(40)
affinity = torch.softmax(torch.matmul(q, k.transpose(0, 1)) * scale, dim=-1)   # [4096, 4096], row-stochastic
current = seeds_device
for iteration in ITERATIONS:                      # (1, 2, 3, 4)
    current = torch.matmul(affinity, current)
    ...
    per_step_head[iteration][step_index, head] = tracked
```
- `per_step_head[4][t, h]` is exactly `A_{t,h}⁴ · s₀`. It is the 4th power of that step's and head's own affinity matrix, **not** `(mean_t,h A)⁴` and not a cumulative sum over iterations.
- There is no residual term and no re-normalization between iterations.

**Final pre-threshold map.** T = {15, …, 30} (16 steps) and H = {0, 1, 2, 4, 5, 6}, from `joint_head_selections()["leave_3_7"]` (`focused_self_attention.py:548-551`):

```
M = (1 / (16 · 6)) · Σ_{t∈T} Σ_{h∈H}  A_{t,h}^4 · s₀          (reshape to 64×64)
```

The joint suite computes M as `values[:, heads].mean(dim=1)` followed by `einsum("t,tij->ij", weights, ·)` (`focused_self_attention.py:591-593`). The analysis code uses `mean(dim=(0,1))`. Both are identical uniform means.

### 2.5 Normalization, threshold, resize, cleanup

Joint-suite code, `focused_self_attention.py:594-609`:
```python
normalized = normalize_map(score_map, "percentile_01_99")
resized = F.interpolate(normalized[None, None], size=target.shape, mode="bilinear", align_corners=False)[0, 0]
...
prediction = (resized >= 0.45).cpu().numpy()
processed = torch.from_numpy(postprocess_mask(prediction, "remove_fraction_0.01"))
```

| Step | Exact setting |
|---|---|
| Normalization | Percentile 1–99 **on the 64×64 map, before resizing**: `torch.quantile(x, [0.01, 0.99])` (linear interpolation), `(x−q01)/(q99−q01)`, `clamp_(0,1)`. Returns zeros if `q99 ≤ q01`. (`attention_aggregation.py:163-171`) |
| Resize | `F.interpolate(…, size=target.shape, mode="bilinear", align_corners=False)`. `target.shape` is **512×512** (all annotations are 512×512 RGB PNGs). |
| Threshold | `resized >= 0.45` (non-strict) |
| Component removal | `remove_fraction_0.01` (`postprocessing.py:20-28`): `ndimage.label(mask)` with the default structure, which is **4-connectivity**; keep components with `size >= ceil(512·512·0.01) = 2622` pixels; `keep[0] = False` (background). If every component is smaller, the mask is empty. |
| DenseCRF | **Not applied** to the final method. |

Target mask: `np.any(rgb != 0, axis=-1)` on the annotation PNG (`live.py:131-132`).

---

## 3. Baselines

### 3.1 `ovam_cross_attention_only` = 0.6540 (our regenerated CA precursor)

- Scored in `scripts/run_streaming_voc_ablations.py:181-185`:
  ```python
  metric.update(sample.classname, ovam_cross_attention_mask(sample, target.shape), target)
  merge_results(local, "baseline", {"ovam_cross_attention_only": metric.result()})
  ```
- In `results.md:366` the same baseline appears under the older name `ovam_unoptimized` (0.65404). The merge scripts accept either name (`merge_streaming_voc_ablations.py:168`, `merge_focused_sa_ablations.py:54`).

`src/segmentation/ovam_baseline.py`:

| Item | Exact setting | Line |
|---|---|---|
| Modules | 15 `attn2` modules: every cross-attention module **except** `mid_block` (matches `StableDiffusionHooker` `locate_middle_block=False`) | 21-31 |
| Per-head map | `ovam_target_map`: unconditional query × conditional K truncated to `prompt_token_count`; softmax over the unpadded prompt; single position `len(prompt.split())` | 66-68, fallback 70-76 |
| Prompt length | index of the first EOT (49407) + 1 | 34-40 |
| Heads | **summed** (`per_head.sum(dim=0)`) | 77 |
| Steps | all `range(len(sample.timesteps))` = **31 calls (0–30)**. `docs/streaming_voc_ablations.md:16` says "sum all 30 denoising steps"; the code sums 31. | 57 |
| Modules → 64 | each module map resized to 64×64 (bilinear, `align_corners=False`), then **summed** | 78 |
| To mask | resize 64 → 512 bilinear `align_corners=False`, divide by the max, then **strict** `> 0.4` | 82-94 |

There is no SA, no CRF and no token optimization.

### 3.2 Implemented full OVAM = 0.6934 (the baseline used in `attention_analysis.md`)

`src/segmentation/ovam_full.py`, called from `scripts/run_attention_analysis.py:177-179`:
- CA: as in 3.1 (`ovam_cross_attention_map`).
- SA (`ovam_self_attention_map`, lines 36-64):
  - Uses the per-forward `ovam_self_map` from `memory.py:79-98`: `scaled_dot_product_attention(query, key, torch.ones_like(key))`, summed over **both CFG branches** (`.sum(dim=0)`) and over the feature channels (`.sum(dim=-1)`), stored float32.
  - Aggregation: mean over 31 steps, then sum over heads, per highest-resolution (64×64) `attn1` module. Each is resized to 64, then **negated**: `-torch.stack(module_maps).mean(dim=0)`.
- Fusion (lines 77-98):
  - CA and SA are each bilinearly resized to the image size.
  - `ca / ca.max()`; SA is min-max scaled to `[0.85, 1]` (`OVAM_SA_MIN = 0.85`).
  - Then `ca * sa`.
- Threshold: `fused > 0.40` (line 140).
- DenseCRF:
  - SimpleCRF `denseCRF.densecrf(image, probs, (10.0, 80, 13, 3.0, 3, 5.0))` (lines 24, 114-126).
  - Runs on **our regenerated** RGB image (`sample.generated_image`), with `probs = stack(~mask, mask)`.
  - This is the only place a CRF is used in the repo.
- Known issue: the all-ones value tensor makes every attention row equal 1 in exact arithmetic. The SA "map" is therefore float32 round-off amplified by min-max scaling. `attention_analysis.md:87-91` reports a correlation of 0.237 with the released SA, and 18.9 vs 5.5 distinct values per image. This matches the authors' hook (`ovam/stable_diffusion_sa/self_att_block_hooker.py:85`, `dummy_value = torch.ones_like(key)`).

### 3.3 Released masks = 0.7036

`scripts/evaluate_released_ovam_masks.py`:
- Rows: manifest rows with `annotated.lower() == "true"`, giving 490 rows (line 71).
- Mask file: `evaluation/ovam_sa/voc_sim/mask/<annotation stem without _annotation>_ovamsa_crf_mask.png` (lines 39-41, 61-65).
- Binarization of both the mask and the annotation: `np.any(rgb != 0, axis=-1)` (lines 56-58). **There is no resizing**; a shape mismatch raises an error (lines 83-84).
- Scored with `BinarySegmentationMetrics` (line 85). The script also writes per-image IoU (lines 86-94).
- Output: `outputs/segmentation/ovam_paper_baseline/summary.json`. The metadata labels it `threshold 0.4, self_attention_min 0.85, dense_crf True`; these are labels only, not settings applied by the script.
- The released `ovam_sa/` folder is not present in this clone's `datasets/ovam_experiment_with_dataset/evaluation/` (only `ovam/voc_sim/{annotations,images,ovam_voc_sim.csv}` are present).

### 3.4 Released heatmaps = 0.6616

`scripts/evaluate_released_ovam_heatmaps.py:39-46`:
```python
heatmap_path = heatmap_dir / f"{Path(row['image']).stem}_heatmap_token0.npy"
heatmap = torch.from_numpy(np.load(heatmap_path)[1]).float()[None, None]
resized = torch.nn.functional.interpolate(heatmap, size=target.shape, mode="bilinear")[0, 0]
prediction = resized / resized.max() > args.threshold          # default 0.4
```
- It uses channel `[1]` of the `token0` array. `align_corners` is unset, which PyTorch treats as False.
- Same 490 rows. No per-image output is written.

### 3.5 Versions, scheduler, guidance, CFG

| Item | Value | Source |
|---|---|---|
| Scheduler | `PNDMScheduler`: `skip_prk_steps=True`, `steps_offset=1`, `timestep_spacing="leading"`, `beta_schedule="scaled_linear"` (0.00085–0.012), `set_alpha_to_one=False`, `prediction_type="epsilon"` | `metadata.json` of `outputs/attention/voc_sim/ovam-voc-sim-0` |
| Guidance | 7.5 | `live.py:116` |
| CFG order | `[unconditional, conditional]` | `metadata.json: cfg_batch_order` |
| Negative prompt | `None`, i.e. the empty-string embedding | `live.py:85-91` |
| Versions (local disk captures, this account) | torch `2.4.1+cu121`, diffusers `0.36.0`, transformers `5.6.2`, CUDA 12.1, Tesla V100S-PCIE-32GB | `metadata.json` |
| Versions (Faiz's live runs that produced every number above) | **Not recorded.** The checkpoint signatures (`run_joint_sa_head_time.py:68-78`, `run_attention_analysis.py:138-147`) store the model, dtypes and manifest, but not library versions. The jobs run `conda activate trdp`. | |
| `requirements.txt` pins | `torch==2.1.2`, `transformers==4.44.2`, `diffusers==0.36.0`. These do not match the recorded local env. | `requirements.txt:1-5` |

Which CFG branch each component uses:
- CA (baseline and final seed): unconditional **query**, conditional **key**.
- Final-method SA: conditional branch Q/K only.
- OVAM SA hook: sums both branches.

---

## 4. Evaluation

### 4.1 Metric code, verbatim: `src/segmentation/metrics.py:12-79`

```python
@dataclass
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    image_iou_sum: float = 0.0
    images: int = 0


class BinarySegmentationMetrics:
    """Accumulate foreground confusion counts independently for each class."""

    def __init__(self) -> None:
        self.counts: dict[str, _Counts] = defaultdict(_Counts)

    def update(self, classname: str, prediction: torch.Tensor, target: np.ndarray) -> None:
        pred = prediction.detach().cpu().numpy().astype(bool, copy=False)
        truth = target.astype(bool, copy=False)
        if pred.shape != truth.shape:
            raise ValueError(f"Prediction {pred.shape} and target {truth.shape} differ")
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        tn = int(np.count_nonzero(~pred & ~truth))
        item = self.counts[classname]
        item.tp += tp
        item.fp += fp
        item.fn += fn
        item.tn += tn
        union = tp + fp + fn
        item.image_iou_sum += tp / union if union else 1.0
        item.images += 1

    def result(self) -> dict:
        per_class: dict[str, dict] = {}
        foreground_ious: list[float] = []
        background_ious: list[float] = []
        image_iou_sum = 0.0
        image_count = 0
        for classname in sorted(self.counts):
            value = self.counts[classname]
            foreground_union = value.tp + value.fp + value.fn
            background_union = value.tn + value.fp + value.fn
            foreground_iou = value.tp / foreground_union if foreground_union else 1.0
            background_iou = value.tn / background_union if background_union else 1.0
            per_class[classname] = {
                "iou": foreground_iou,
                "background_iou": background_iou,
                "mean_image_iou": value.image_iou_sum / value.images,
                "images": value.images,
                "tp": value.tp,
                "fp": value.fp,
                "fn": value.fn,
                "tn": value.tn,
            }
            foreground_ious.append(foreground_iou)
            background_ious.append(background_iou)
            image_iou_sum += value.image_iou_sum
            image_count += value.images
        return {
            "miou": float(np.mean(foreground_ious)),
            "mean_background_iou": float(np.mean(background_ious)),
            "mean_image_iou": image_iou_sum / image_count,
            "images": image_count,
            "classes": len(per_class),
            "per_class": per_class,
        }
```

In formulas:
- **Pooled class IoU (the reported "mIoU"):** for each class c, `IoU_c = Σ_{i∈c} TP_i / Σ_{i∈c} (TP_i + FP_i + FN_i)`. Then `mIoU = (1/20) Σ_c IoU_c`. Only foreground IoU is used; background IoU is reported separately.
- **Mean image IoU:** `(1/490) Σ_i TP_i / (TP_i + FP_i + FN_i)`, with an empty union counted as 1.0. It is a mean over images, not over classes.
- All counts are taken at 512×512.

### 4.2 Excluded indices

The manifest `datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv` has 500 rows: 25 per class across 20 classes. **10 rows** have `annotated == "False"` and an empty `annotation` field, so there is no ground truth for them:

| Row / `image_id` | Class |
|---|---|
| `ovam-voc-sim-4` | aeroplane |
| `ovam-voc-sim-151` | car |
| `ovam-voc-sim-171` | car |
| `ovam-voc-sim-217` | chair |
| `ovam-voc-sim-308` | horse |
| `ovam-voc-sim-321` | horse |
| `ovam-voc-sim-372` | person |
| `ovam-voc-sim-484` | tv |
| `ovam-voc-sim-489` | tv |
| `ovam-voc-sim-498` | tv |

Where they are excluded:
- Live suites: `read_rows`, `run_streaming_voc_ablations.py:85`. The focused, joint and analysis runners reuse it.
- Released-artifact scorers: `evaluate_released_ovam_masks.py:71`, `evaluate_released_ovam_heatmaps.py:37`.
- Disk captures: `captured_dataset.py:94` (`require_annotation`).

This leaves **490 images**. Per-class counts: 25 each, except aeroplane 24, chair 24, person 24, car 23, horse 23, tv 22. No other filtering is applied.

### 4.3 Which images each selection step used

The "dev set" is manifest rows 0–99 (`jobs/capture_ovam_voc_sim_100.slurm`: `--count 100 --start-index 0`). That is 25 each of aeroplane, bicycle, bird and boat, of which **99 are annotated** (row 4 is missing). **These 99 images are a subset of the 490 test images.**

| # | Choice in the final method | Selected on | Selection data / source |
|---|---|---|---|
| 1 | CA layers `BEST_12_INDICES` | 99 dev images (4 classes) | Exhaustive search over 65,535 subsets on disk captures; conditional-branch probabilities; min-max, t = 0.40 → 0.74090 (`results.md:232`) |
| 2 | CA normalization `percentile_01_99` (+ t = 0.45 for CA-only) | 99 dev images | `results.md:263-290`; 0.74394 |
| 3 | Top-4 CA heads per layer | 99 dev images | Rank heads by pooled mIoU of each single head (`stage1_suite.py:283-290`; per-head prediction `percentile_01_99`, `>= 0.45`, `stage1_suite.py:67-72`) |
| 4 | SA module `up_blocks.3.attentions.2…attn1` | **all 490** | Streaming family `self_attention/r64` (0.77773) |
| 5 | Seed = `top4_heads_percentile` | **all 490** | Focused suite (0.79222) |
| 6 | Iterations = 4, heads `leave_3_7`, steps `trailing_15_30`, threshold 0.45 | **all 490** | Joint suite (`focused_self_attention.py:556-613`) |
| 7 | `remove_fraction_0.01` | **all 490** | Joint suite `joint_postprocessing` group (only tested at t = 0.45) |

Two caveats about steps 1–3.

**The head rankings were selected on a different cross-attention map than the one they are applied to.**
- The rankings came from disk captures. For those, `CapturedSample.cross_map_key` returns `"target_maps"` (`captured_dataset.py:50-52`, comment: "Historical disk captures contain full conditional probabilities"). The maps are read from `{module}.probabilities` at `branch=1` (conditional query), averaged over **all** class-token positions (`attention_aggregation.py:117-119`).
- The final method applies those rankings to `ovam_target_map`: unconditional query, unpadded softmax, first class token only.

**The suite's metadata note contradicts how the heads were chosen.** It says "head ranking is per-image and label-free" (`focused_self_attention.py:498-501`). In fact the head ranking is a **fixed, dataset-level, ground-truth-based** ranking taken from the 99 dev images.

### 4.4 Are per-image intersection/union counts saved for every configuration?

**No.**

| Run | What is saved | Where |
|---|---|---|
| Streaming ablations (6,261 configs), focused (1,632), joint suite | Only **per-class accumulated** `tp, fp, fn, tn, images` (plus `image_iou_sum`); no per-image counts | Faiz's `outputs/segmentation/voc_sim_full_streaming/…`, `…/voc_sim_full_joint_sa_head_time/class_XX_<name>/checkpoint.pt` and merged JSON (`run_streaming_voc_ablations.py:118-124`, `263-277`) |
| Attention analysis (`run_attention_analysis.py`) | **Per-image** `tp, fp, fn, tn, iou` plus map diagnostics for 9 selected methods (`attention_analysis.py:277-293`). Also, per image, the 16 CA layers, all SA modules, the 8 SA heads and the 31 steps (each via `map_statistics`, which returns tp/fp/fn/tn, `attention_analysis.py:164-168`). | Faiz's `outputs/analysis/voc_sim_attention_report_implemented_ovam_highres/analysis_records.json.gz` (`attention_analysis.md:338`) |
| Released masks | Per-image IoU only (no counts) | `outputs/segmentation/ovam_paper_baseline/summary.json` |
| Released heatmaps | Nothing per image | `…/cross_attention_only_summary.json` |

The 9 methods in the analysis run are: implemented OVAM, published OVAM, OVAM CA-only, top-4 seed, best-12 seed, 0.7777, 0.7922, 0.7991 and 0.7997.

### 4.5 Held-out or cross-validation evaluation

**None exists.** No code splits the 490 images, runs k-fold, or selects on one subset and reports on another. The only related items are:
- the caveats at `results.md:258` and `results.md:285` and `attention_analysis.md:330`, which state that selection and reporting use the same images;
- the paired image bootstrap in `merge_attention_analysis.py:372-386`: 20,000 resamples, seed `20261012`, result `+0.0999 [+0.0845, +0.1151]`. It measures image-sampling noise for the **already-selected** method and does not correct for selection.

---

## 5. Other

| Question | Answer |
|---|---|
| DenseCRF on the final method? | **No.** It is not in the final pipeline and no variant combines it with the final method. CRF is used only in the OVAM baseline (`ovam_full.py:114-146`). `stage1_suite.py:347` records `"dense_crf": "not run: pydensecrf is not installed"`. |
| SeeDiff run? | **No.** The string `seediff` appears nowhere in `src/`, `scripts/`, `jobs/` or the docs. |
| OVAM token optimization run? | **No.** All OVAM numbers are "unoptimized". `evaluate_released_ovam_masks.py --optimized` only relabels a run (lines 49-51, 102-107); nothing in the repo optimizes a token. |
| Attention data on disk | This account, `outputs/attention/voc_sim/`: **50 complete captures (`ovam-voc-sim-0` … `-49`), 364 GB in total**, about 7.3 GB each. Per sample: 31 × `step_NNN.safetensors` (about 251 MB each) and `cross_attention_kv.safetensors` (7.7 MB); `metadata.json` reports `attention_bytes: 7788598968`. Not tracked in git. These are old-format captures and **are not used by any full-VOC-Sim result**; those were computed live (`docs/streaming_voc_ablations.md:5-8`). Faiz's 100-capture dev set (`outputs/attention/voc_sim`, rows 0–99) is on his account; its size was not measured. |

### Diagnostic formulas (`src/segmentation/attention_analysis.py:105-187`)

Setup:
- Each map is resized to 64×64 (bilinear) and normalized `percentile_01_99`, giving `s ∈ [0,1]^{4096}` (line 112).
- The ground truth is downsampled to 64×64 with **nearest** interpolation, giving `G` (lines 71-73, 113).
- `p_i = s_i / Σ_j s_j` (lines 115-116).

| Measure | Formula | Line |
|---|---|---|
| Normalized entropy | `H = −Σ p_i log p_i`; reported as `H / log(4096)` | 117-118, 174 |
| Effective support | `exp(H) / 4096` | 175 |
| Top-1% mass | the 41 largest `s_i` divided by `Σ s` | 119-120 |
| FG/BG gap | `mean(s[G]) − mean(s[¬G])` | 179 |
| Standardized gap | `(mean(s[G]) − mean(s[¬G])) / std(s)`, where std is over all 4096 values (unbiased torch `.std()`) | 125, 180 |
| Boundary gradient ratio | `∇ = sqrt(dx² + dy²)` with backward absolute differences (first row/column = 0); boundary `B = dilate(G) XOR erode(G)` (3×3 cross); ratio `= mean(∇[B]) / mean(∇)` | 126-135, 182-184 |
| Centroid error | `‖(c_x,c_y)_map − (c_x,c_y)_GT‖₂ / ‖(63, 63)‖₂`; the map centroid is weighted by `s`, the GT centroid is the mean of GT pixel coordinates | 136-148 |
| Total variation | `(mean(dx) + mean(dy)) / 2` | 181 |
| Average precision | Mean of precision at each positive in a stable descending sort of `s` over 4096 pixels | 76-83 |
| ROC AUC | Mann–Whitney with tie-averaged ranks | 86-102 |
| Soft IoU | `Σ s·G / (Σ s + Σ G − Σ s·G)` | 160-161 |
| IoU / tp / fp / fn / tn | On the full-resolution (512²) binary prediction; `iou_at_64` uses the nearest-downsampled prediction | 149-170 |

Two caveats:
- **Implemented and released OVAM:** the threshold-free statistics use the fused CA×SA map, but IoU uses the post-CRF mask (`attention_analysis.md:38`).
- **Order of operations for every map:** it is re-normalized with `percentile_01_99` *before* the statistics. This includes OVAM, whose own pipeline uses max normalization.

---

## Open items for the reviewer

1. Get the top-4 head indices per layer from Faiz's `voc_sim_100_stage1_suite/results.json`, using the command in §2.3.
2. Record the library versions used for Faiz's live runs (`conda list -n trdp` on his account).
3. The head rankings were chosen on conditional, all-class-token, padded CA maps but are applied to unconditional, first-token, unpadded CA maps (§4.3).
4. Seven of the selection choices (module, seed, iterations, heads, steps, threshold, cleanup) were made on all 490 test images. Choices 1–3 were made on 99 images that are also part of the test set. No held-out number exists.
5. `docs/streaming_voc_ablations.md:16` says "30 denoising steps"; the code aggregates 31 UNet calls (indices 0–30).
