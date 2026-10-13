# Stage 1 and Stage 2 attention ablation summary

| Family | Experiment | Best configuration | mIoU |
|---|---|---|---:|
| stage1 | class_adaptive | `class_layers__percentile__t0.45` | 0.76908 |
| stage1 | fusion | `inverse_entropy_heads` | 0.74396 |
| stage1 | heads | `global_leave_head_4` | 0.74559 |
| stage1 | postprocessing | `remove_fraction_0.001` | 0.74424 |
| stage1 | time_layer | `best12__trailing_05_30` | 0.76246 |
| self_attention_resolution | 8 | `cross_attention_baseline` | 0.74394 |
| self_attention_resolution | 16 | `window__steps_26_30__iter4__continuous__residual_a0.25` | 0.74809 |
| self_attention_resolution | 32 | `all_modules_all_steps__iter4__seed_t0.70__background_ratio` | 0.80144 |
| self_attention_resolution | 64 | `all_modules_all_steps__iter4__seed_t0.60__background_difference` | 0.83551 |
| self_attention_progressive | 8-16 | `cross_attention_baseline` | 0.74394 |
| self_attention_progressive | 16-32 | `iter4__residual_a0.75__seed_t0.70__background_ratio` | 0.77418 |
| self_attention_progressive | 32-64 | `iter4__residual_a0.75__seed_t0.60__background_ratio` | 0.81808 |
| self_attention_progressive | 8-16-32 | `cross_attention_baseline` | 0.74394 |
| self_attention_progressive | 16-32-64 | `iter4__residual_a0.75__seed_t0.60__foreground` | 0.80482 |
| self_attention_progressive | 8-16-32-64 | `iter4__residual_a0.75__seed_t0.70__foreground` | 0.77437 |
| self_attention_progressive | 64-32 | `iter4__residual_a0.75__seed_t0.60__background_ratio` | 0.80650 |
| self_attention_progressive | 32-16 | `iter4__residual_a0.50__seed_t0.40__background_difference` | 0.74960 |
| self_attention_progressive | 16-8 | `cross_attention_baseline` | 0.74394 |
| self_attention_progressive | 64-32-16 | `iter4__residual_a0.50__seed_t0.60__background_ratio` | 0.76051 |
| self_attention_progressive | 32-16-8 | `cross_attention_baseline` | 0.74394 |
| self_attention_progressive | 64-32-16-8 | `cross_attention_baseline` | 0.74394 |

Best overall: **0.83551 mIoU**, `all_modules_all_steps__iter4__seed_t0.60__background_difference` from self_attention_resolution/64.
