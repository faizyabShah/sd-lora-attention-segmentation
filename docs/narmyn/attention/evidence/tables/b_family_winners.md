**Exploratory VOC-Sim experiments: best configuration per family (in-sample, 490 images)**

| Exp. | Family | Configurations | Best configuration | Best mIoU |
|---|---|---:|---|---:|
| B8 | baseline | 1 | ovam_unoptimized | 0.6540 |
| B8 | layers | 24 | mid_up | 0.6828 |
| B8 | resolution | 15 | r8_r16 | 0.7122 |
| B8 | self_attention/progressive_16-32 | 229 | iter4__non_residual__seed_t0.30__foreground | 0.7413 |
| B8 | self_attention/progressive_16-32-64 | 229 | iter2__non_residual__seed_t0.30__foreground | 0.7577 |
| B8 | self_attention/progressive_16-8 | 229 | iter1__residual_a0.50__seed_t0.30__background_difference | 0.6913 |
| B8 | self_attention/progressive_32-16 | 229 | iter2__residual_a0.50__seed_t0.30__background_difference | 0.7201 |
| B8 | self_attention/progressive_32-16-8 | 229 | iter2__residual_a0.50__seed_t0.30__background_difference | 0.6944 |
| B8 | self_attention/progressive_32-64 | 229 | iter4__non_residual__seed_t0.30__foreground | 0.7683 |
| B8 | self_attention/progressive_64-32 | 229 | iter4__residual_a0.75__seed_t0.70__background_difference | 0.7558 |
| B8 | self_attention/progressive_64-32-16 | 229 | iter4__residual_a0.50__seed_t0.70__background_difference | 0.7282 |
| B8 | self_attention/progressive_64-32-16-8 | 229 | iter4__residual_a0.50__seed_t0.30__background_difference | 0.7000 |
| B8 | self_attention/progressive_8-16 | 229 | iter1__non_residual__seed_t0.80__background_ratio | 0.7005 |
| B8 | self_attention/progressive_8-16-32 | 229 | iter1__non_residual__seed_t0.80__background_ratio | 0.7264 |
| B8 | self_attention/progressive_8-16-32-64 | 229 | iter4__residual_a0.75__seed_t0.30__background_difference | 0.7474 |
| B8 | self_attention/r16 | 778 | all_modules_all_steps__iter1__seed_t0.30__foreground_residual_a0.50 | 0.7161 |
| B8 | self_attention/r32 | 778 | module__up_blocks.2.attentions.2.transformer_blocks.0.attn1__iter4__co | 0.7641 |
| B8 | self_attention/r64 | 778 | module__up_blocks.3.attentions.2.transformer_blocks.0.attn1__iter4__co | 0.7777 |
| B8 | self_attention/r8 | 238 | window__steps_26_30__iter1__continuous__residual_a0.75 | 0.7034 |
| B8 | stage1/fusion | 118 | resolution_weights__16-1_32-0.5_64-2_8-0.5 | 0.6770 |
| B8 | stage1/heads | 212 | per_layer_top_4_heads | 0.6830 |
| B8 | stage1/postprocessing | 18 | fill_holes | 0.6661 |
| B8 | stage1/time_layer | 237 | all16__cumulative_00_25 | 0.6844 |
| B8 | threshold_normalization | 204 | all_16_layers__percentile_01_99__t0.40 | 0.6975 |
| B8 | time_resolution | 105 | all_00_30__r8_r16 | 0.7122 |
| B8 | time_window | 7 | steps_15_20 | 0.6752 |
| B9 | focused/cross_attention_seeds | 6 | top4_heads_percentile | 0.6831 |
| B9 | focused/output_calibration | 936 | top4_heads_percentile__r64__iter2__percentile_01_99__t0.45 | 0.7922 |
| B9 | focused/postprocessing | 90 | best12_percentile__r64__iter4__percentile_01_99__t0.45__remove_fractio | 0.7780 |
| B9 | focused/sa_heads | 108 | iter3__leave_head_6 | 0.7797 |
| B9 | focused/sa_time | 348 | iter3__trailing_10_30 | 0.7801 |
| B9 | focused/seed_operator | 144 | top4_heads_percentile__r64__iter2 | 0.7922 |
| B10 | joint/joint_head_time_threshold | 3900 | iter3__leave_3_7__steps_15_20__t0.40 | 0.7991 |
| B10 | joint/joint_postprocessing | 780 | iter4__leave_3_7__trailing_15_30__t0.45__remove_fraction_0.01 | 0.7997 |
| B15 | heads | 255 | heads_0_2_3_4_5_6_7 | 0.6764 |
| B15 | layers_structured | 39 | down_middle | 0.6818 |
| B15 | resolution | 15 | r8_r16_r64 | 0.6957 |
| B15 | time | 79 | cumulative_00_20 | 0.6752 |
| B15 | time_resolution | 1185 | cumulative_00_25__r8_r16 | 0.6968 |
