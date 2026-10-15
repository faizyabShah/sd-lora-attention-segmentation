**Where each setting of the method came from**

| Setting | Value used so far | Chosen in | Images used to choose | Evidence | Controlled status |
|---|---|---|---|---|---|
| CA layers | 12 of 16 (drop 1, 3, 4, 8) | B4 exhaustive search | 99 dev images | ledger B4; Fig. a1_exhaustive | A1 CA-L (CV): drop modules 10, 11; A3 CA-C: subsumed by CA-R (frozen: 11 modules at 8, 16, 64) |
| CA heads | fixed top-4 per layer | B6 stage-1 suite | 99 dev images | ledger B6 | A1 CA-H (CV): drop head 1; A3 CA-C: adds nothing on top of CA-R (tie), so all 8 heads frozen |
| CA attribution | OVAM: unconditional query, unpadded softmax, first piece | inherited from OVAM | — | Table a1_ca_a | A1 CA-A (CV): reference (conditional, padded, all pieces) kept; OVAM's cell is −0.0156 CA-only |
| CA calls | all 31 (0–30) | B1/B8 windows; B14 | 99 dev / 490 | ledger B1, B8, B14; Fig. a2_ca_t_calls | A2 CA-T (CV): keep all calls 0–30 |
| CA resolutions | all four (8, 16, 32, 64) | B1/B8 resolution screens | 99 dev / 490 | Table a2_ca_r; Fig. a2_ca_r_subsets | A2 CA-R (CV): keep 8+16+64; A3 CA-C: frozen (4/5 votes, OOF +0.026) |
| CA normalization, threshold | percentile 1–99, ≥ 0.45 | B5 | 99 dev images | ledger B5 | protocol fixes min–max, 0.45 for the CA phase |
| SA module | U3.2 (up_blocks.3.attentions.2 attn1, 64×64) | B8 SA screens | 490 (in-sample) | ledger B8 | SA-L (Claude), seeded by the frozen CA |
| SA passes | 4 (revised: 1) | B10; revised from B11/B13 | 490; 1-pass informed by COCO-Cap | Fig. b_joint_passes_time | after SA freeze |
| SA heads | {0,1,2,4,5,6} | B10 | 490 (in-sample) | Fig. b_joint_heads_threshold | SA-H (Claude), seeded by the frozen CA |
| SA calls | 15–30 | B10 | 490 (in-sample) | Fig. b_joint_passes_time | SA-T (Codex), seeded by the frozen CA |
| Mask threshold | 0.45 | B5, B9, B10 | 99 dev / 490 | Fig. b_joint_heads_threshold | fixed at 0.45; 0.35/0.55 as diagnostics |
| Cleanup | remove components < 1% | B10 | 490 (in-sample) | Table b_component_ablations | deferred phase |
| DenseCRF | not in the final method | B17 (nothing selected) | — | Table b_token_crf | deferred phase |
| Optimized token | not in the final method | B17 (nothing selected; tokens trained by OVAM on separate images) | — | Table b_token_crf | deferred phase |
