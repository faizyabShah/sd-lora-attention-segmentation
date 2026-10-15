**Controlled CA screens (VOC-Sim, 5 folds, selection at threshold 0.45). The single-axis screens promote candidates; only the combined confirmation (CA-C) freezes the CA.**

| Screen | Package | Reference | Vote winner (votes) | OOF mIoU reference | OOF mIoU choice | OOF gain | Paired Δ image IoU [95% CI] | Δ mIoU @0.35 / 0.45 / 0.55 | Outcome |
|---|---|---|---|---:|---:|---:|---|---|---|
| CA-T | A (Codex) | time/all_00_30 | time/all_00_30 (5/5) | 0.6661 | 0.6661 | 0.0000 | +0.0000 [+0.0000, +0.0000] | +0.0000 / +0.0000 / +0.0000 | reference won the fold vote |
| CA-R | A (Codex) | resolution/r8_r16_r32_r64 | resolution/r8_r16_r64 (5/5) | 0.6661 | 0.6957 | 0.0296 | +0.0211 [+0.0139, +0.0286] | +0.0009 / +0.0296 / +0.0442 | fold vote and OOF guard passed |
| CA-L | B (Claude) | L/all_16 | L/leave2_10_11 (5/5) | 0.6661 | 0.6923 | 0.0262 | +0.0209 [+0.0155, +0.0265] | -0.0021 / +0.0262 / +0.0501 | fold vote and OOF guard passed |
| CA-H | B (Claude) | H/0_1_2_3_4_5_6_7 | H/0_2_3_4_5_6_7 (4/5) | 0.6661 | 0.6741 | 0.0080 | +0.0089 [+0.0047, +0.0131] | -0.0002 / +0.0104 / +0.0193 | fold vote and OOF guard passed |
| CA-A | B (Claude) | A/cond_padded_all | A/cond_padded_all (5/5) | 0.6661 | 0.6661 | 0.0000 | +0.0000 [+0.0000, +0.0000] | +0.0000 / +0.0000 / +0.0000 | reference won the fold vote |
| CA-C | C (Claude) | C/reference | C/R (4/5) | 0.6653 | 0.6914 | 0.0260 | +0.0217 [+0.0145, +0.0293] | +0.0010 / +0.0303 / +0.0447 | fold vote and OOF guard passed |
