# Project state handoff (2026-11-25): where we stand

For a fresh Claude chat. This is the big picture: what the project is, which method is current, what has been established,
what is running, and what is left. Internal names (A3, V0/V3, MK4, LR, SGxx) are used here for precision; **never put them
in manuscript prose.**

---

## 1. The project in one paragraph

Training-free **class-word segmentation from Stable Diffusion 1.5 attention**. Given an image and a class word, the method
produces one binary mask for that class. There are two papers:

| | Paper 1 (`paper/`) | Paper 2 (`paper2/main.tex`), the **active** one |
|---|---|---|
| Method | **BRACE**: A3 cross-attention seed → V3 capped 16→32→64 self-attention hierarchy → DenseCRF R2, threshold 0.55 | uniform CA seed → **negative-contrast gate (MK4)** → **low-res consensus (MK4-LR, γ = 1)** → fine-only 64×64 propagation (V0) → min-max → fixed 0.55 (DenseCRF optional) |
| Status | **Frozen** (owner override, protocol §135; not a registered promotion; keep that distinction) | Method frozen for the paper; tables being filled |
| Images | generated benchmarks (VOC-Sim, COCO-Cap from OVAM) | generated **and** real photos |
| Venue | — | ICPR 2027 (LNCS template) |

Owner: Faizyab (Faiz, cluster user `shafa`). Collaborators: Narmeen (user `sidna`, her own Claude, audits) and JC (reviewer,
comments tracked in `docs/JC_REVIEW_WORKPLAN.md`). Don't message collaborators unless told to.

---

## 2. Paper 2 method (frozen)

1. **One noised SD 1.5 pass** per (photo, word) at t = 100, noise seed 0, prompt "A photograph of a {word}"
   (`jc_realvoc.one_pass`). On generated images the generation's own attention is used.
2. **Uniform cross-attention seed** (uniform module weights; no SA-guided weights).
3. **Reference profiles from captioned photos, label-free:** COCO train2017 photos are selected by caption match only
   (20 positives whose captions mention the word, 480 negatives that don't). The object profile comes from the top 5% of
   queries. The 4 negative centroids come from the top 1% activations on non-mentioning photos (spherical k-means).
   No masks or labels are used.
4. **MK4 gate:** a per-module nearest-negative contrast gate (object profile vs 4 negative centroids).
5. **MK4-LR:** the mean of the six gates below 64×64, upsampled and applied at every module, γ = 1.
6. **Fine-only V0 propagation:** two 64×64 SA steps, min-max, **fixed threshold 0.55**. DenseCRF R2 is secondary.

Why these choices (evidence trail, all in `runs/3x–4x/result.md`):
- **LR over MK4.** Confirmed on untouched COCO val2017 (run 40): V3 +0.017, V0 +0.031 [CI > 0].
- **No hierarchy.** It hurts on real photos once LR is on (V3 − V0 = −0.029 in run 40, −0.016 in run 37) and is about
  neutral on generated images.
- **No SA weights.** They add about +0.005 once LR is on.
- **γ = 1, not 2.** γ = 2 was selected on dev in runs 37/38, but γ = 1 gives similar numbers and was fixed in advance for
  runs 40–44.
- **Robust consensus (median/trim) failed** to beat LR end to end (run 41).
- **Word not in the prompt (OVAM sense).** Confirmed on 60 COCO classes never used in development (run 42): LR − none
  +0.022, LR − MK4 +0.014. The gain is only 0.24× the seen-class gain, and LR is flat across 5/20/100 profile photos.
- **Framing rule:** say "extends to classes whose word is not in the prompt"; **never** claim "open-vocabulary
  segmentation".

---

## 3. Results established so far

### 3a. Generated benchmarks (run 43, confirmatory; fills paper2 Table 2)
Profiles come from the other half of the same set; raw at 0.55.

| | VOC-Sim | COCO-Cap |
|---|---:|---:|
| **Ours (MK4-LR, V0) raw** | 0.801 | 0.736 |
| Ours + DenseCRF | 0.805 | 0.744 |
| No gate (ablation) | 0.782 | 0.662 |
| MK4 | 0.778 | 0.695 |
| NERVE (official, re-run) | 0.805 | 0.702 |
| SCLIP (re-run) | 0.719 | 0.583 |

Primary contrasts:
- **P1** (ours − no gate, VOC-Sim): +0.019 [+0.010, +0.028], helps.
- **P2** (ours − no gate, COCO-Cap): +0.074 [+0.060, +0.088], helps.
- **P5** (ours − no gate, COCO-Cap, word not in prompt): +0.078, helps.
- **P3** (ours − NERVE, VOC-Sim): −0.005, not resolved (tie).
- **P4** (ours − NERVE, COCO-Cap): +0.033 [+0.007, +0.060], beats NERVE.

Other readings:
- Few-shot profiles (5 positives) give 0.794 / 0.722.
- SA weights add about 0 with LR. V3 adds +0.027 on VOC-Sim only (descriptive).
- **Why run 38 showed about 0.80 on COCO-Cap and run 43 shows 0.736:** run 38 scored only the easier test half
  (0.762 vs 0.712 on the dev half). The rest came from a threshold and γ = 2 tuned on dev, plus CRF. Run 43 is untuned
  and covers the full set (`runs/43_generated_benchmarks/posthoc_run38_halves.md`).
- NERVE cross-check: Narmeen's run of the same variant gave 0.803 / 0.701, ours 0.805 / 0.702.
- ChatGPT handoff for run 43: `outputs/segmentation/analysis/RUN43_CHATGPT_HANDOFF.md`.

### 3b. Real photos
- **Run 37** (real VOC val, exploratory, third look at VOC val): LR/RW beat MK4 by +0.02 (V3) and +0.04 (V0). The best,
  V0 + RW, reaches 0.687 vs BRACE-equivalent 0.608.
- **Run 40** (COCO val2017, 20 VOC classes, confirmatory): see §2.
- **Run 42** (60 unseen COCO classes, confirmatory): see §2. Unseen raw V0 is none 0.438, MK4 0.445, LR 0.459.

### 3c. Closed directions (don't re-propose without new evidence)
See `runs/FAILURE_INVENTORY.md` and the memory notes. In brief:
- SD-only seed competition and background keys.
- Multi-timestep capture.
- SN, IB, GM, MP and Fisher gate variants.
- Robust consensus.
- Per-image thresholding.
- Equivariance and noise ensembles.
- Embedding and subspace manipulation.
- VAE ownership.
- Regeneration replay.
- Late and early CLIP fusion (registered fails; multiplicative wins were post hoc only).

---

## 4. Run 44, in progress: real-photo comparison (fills paper2 Table 3 `tab:real`)

- **Pre-registered** in commit `a7a10b4cd` (`runs/44_real_benchmarks/PREREG.md`). Plan and owner decisions are in
  `PLAN.md` §2, §6 and §8.
- **Protocol:** class word given; one binary mask per (photo, class) item. Primary items have class area ≥ 1% on all three
  columns. Class-macro mIoU; 2,000-sample paired photo bootstrap. Nothing is tuned for any method.

| Column | Classes | Photos | Primary items |
|---|---:|---:|---:|
| PASCAL VOC 2012 val | 20 | 1,409 | 1,947 |
| PASCAL Context val | 59 | 5,104 | 22,691 |
| COCO-Object (val2017 minus the §136 set) | 80 | 4,571 | 9,816 |

- **Primary contrasts:**
  - P1–P3: ours − no gate (VOC / Context / COCO-Object).
  - P4–P6: ours − NERVE (VOC / Context / COCO-Object).
- **Baselines** (12 methods, all re-run as released):
  - DAAM and OVAM (one noised pass, our latent).
  - DiffSegmenter (fixed 0.5).
  - iSeg (TagCLIP labels as primary; class given as secondary).
  - SCLIP, ClearCLIP (`model_type='ClearCLIP'`), NACLIP, ProxyCLIP.
  - NERVE.
  - OVDiff (README pipeline, 32 support images per class).
- **VOC21 reproductions** check that each repo runs as published:

  | Method | Reproduced | Published |
  |---|---:|---:|
  | SCLIP | 59.64 | 59.1 |
  | ClearCLIP | 51.80 | 51.8 |
  | NACLIP | 64.06 | 64.1 |
  | ProxyCLIP | 61.24 | 61.3 |

### Job status (SLURM, as of the 2026-11-25 check; logs in `/home/shafa/sg_comp/tmp/r44/logs/`)

| Part | Status |
|---|---|
| Ours: profile dump 0 (263488) | done |
| Ours: VOC capture (263491) | done |
| Ours: COCO captures (263492/3) | done |
| Ours: profile dump 1 (263489) → fit (263513) → Context captures (263514–6) | queued |
| ClearCLIP, NACLIP, ProxyCLIP, NERVE, iSeg | **done, all 3 columns** |
| SCLIP | VOC and Context done. COCO ran out of memory on a 16 GB V100 at photo 89/4,571; **resubmitted on an A100 as 263549** (the driver resumes), and score-coco now waits for it |
| DiffSegmenter (263503–5) | 2 of 3 running |
| DAAM (263506/7), OVAM (263508–10) | queued |
| OVDiff: sample (263517/20/23) → proto (263518/21/24) → predict (263519/22/25), per column | queued. **Its proto and predict steps are untested**; the VOC chain is their first real test |
| Scorers (263526–8, afterany on all method jobs) → report (263529) | queued. The report writes `runs/44_real_benchmarks/result.md` |

**Constraints:**
- 4 GPUs per user.
- The cluster is crowded; another user has about 95 A100/V100 jobs queued.
- Only renyi has A100s (2). erdos, wald and neumann have V100s with 16 GB.

**If something fails:**
- Fix the job and resubmit it. All drivers skip outputs that already exist.
- `jc_score44.py` skips score files that already exist. To rescore a column, **move** (don't delete) its `scores_<col>` file
  first.
- If OVDiff cannot be made to work, report it as "not run" with the reason (pre-registered).

**After the results:**
1. Commit `result.md`.
2. Report in ChatGPT style.
3. Fill Table 3 when the owner asks.
4. Write a run 44 memory note.

---

## 5. Paper 2 manuscript status (`paper2/main.tex`)

| Element | Status |
|---|---|
| Title, abstract, intro, related-work prose, method prose | **skeleton only** (comment placeholders) |
| Table 1 `tab:sets` | filled |
| Table 2 `tab:generated` | **filled** from run 43 (%, with word-in-prompt split; published rows quoted) |
| Table 3 `tab:real` | columns set; awaiting run 44 |
| `tab:multiclass` (VOC20/VOC21/COCO-Object multi-class) | empty; **optional** (no multi-class rule; see framing) |
| `tab:openvocab` (seen 20 vs other 60; 5/20/100 photos) | empty; data exists from runs 40/42 |
| `tab:ablation`, `tab:gate` | empty; data mostly exists from runs 37–42 (some cells need run 44 or new runs) |
| Figures (teaser, method, qualitative, open-vocab, why-low-res) | placeholders |
| Supplementary tables | empty |
| `refs.bib` | some entries still `\citeneeded` (DiffSegmenter, iSeg, ClearCLIP, NACLIP, ProxyCLIP, dataset cites); Attn2Mask venue (CVPRW?) to confirm |

When giving .tex snippets, put them in chat and keep internal names out of the prose.

---

## 6. Rules that must be kept

- **Science:**
  - BRACE is frozen. Paper 2's method is frozen as in §2.
  - Inference is label-free; ground truth is for scoring only.
  - Pre-register and commit before scoring. Report failures. Label results as exploratory / in-sample / OOF /
    fixed-threshold / confirmatory.
  - Compare variants only within one capture.
  - Fit profiles, γ, weights and thresholds on **real** photos for real-photo runs, never on generated images.
  - VOC-Sim is for development; COCO-Cap is untuned transfer.
- **Files:** delete only after an explicit yes from the owner. Move smoke outputs to `_smoke`, don't delete them.
- **Git:** commit locally; **push only when asked** (about 55 commits are currently unpushed on `main`). Commit trailer:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Compute:**
  - **Never run GPU or compute work on the login node** (it has no numpy).
  - Use SLURM: `-p gpu --gres=gpu:1`, or `-p cpu` for CPU work; `--mem` must be an integer; MaxTime 12:00:00.
  - The node `wald` may be used.
  - Helper scripts go in `/home/shafa/sg_comp/tmp/`.
- **Storage:** the NAS is about 97% full (about 1.2 TB free).
- **Answer style:** a bold one-line answer, numbered sections, tables, and an explicit split between what is established and
  what is not.

---

## 7. Where things live

- **Cache:** `BRACE_CACHE_DIR=/home/shafa/sg_comp`. Run 44 data is in `/home/shafa/sg_comp/jc_real44/`.
- **Run 44 scripts** (`scripts/`):
  - `jc_real44.py`: items.
  - `jc_names44.py`: name maps.
  - `jc_ours44.py`: ours, including the SYN44 caption rules for 33 Context words.
  - `jc_clip44.py`, `jc_nerve44.py`, `jc_iseg44.py`, `jc_diffseg44.py`, `jc_daamovam44.py`, `jc_ovdiff44.py`: baselines.
  - `jc_score44.py`: score and report.
  - Job scripts are in `runs/44_real_benchmarks/jobs/` (`submit.sh`, `submit2.sh`, `jobs*.txt`).
- **Earlier runs:** `scripts/jc_eval43.py`, `jc_nerve43.py` (run 43); `jc_ov42.py` (run 42); `jc_coco40.py` (run 40).
- **Conda environments:**

  | Environment | Used for |
  |---|---|
  | `trdp` | ours |
  | `clipseg` | CLIP methods |
  | `iseg` | iSeg |
  | `ovam44` | DAAM / OVAM |
  | `ovdiff` | OVDiff |
  | `/home/sidna/miniconda3/envs/nerve` | NERVE (read-only) |
  | `/home/sidna/miniconda3/envs/diffseg` | DiffSegmenter |

- **Baseline repos:** `other repositories/<name>`.
- **Protocol and records:**
  - `docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md`;
  - `runs/FAILURE_INVENTORY.md`, `runs/IDEAS_TO_TRY.md`;
  - per-run `PREREG.md` / `result.md`.
- **Memory notes** (one per run): `~/.claude/projects/-home-shafa-sd-lora-attention-segmentation/memory/`.

---

## 8. Open items and next steps

1. **Run 44:** let the chain finish. Check that SCLIP COCO (263549) and the OVDiff proto/predict steps succeed, then read
   `result.md`.
2. **Fill paper2 Table 3**, then the tables whose data already exists: `tab:openvocab` (runs 40/42), `tab:ablation` and
   `tab:gate` (runs 37–42; list any missing cells before running anything new).
3. **Write the prose:** abstract, intro, method, protocol, limitations (VOC val was seen in development; caption rules are
   noisy; the unseen gain is small; the method relies on SD 1.5).
4. **Figures:** teaser, method diagram, qualitative examples, the per-class unseen-gain figure, and the per-module AUROC
   figure for "why low resolution".
5. **Citations:** fill the `\citeneeded` entries.
6. **Owner decisions:** whether to fill `tab:multiclass`; when to push.
