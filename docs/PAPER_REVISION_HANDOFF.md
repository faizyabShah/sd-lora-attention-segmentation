# Paper revision handoff to Claude — 26 October 2026

Codex rewrote the main manuscript at `paper/attention_segmentation.tex` and `paper/sec/0_abstract.tex` through `9_discussion.tex`, added five source-traceable figures in `paper/figures/`, updated citations, and added a nested component table plus a published class-word OVAM Table 1 comparison. The manuscript now has a coherent CA seed → SA propagation → second pass → fixed DenseCRF → joint temporal-selection story. It uses VOC-Sim for selection and COCO-Cap only for transfer.

## Review requested

Please **review; do not silently promote or replace a result**. In particular:

1. Check the exact implemented CA/SA/threshold/CRF description against the current code, especially the tokenizer-piece and normalization conventions.
2. Re-audit the nested SG22 values from `sg22/report.md` and the 20 class deltas from `sg22/per_class_iou.csv`. The figure shows two opposing SG18 **raw one-pass** examples, not final P2+R2 masks.
3. Check the published OVAM Table 1 transcription: DAAM .662/.484, Attn2Mask .687/.550, OVAM word .704/.582, Grounded Diffusion .621/.502, DatasetDM .803/.593. Optimized-token rows are omitted. Our regenerated-image scores are explicitly marked as non-paired with those paper numbers.
4. Check the A3 caveat. We report .823 forced OOF/.662 COCO as an **exploratory** implemented operator, not the registered selected path or an exact GoCA implementation. Protocol Section 103 documents the transpose issue. The full-VOC .824 is in-sample.
5. Review how SG25/SG27 appear as transfer observations without COCO-based promotion. SG28 is pending and absent from the paper.

Evidence and figure provenance are in `paper/README.md`; the generator is `scripts/paper/build_story_figures.py`. Codex performed a full LaTeX/BibTeX compile on this node with temporary compatibility stubs for the TeX Live 2018 packages `silence`, `cleveref`, and `multirow`; these stubs are **not committed**. Please build with the real packages on Overleaf or your environment before submission.

If you find a discrepancy, reply in the shared protocol with the exact file/line, source artifact, and proposed wording or correction. We should revise the manuscript together before treating it as submission-ready.
