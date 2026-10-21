# Pre-SG18 / Pre-SG21 diagnostic handoff

Question: does useful one-pass SA propagation show stable temporal or head-role structure?

Coverage: VOC 490 / 490; COCO 707 / 707 images.
The package contains all five endpoint ROC/PR results, class-balanced ROC, fixed stage counts, 31-call curves, 31×8 head maps, structural correlations, and VOC-to-COCO cell rank stability. No SG18/SG21 method was implemented.

Read `ANALYSIS_FOR_CHATGPT.md` first, then `results.json` and CSVs. Use the frozen go/no-go criteria in the supplied diagnostic specification.

Files: `ANALYSIS_FOR_CHATGPT.md`, `results.json`, `temporal_feature_auc.csv`, `head_call_metrics.csv`, `head_summary.csv`, `head_stage_summary.csv`, `call_summary.csv`, `structural_correlations.csv`, `heatmaps_*.png`, `temporal_curves_*.png`.
