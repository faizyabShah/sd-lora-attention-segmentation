# Narmyn: attention-segmentation work

| Document | Content |
|---|---|
| `evidence/EVIDENCE.md` | every experiment run so far (tier B exploratory, tier A controlled), where each setting came from, full per-configuration tables and figures; `evidence/evidence_section.tex` for the report (built by `scripts/narmyn/attention/build_evidence.py`) |
| `TOKEN_CRF_EXPERIMENT.md` | OVAM's released optimized tokens and DenseCRF on the earlier final method, on all 490 VOC-Sim and 707 COCO-Cap images (exploratory, pre-protocol) |
| `attention_report_for_review_d980b91.md` | code-level review of the VOC-Sim pipeline as of `d980b91`: exact final-method settings, baselines, metric code, selection provenance, open issues |
| `figures/ovam_sa_issue.png` | analysis of the upstream OVAM repository's all-ones-value hook and its rounding residue; this historical/reference-fidelity behavior is not the QK-derived SA operator used in our controlled experiments |
| `figures/ovam_sa_kernels.png` | the same upstream-hook residual under different attention kernels; retained to document the reference implementation, not as an open issue for the controlled pipeline |

Shared coordination documents (both collaborators edit these):

- `docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md`: the controlled CA/SA
  experiment protocol and its status table.
- `docs/ATTENTION_FUSION_AND_TOKEN_IDEAS.md`: backlog of CA/SA fusion operators
  and token methods for the deferred phase.

Code: `scripts/narmyn/attention/`, jobs: `jobs/narmyn/attention/`, results:
`outputs/segmentation/narmyn_token_crf/`.
