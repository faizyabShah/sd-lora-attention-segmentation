# Narmyn: attention-segmentation work

| Document | Content |
|---|---|
| `TOKEN_CRF_EXPERIMENT.md` | OVAM's released optimized tokens and DenseCRF on the earlier final method, on all 490 VOC-Sim and 707 COCO-Cap images (exploratory, pre-protocol) |
| `attention_report_for_review_d980b91.md` | code-level review of the VOC-Sim pipeline as of `d980b91`: exact final-method settings, baselines, metric code, selection provenance, open issues |
| `figures/ovam_sa_issue.png` | why OVAM's self-attention step carries almost no signal: attention applied to an all-ones value tensor returns 1 for every pixel in exact arithmetic, so the map is rounding residue stretched by min–max scaling |
| `figures/ovam_sa_kernels.png` | the same residual under different attention kernels: its structure, and the sign of its correlation with the object, depend on the arithmetic kernel |

Shared coordination documents (both collaborators edit these):

- `docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md`: the controlled CA/SA
  experiment protocol and its status table.
- `docs/ATTENTION_FUSION_AND_TOKEN_IDEAS.md`: backlog of CA/SA fusion operators
  and token methods for the deferred phase.

Code: `scripts/narmyn/attention/`, jobs: `jobs/narmyn/attention/`, results:
`outputs/segmentation/narmyn_token_crf/`.
