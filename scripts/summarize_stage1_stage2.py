"""Collect completed Stage-1 and Stage-2 ablation winners."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("outputs/segmentation")
STAGE1 = ROOT / "voc_sim_100_stage1_suite/results.json"
SA_ROOT = ROOT / "voc_sim_100_self_attention"
SEQUENCES = (
    "8-16", "16-32", "32-64", "8-16-32", "16-32-64", "8-16-32-64",
    "64-32", "32-16", "16-8", "64-32-16", "32-16-8", "64-32-16-8",
)


def winner(payload: dict) -> dict:
    name = payload["best"]
    return {"configuration": name, **payload["results"][name]}


def main() -> None:
    stage1 = json.loads(STAGE1.read_text(encoding="utf-8"))
    summary = {
        "stage1": {
            group: {"configuration": name, **stage1["results"][group][name]}
            for group, name in stage1["best"].items()
        },
        "self_attention_resolution": {},
        "self_attention_progressive": {},
    }
    for resolution in (8, 16, 32, 64):
        payload = json.loads((SA_ROOT / f"r{resolution}/results.json").read_text(encoding="utf-8"))
        summary["self_attention_resolution"][str(resolution)] = winner(payload)
    for sequence in SEQUENCES:
        payload = json.loads(
            (SA_ROOT / f"progressive_{sequence}/results.json").read_text(encoding="utf-8")
        )
        summary["self_attention_progressive"][sequence] = winner(payload)

    candidates = []
    for family, values in summary.items():
        for experiment, value in values.items():
            candidates.append((value["miou"], family, experiment, value["configuration"]))
    best_miou, family, experiment, configuration = max(candidates)
    summary["best_overall"] = {
        "family": family,
        "experiment": experiment,
        "configuration": configuration,
        "miou": best_miou,
    }
    output_dir = ROOT / "voc_sim_100_stage1_stage2_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage 1 and Stage 2 attention ablation summary",
        "",
        "| Family | Experiment | Best configuration | mIoU |",
        "|---|---|---|---:|",
    ]
    for family_name in ("stage1", "self_attention_resolution", "self_attention_progressive"):
        for experiment_name, value in summary[family_name].items():
            lines.append(
                f"| {family_name} | {experiment_name} | `{value['configuration']}` | {value['miou']:.5f} |"
            )
    lines.extend(
        [
            "",
            f"Best overall: **{best_miou:.5f} mIoU**, `{configuration}` "
            f"from {family}/{experiment}.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
