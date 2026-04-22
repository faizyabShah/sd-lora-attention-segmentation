"""Script entrypoint to run evaluation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.blip_score import compute_blip_score
from src.evaluation.clip_score import compute_clip_score
from src.evaluation.fid import compute_fid
from src.evaluation.lpips import compute_lpips


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated images against a real dataset.")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    paths = config["paths"]
    captions_df = pd.read_csv(paths["captions_csv"])
    captions = captions_df["caption"].fillna("").tolist()

    fid_score = compute_fid(paths["real_dir"], paths["generated_dir"])
    lpips_score = compute_lpips(paths["real_dir"], paths["generated_dir"])
    clip_score = compute_clip_score(paths["generated_dir"], captions)
    blip_score = compute_blip_score(paths["generated_dir"], captions)

    output_dir = Path(paths["metrics_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.csv"
    results_df = pd.DataFrame(
        [
            {
                "FID": fid_score,
                "LPIPS": lpips_score,
                "CLIPScore": clip_score,
                "BLIPScore": blip_score,
            }
        ]
    )
    results_df.to_csv(result_path, index=False)
    print(results_df.to_string(index=False))
    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    main()
