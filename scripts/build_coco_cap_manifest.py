#!/usr/bin/env python3
"""Build the annotated OVAM COCO-Cap manifest from the released files."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path(
    "datasets/ovam_experiment_with_dataset/evaluation/ovam/coco_captions"
)
ANNOTATION_PATTERN = re.compile(
    r"^(?P<classname>.+)_sd-15_caption_(?P<caption_id>\d+)_"
    r"(?P=classname)_(?P<seed>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captions",
        type=Path,
        default=Path(
            "datasets/ovam_experiment_with_dataset/evaluation/coco_captions_sampled.csv"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_rows(captions: Path, dataset_root: Path) -> list[dict[str, str]]:
    with captions.open(newline="", encoding="utf-8") as handle:
        caption_rows = list(csv.DictReader(handle))
    by_caption = {row["caption_id"]: row for row in caption_rows}
    if len(by_caption) != len(caption_rows):
        raise ValueError("caption_id is not unique in the sampled COCO captions")

    rows: list[dict[str, str]] = []
    for annotation in sorted((dataset_root / "annotations").glob("*.png")):
        match = ANNOTATION_PATTERN.fullmatch(annotation.stem)
        if match is None:
            raise ValueError(f"Unexpected annotation filename: {annotation.name}")
        fields = match.groupdict()
        source = by_caption.get(fields["caption_id"])
        if source is None:
            raise ValueError(f"No caption metadata for {annotation.name}")
        if source["category_name"] != fields["classname"]:
            raise ValueError(
                f"Class mismatch for {annotation.name}: {source['category_name']}"
            )
        image_name = (
            f"{fields['classname']}_sd15_caption{fields['caption_id']}_"
            f"seed{fields['seed']}.png"
        )
        image_path = dataset_root / "images" / image_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append(
            {
                "image_id": (
                    f"ovam-coco-{fields['caption_id']}-"
                    f"{fields['classname'].replace(' ', '_')}-{fields['seed']}"
                ),
                "classname": fields["classname"],
                "prompt": source["caption"].lower(),
                # OVAM's COCO script encodes the category separately so it can
                # localize examples whose caption uses a synonym or omits it.
                "attribution_prompt": fields["classname"].lower(),
                "model": "runwayml/stable-diffusion-v1-5",
                "seed": fields["seed"],
                "num_inferece_steps": "30",
                "annotated": "True",
                "image": f"images/{image_name}",
                "annotation": f"annotations/{annotation.name}",
                "caption_id": fields["caption_id"],
                "coco_image_id": source["image_id"],
                "word_included": source["word_included"],
            }
        )
    if not rows:
        raise RuntimeError(f"No annotations found below {dataset_root}")
    return rows


def main() -> None:
    args = parse_args()
    output = args.output or args.dataset_root / "ovam_coco_cap.csv"
    rows = build_rows(args.captions, args.dataset_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["classname"] for row in rows)
    print(f"Wrote {len(rows)} annotated rows across {len(counts)} classes to {output}")
    print("Class counts:", ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))


if __name__ == "__main__":
    main()

