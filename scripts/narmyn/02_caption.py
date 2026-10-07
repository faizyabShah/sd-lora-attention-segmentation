"""Caption the cropped Cityscapes frames with Florence-2-large.

Port of the Colab notebook that captioned the 5k train/val/test set, adapted to
run unattended over all 24,995 crops from 01_crop.py:

  * checkpoints every CHECKPOINT_EVERY captions and resumes from the checkpoint,
    so a job that hits the wall clock picks up where it stopped;
  * keys every entry by its path relative to the crop root, i.e.
    <split>/<city>/<file>.png, pooled into one metadata.jsonl;
  * reuses 01_crop.is_valid_image() to skip any unreadable or all-black crop.

Use --city / --limit to smoke-test on a subset. Those runs write to tagged
output and checkpoint filenames so a partial test can never clobber or poison
the full run's metadata.jsonl.
"""

import os
import argparse
import importlib
import json
import random
import re
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

# ---------------------------------------------------------------- constants --
# Data and outputs live under $TRDP_ROOT when set (the pipeline working
# tree on the cluster); otherwise they resolve relative to this repo.
PROJECT_ROOT = Path(os.environ.get("TRDP_ROOT",
                                   Path(__file__).resolve().parents[2]))
INPUT_ROOT = PROJECT_ROOT / "data" / "cityscapes_cropped"
# Sits beside florence2_detailedcaptions_5k, the earlier 5k-image Colab run.
OUTPUT_DIR = PROJECT_ROOT / "data" / "florence2_detailedcaptions"
CHECKPOINT_DIR = PROJECT_ROOT / "output" / "captions"

CHECKPOINT_EVERY = 100

MODEL_ID = "microsoft/Florence-2-large"
TASK = "<DETAILED_CAPTION>"
NUM_BEAMS = 3
MAX_NEW_TOKENS = 1024

# Florence-2 opens detailed captions with boilerplate; drop it and re-capitalize.
PREFIX_RE = re.compile(
    r"^(?:the image (?:shows|depicts|features)|this image shows)\s+",
    re.IGNORECASE,
)

# is_valid_image lives in 01_crop.py, whose name is not a valid identifier.
sys.path.insert(0, str(Path(__file__).resolve().parent))
is_valid_image = importlib.import_module("01_crop").is_valid_image


def fixed_get_imports(filename):
    """Drop flash_attn from Florence-2's declared imports.

    Florence-2's remote code lists flash_attn unconditionally but never needs
    it when attn_implementation="sdpa"; without this the load fails outright.
    """
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def load_model(device):
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
    print(f"{MODEL_ID} loaded", flush=True)
    if device == "cuda":
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    return processor, model


def clean_caption(text: str) -> str:
    text = PREFIX_RE.sub("", text.strip())
    return text[0].upper() + text[1:] if text else text


def caption_one(processor, model, device, path: Path) -> str:
    image = Image.open(path).convert("RGB")
    inputs = processor(text=TASK, images=image, return_tensors="pt").to(
        device, torch.float16
    )
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=NUM_BEAMS,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=TASK, image_size=(image.width, image.height)
    )
    return clean_caption(parsed[TASK])


def collect_images(city: str | None, limit: int | None):
    """Every crop, sorted for reproducibility, optionally filtered."""
    paths = sorted(INPUT_ROOT.rglob("*.png"))
    if city:
        paths = [p for p in paths if p.parent.name == city]
    if limit:
        paths = paths[:limit]
    return paths


def save_outputs(results, metadata_path: Path, captions_path: Path):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with captions_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def report(results):
    if not results:
        print("no captions produced")
        return
    print("=" * 70)
    print("SAMPLE CAPTIONS")
    print("=" * 70)
    for entry in random.sample(results, min(5, len(results))):
        print(f"\nFile: {entry['file_name']}")
        print(f"Caption: {entry['text']}")
        print("-" * 70)

    lengths = [len(e["text"].split()) for e in results]
    print("\nCaption statistics:")
    print(f"  Total captions: {len(results)}")
    print(f"  Average length: {sum(lengths) / len(lengths):.0f} words")
    print(f"  Shortest: {min(lengths)} words")
    print(f"  Longest: {max(lengths)} words")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", help="only caption crops in this city directory")
    ap.add_argument("--limit", type=int, help="only caption the first N crops")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA is not available -- run this on a GPU node.", file=sys.stderr)
        return 1

    if not INPUT_ROOT.is_dir():
        print(f"ERROR: crop root does not exist: {INPUT_ROOT}", file=sys.stderr)
        return 1

    # A filtered run gets its own output + checkpoint names so smoke tests
    # never overwrite or seed the full run's files.
    tag = args.city or (f"first{args.limit}" if args.limit else None)
    if tag:
        metadata_path = OUTPUT_DIR / f"metadata.{tag}.jsonl"
        captions_path = OUTPUT_DIR / f"captions.{tag}.json"
        checkpoint_path = CHECKPOINT_DIR / f"_checkpoint.{tag}.json"
    else:
        metadata_path = OUTPUT_DIR / "metadata.jsonl"
        captions_path = OUTPUT_DIR / "captions.json"
        checkpoint_path = CHECKPOINT_DIR / "_checkpoint.json"

    paths = collect_images(args.city, args.limit)
    if not paths:
        print(f"ERROR: no PNGs found under {INPUT_ROOT}"
              + (f" for city {args.city!r}" if args.city else ""), file=sys.stderr)
        return 1

    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as f:
            results = json.load(f)
        done = {r["file_name"] for r in results}
        print(f"Resuming from checkpoint: {len(results)} already captioned", flush=True)
    else:
        results, done = [], set()

    print(f"crop root:  {INPUT_ROOT}", flush=True)
    print(f"metadata:   {metadata_path}", flush=True)
    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(f"{len(paths)} crops to consider ({len(done)} already done)", flush=True)

    processor, model = load_model(args.device)

    skipped_invalid = 0
    since_checkpoint = 0

    for i, path in enumerate(paths, 1):
        rel = str(path.relative_to(INPUT_ROOT))
        if rel in done:
            continue

        if not is_valid_image(path):
            # is_valid_image already logs the reason.
            skipped_invalid += 1
            continue

        results.append({"file_name": rel, "text": caption_one(
            processor, model, args.device, path)})
        done.add(rel)
        since_checkpoint += 1

        # Count new captions rather than len(results), so the cadence stays
        # correct after a resume.
        if since_checkpoint >= CHECKPOINT_EVERY:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint_path.open("w", encoding="utf-8") as f:
                json.dump(results, f)
            since_checkpoint = 0
            print(f"  [checkpoint: {len(results)} captioned, {i}/{len(paths)} scanned]",
                  flush=True)

    save_outputs(results, metadata_path, captions_path)
    checkpoint_path.unlink(missing_ok=True)

    print(f"\nSaved {len(results)} captions")
    print(f"  JSONL: {metadata_path}")
    print(f"  JSON:  {captions_path}")
    if skipped_invalid:
        print(f"  skipped (invalid crop): {skipped_invalid}")
    report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
