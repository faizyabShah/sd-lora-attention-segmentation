"""Generate one frozen-SD-1.5 image per Florence-2 caption.

The baseline arm of the comparison: stock Stable Diffusion 1.5, no LoRA, driven
by the captions from 02_caption.py. Port of the Kaggle notebook, adapted for
the full 24,995-caption set:

  * per-prompt seeding (generator seeded with the prompt's index) so a rerun
    reproduces the same image for the same prompt;
  * resume-safe -- an existing output PNG is skipped before any GPU work, so a
    job killed at the wall clock can simply be resubmitted;
  * prompt_mapping.json is written up front, not at the end, so the
    index -> (file_name, text) trace survives an interrupted run.

--start/--end slice the prompt list, which lets several jobs share the work
across GPUs (see generate_frozen.slurm).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

# ---------------------------------------------------------------- constants --
# Data and outputs live under $TRDP_ROOT when set (the pipeline working
# tree on the cluster); otherwise they resolve relative to this repo.
PROJECT_ROOT = Path(os.environ.get("TRDP_ROOT",
                                   Path(__file__).resolve().parents[2]))
PROMPTS_FILE = PROJECT_ROOT / "data" / "florence2_detailedcaptions" / "metadata.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output" / "frozen_images"
MAPPING_FILE = OUTPUT_DIR / "prompt_mapping.json"

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
HEIGHT = 768
WIDTH = 768
NUM_INFERENCE_STEPS = 30
PROGRESS_EVERY = 100


def load_entries():
    """Every {file_name, text} row, in file order -- index i is the seed."""
    with PROMPTS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_mapping(entries):
    """Index -> source crop + prompt, so generated images stay traceable.

    Derived entirely from metadata.jsonl, so we write it before generating
    rather than after: an interrupted run still leaves a usable mapping.
    """
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping = [
        {"image": f"{i:05d}.png", "prompt": e["text"], "original_file": e["file_name"]}
        for i, e in enumerate(entries)
    ]
    # Parallel slices all write identical content, so write-then-rename: two
    # jobs starting together must never leave a torn file. Last writer wins.
    tmp = MAPPING_FILE.with_suffix(f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    tmp.replace(MAPPING_FILE)
    return len(mapping)


def load_pipeline(device):
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    # 25k per-image progress bars would bury the log.
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=0, help="first prompt index")
    ap.add_argument("--end", type=int, help="stop before this prompt index")
    ap.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    ap.add_argument("--size", type=int, default=HEIGHT, help="square edge in px")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA is not available -- run this on a GPU node.", file=sys.stderr)
        return 1
    if not PROMPTS_FILE.is_file():
        print(f"ERROR: prompts file not found: {PROMPTS_FILE}", file=sys.stderr)
        return 1

    entries = load_entries()
    total_written = write_mapping(entries)

    start = max(0, args.start)
    end = min(len(entries), args.end if args.end is not None else len(entries))
    if start >= end:
        print(f"ERROR: empty slice [{start}, {end}) of {len(entries)} prompts",
              file=sys.stderr)
        return 1

    print(f"prompts file: {PROMPTS_FILE}", flush=True)
    print(f"output dir:   {OUTPUT_DIR}", flush=True)
    print(f"mapping:      {MAPPING_FILE} ({total_written} entries)", flush=True)
    print(f"{len(entries)} prompts; this job handles [{start}, {end}) "
          f"at {args.size}x{args.size}, {args.steps} steps", flush=True)

    todo = [i for i in range(start, end) if not (OUTPUT_DIR / f"{i:05d}.png").exists()]
    print(f"{end - start - len(todo)} already generated, {len(todo)} to go", flush=True)
    if not todo:
        print("nothing to do -- slice already complete")
        return 0

    print(f"loading {MODEL_ID} ...", flush=True)
    pipe = load_pipeline(args.device)
    if args.device == "cuda":
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    generated = 0
    t0 = time.time()

    for i in todo:
        out_path = OUTPUT_DIR / f"{i:05d}.png"
        generator = torch.Generator(args.device).manual_seed(i)
        image = pipe(
            entries[i]["text"],
            generator=generator,
            num_inference_steps=args.steps,
            height=args.size,
            width=args.size,
        ).images[0]

        # Write then rename: a job killed mid-save must not leave a truncated
        # PNG that the resume check would treat as finished.
        tmp = out_path.with_suffix(".tmp")
        image.save(tmp, format="PNG")
        tmp.replace(out_path)

        generated += 1
        if generated % PROGRESS_EVERY == 0:
            rate = generated / (time.time() - t0)
            eta = (len(todo) - generated) / rate / 3600
            print(f"  {generated}/{len(todo)} generated "
                  f"({rate:.2f} img/s, ~{eta:.1f} h left in this slice)", flush=True)

    elapsed = time.time() - t0
    on_disk = len(list(OUTPUT_DIR.glob("[0-9]*.png")))
    print(f"\nGenerated {generated} images in {elapsed / 3600:.2f} h")
    print(f"  {on_disk}/{len(entries)} images now in {OUTPUT_DIR}")
    if on_disk < len(entries):
        print("  slice incomplete -- resubmit to continue (existing images are skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
