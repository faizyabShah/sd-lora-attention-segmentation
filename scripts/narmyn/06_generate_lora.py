"""Generate one image per caption using a trained LoRA adapter.

The treatment arm matching 03_generate_frozen.py's baseline. Prompt list, order,
seeding, size and step count are taken from that script so the two sets are
paired: prompt i gets the same caption and the same initial latent under the
frozen model and under every rank. Only the adapter differs, which is what makes
a per-image comparison (CLIP/BLIP) and a matched FID/KID meaningful.

Resume-safe, and deliberately so: this is meant to run on gpu_lowpriority, where
a higher-tier job preempts and requeues it. On restart the finished images are
skipped, so a preemption costs only the partial image in flight.
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

# Reuse the baseline's prompt loading and constants so the two arms cannot
# drift apart; 03_generate_frozen is not a valid identifier, hence importlib.
sys.path.insert(0, str(Path(__file__).resolve().parent))
_frozen = importlib.import_module("03_generate_frozen")

# Data and outputs live under $TRDP_ROOT when set (the pipeline working
# tree on the cluster); otherwise they resolve relative to this repo.
PROJECT_ROOT = Path(os.environ.get("TRDP_ROOT",
                                   Path(__file__).resolve().parents[2]))
LORA_ROOT = PROJECT_ROOT / "output" / "lora_weights"
OUT_ROOT = PROJECT_ROOT / "output" / "lora_images"

MODEL_ID = _frozen.MODEL_ID
SIZE = _frozen.HEIGHT
STEPS = _frozen.NUM_INFERENCE_STEPS
PROGRESS_EVERY = _frozen.PROGRESS_EVERY


WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def resolve_weights(lora_path: Path | None, rank: int, checkpoint: int | None) -> Path:
    """Directory holding the LoRA weights.

    --lora-path wins when given; otherwise it is derived from --rank (and
    --checkpoint). Final dirs and checkpoint-N dirs both hold a file named
    WEIGHT_NAME, but with different key conventions inside: the final export
    uses diffusers keys (lora.down/lora.up) while checkpoints use peft keys
    (lora_A/lora_B). load_lora_weights converts peft keys on the way in, so
    both load correctly -- confirmed by 05_sample_checkpoints.py producing
    visibly different images per checkpoint.
    """
    if lora_path is not None:
        path = lora_path
    else:
        rank_dir = LORA_ROOT / f"rank{rank}"
        path = rank_dir / f"checkpoint-{checkpoint}" if checkpoint else rank_dir

    if not (path / WEIGHT_NAME).exists():
        print(f"ERROR: {WEIGHT_NAME} not found in {path}", file=sys.stderr)
        sys.exit(1)
    return path


def write_mapping(entries, out_dir: Path) -> int:
    """Index -> source crop + prompt, matching the frozen baseline's schema."""
    mapping = [
        {"image": f"{i:05d}.png", "prompt": e["text"], "original_file": e["file_name"]}
        for i, e in enumerate(entries)
    ]
    dst = out_dir / "prompt_mapping.json"
    tmp = dst.with_suffix(f".tmp{os.getpid()}")   # parallel slices must not tear it
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    tmp.replace(dst)
    return len(mapping)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank", type=int, required=True,
                    help="names the output subdir; also derives --lora-path if omitted")
    ap.add_argument("--lora-path", type=Path,
                    help="weights dir: a rank dir or a checkpoint-N dir")
    ap.add_argument("--checkpoint", type=int,
                    help="shorthand for --lora-path <rank dir>/checkpoint-N")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int)
    # metadata.jsonl is ordered test -> train -> train_extra, so a contiguous
    # head is all one split. A stride spreads the sample over every split and
    # city, and picks the same indices for every rank/checkpoint, keeping the
    # comparison paired.
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth prompt (1 = all)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA unavailable -- run on a GPU node.", file=sys.stderr)
        return 1

    weights = resolve_weights(args.lora_path, args.rank, args.checkpoint)
    # Tag the output dir when sampling anything other than the final weights,
    # so comparison runs never overwrite the rank's main image set.
    ckpt = args.checkpoint
    if ckpt is None and weights.name.startswith("checkpoint-"):
        ckpt = weights.name.split("-", 1)[1]
    tag = f"rank{args.rank}" + (f"_ckpt{ckpt}" if ckpt else "")
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = _frozen.load_entries()
    n_mapped = write_mapping(entries, out_dir)
    start = max(0, args.start)
    end = min(len(entries), args.end if args.end is not None else len(entries))
    if start >= end:
        print(f"ERROR: empty slice [{start}, {end}) of {len(entries)}", file=sys.stderr)
        return 1

    print(f"lora weights: {weights}", flush=True)
    print(f"output dir:   {out_dir}", flush=True)
    print(f"mapping:      {out_dir / 'prompt_mapping.json'} ({n_mapped} entries)", flush=True)
    print(f"{len(entries)} prompts; this job handles [{start}, {end}) "
          f"at {args.size}x{args.size}, {args.steps} steps", flush=True)

    wanted = list(range(start, end, max(1, args.stride)))
    todo = [i for i in wanted if not (out_dir / f"{i:05d}.png").exists()]
    print(f"{len(wanted)} in this slice (stride {args.stride}); "
          f"{len(wanted) - len(todo)} already generated, {len(todo)} to go", flush=True)
    if not todo:
        print("nothing to do -- slice already complete")
        return 0

    print(f"loading {MODEL_ID} + LoRA ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_lora_weights(str(weights), weight_name=WEIGHT_NAME)
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    generated = 0
    t0 = time.time()
    for i in todo:
        out_path = out_dir / f"{i:05d}.png"
        # Same seed formula as the frozen baseline: prompt index == seed.
        generator = torch.Generator(args.device).manual_seed(i)
        image = pipe(
            entries[i]["text"], generator=generator,
            num_inference_steps=args.steps, height=args.size, width=args.size,
        ).images[0]

        tmp = out_path.with_suffix(f".tmp{os.getpid()}")
        image.save(tmp, format="PNG")
        tmp.replace(out_path)

        generated += 1
        if generated % PROGRESS_EVERY == 0:
            rate = generated / (time.time() - t0)
            eta = (len(todo) - generated) / rate / 3600
            print(f"  {generated}/{len(todo)} generated "
                  f"({rate:.2f} img/s, ~{eta:.1f} h left in this slice)", flush=True)

    on_disk = len(list(out_dir.glob("[0-9]*.png")))
    print(f"\nGenerated {generated} images in {(time.time() - t0) / 3600:.2f} h")
    print(f"  {on_disk}/{len(entries)} images now in {out_dir}")
    if on_disk < len(entries):
        print("  incomplete -- resubmit to continue (existing images are skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
