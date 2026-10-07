"""Render the domain shift developing across a rank's LoRA checkpoints.

Loads every checkpoint-N saved during training, generates the same prompts with
the same seeds under each, and lays them out as a grid: one row per training
step, one column per prompt. The frozen SD 1.5 baseline is the top row.

Because prompt and seed are held fixed down each column, everything that
changes between rows is the adapter -- so the column reads as a time-lapse of
the fine-tune taking hold.

This works from the saved checkpoints, so it needs no change to training and
runs on ranks that already finished.
"""

import os
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# ---------------------------------------------------------------- constants --
# Data and outputs live under $TRDP_ROOT when set (the pipeline working
# tree on the cluster); otherwise they resolve relative to this repo.
PROJECT_ROOT = Path(os.environ.get("TRDP_ROOT",
                                   Path(__file__).resolve().parents[2]))
LORA_ROOT = PROJECT_ROOT / "output" / "lora_weights"
CAPTIONS_FILE = PROJECT_ROOT / "data" / "florence2_detailedcaptions" / "metadata.jsonl"
OUT_ROOT = PROJECT_ROOT / "output" / "checkpoint_grids"

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SIZE = 768
STEPS = 30
SEED = 1234

# Spread across the caption file, which is ordered test -> train -> train_extra.
DEFAULT_PROMPT_INDICES = (0, 5000, 15000)

LABEL_H = 34      # px reserved above each row of tiles
COL_LABEL_H = 58  # px reserved at the top for prompt captions


def find_checkpoints(rank_dir: Path):
    """[(step, dir)] for every checkpoint holding LoRA weights, in step order."""
    found = []
    for d in rank_dir.glob("checkpoint-*"):
        m = re.search(r"checkpoint-(\d+)", d.name)
        if m and (d / "pytorch_lora_weights.safetensors").exists():
            found.append((int(m.group(1)), d))
    return sorted(found)


def load_prompts(args):
    if args.prompt:
        return list(args.prompt)
    with CAPTIONS_FILE.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    idx = args.prompt_index or list(DEFAULT_PROMPT_INDICES)
    return [rows[i]["text"] for i in idx]


def make_grid(rows, row_labels, col_labels, tile=320):
    """Compose a labelled grid. rows[r][c] is a PIL image."""
    from PIL import ImageDraw, ImageFont

    def font(size):
        for p in ("/usr/share/fonts/dejavu/DejaVuSans.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if Path(p).exists():
                return ImageFont.truetype(p, size)
        return ImageFont.load_default()

    f_row, f_col = font(20), font(14)
    n_rows, n_cols = len(rows), len(rows[0])
    label_w = 190

    W = label_w + n_cols * tile
    H = COL_LABEL_H + n_rows * tile
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    for c, text in enumerate(col_labels):
        words, line, lines = text.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 46:
                lines.append(line); line = w
            else:
                line = f"{line} {w}".strip()
            if len(lines) == 3:
                break
        lines.append(line)
        draw.text((label_w + c * tile + 6, 4), "\n".join(lines[:3]),
                  fill="black", font=f_col)

    for r, row in enumerate(rows):
        y = COL_LABEL_H + r * tile
        draw.text((8, y + tile // 2 - 10), row_labels[r], fill="black", font=f_row)
        for c, img in enumerate(row):
            canvas.paste(img.resize((tile, tile), Image.LANCZOS),
                         (label_w + c * tile, y))
    return canvas


def generate(pipe, prompts, size, steps, seed, device):
    """One image per prompt; seed is fixed per column across all rows."""
    out = []
    for i, prompt in enumerate(prompts):
        g = torch.Generator(device).manual_seed(seed + i)
        out.append(pipe(prompt, generator=g, num_inference_steps=steps,
                        height=size, width=size).images[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--prompt", action="append", help="repeatable; overrides defaults")
    ap.add_argument("--prompt-index", type=int, action="append",
                    help="repeatable; row index into metadata.jsonl")
    ap.add_argument("--every", type=int, default=1,
                    help="use every Nth checkpoint (1 = all)")
    ap.add_argument("--no-frozen", action="store_true", help="skip the baseline row")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tile", type=int, default=320, help="grid tile size in px")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA unavailable -- run on a GPU node.", file=sys.stderr)
        return 1

    rank_dir = LORA_ROOT / f"rank{args.rank}"
    if not rank_dir.is_dir():
        print(f"ERROR: no such rank dir: {rank_dir}", file=sys.stderr)
        return 1

    checkpoints = find_checkpoints(rank_dir)[:: args.every]
    final = rank_dir / "pytorch_lora_weights.safetensors"
    if not checkpoints and not final.exists():
        print(f"ERROR: no checkpoints or final weights under {rank_dir}", file=sys.stderr)
        return 1

    prompts = load_prompts(args)
    out_dir = OUT_ROOT / f"rank{args.rank}"
    (out_dir / "tiles").mkdir(parents=True, exist_ok=True)

    print(f"rank {args.rank}: {len(checkpoints)} checkpoints, {len(prompts)} prompts",
          flush=True)
    for i, p in enumerate(prompts):
        print(f"  [{i}] seed={args.seed + i}  {p[:88]}", flush=True)

    print(f"loading {MODEL_ID} ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    rows, labels = [], []
    t0 = time.time()

    if not args.no_frozen:
        pipe.unload_lora_weights()
        rows.append(generate(pipe, prompts, args.size, args.steps, args.seed, args.device))
        labels.append("frozen SD 1.5")
        print(f"  frozen baseline done ({time.time() - t0:.0f}s)", flush=True)

    stages = [(f"step {s}", d) for s, d in checkpoints]
    if final.exists():
        stages.append(("final", rank_dir))

    for label, path in stages:
        pipe.unload_lora_weights()          # never stack adapters
        pipe.load_lora_weights(str(path))
        rows.append(generate(pipe, prompts, args.size, args.steps, args.seed, args.device))
        labels.append(label)
        print(f"  {label} done ({time.time() - t0:.0f}s)", flush=True)

    for r, label in enumerate(labels):
        tag = label.replace(" ", "_")
        for c, img in enumerate(rows[r]):
            img.save(out_dir / "tiles" / f"{tag}__prompt{c}.png")

    grid_path = out_dir / f"rank{args.rank}_checkpoint_grid.png"
    make_grid(rows, labels, prompts, tile=args.tile).save(grid_path)

    with (out_dir / "prompts.json").open("w", encoding="utf-8") as f:
        json.dump({"rank": args.rank, "seed_base": args.seed, "size": args.size,
                   "steps": args.steps, "rows": labels, "prompts": prompts}, f, indent=2)

    print(f"\ngrid:  {grid_path}")
    print(f"tiles: {out_dir / 'tiles'}  ({len(labels) * len(prompts)} images)")
    print(f"took {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
