"""Cluster entry point for the Cityscapes LoRA experiments.

The completed SD 1.5 sweep lives in outputs/rank_*/seed_* and
outputs/generated/rank_*/seed_*. GPU operations run through Slurm.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

RANKS = (8, 16, 32, 64, 128)
SEEDS = (1, 2, 3, 4, 5)
ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "cityscapes_cropped"
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def entries() -> list[dict]:
    with (DATASET / "metadata.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("No captions found in cityscapes_cropped/metadata.jsonl")
    return rows


def experiment_paths(rank: int, seed: int) -> tuple[Path, Path, Path]:
    if rank not in RANKS or seed not in SEEDS:
        raise ValueError(f"Expected rank in {RANKS} and seed in {SEEDS}")
    return (
        ROOT / "outputs" / f"rank_{rank}" / f"seed_{seed}",
        ROOT / "outputs" / "generated" / f"rank_{rank}" / f"seed_{seed}",
        ROOT / "outputs" / "logs" / f"rank_{rank}" / f"seed_{seed}",
    )


def generation_plan(rank: int | None, seed: int | None, frozen: bool) -> tuple[Path | None, Path, int]:
    """Return weights, output directory, and first image seed for one run."""
    if frozen:
        if rank is not None:
            raise ValueError("Frozen generation has no LoRA rank")
        if seed is None:
            # Preserve the historical unseeded frozen baseline and its image seeds.
            return None, ROOT / "outputs" / "cityscapes-gen-frozen", 0
        if seed not in SEEDS:
            raise ValueError(f"Expected frozen seed in {SEEDS}")
        return None, ROOT / "outputs" / "generated" / "frozen" / f"seed_{seed}", 10000 * seed
    if rank is None or seed is None:
        raise ValueError("LoRA generation requires rank and seed")
    weights, generated, _ = experiment_paths(rank, seed)
    return weights, generated, 10000


def train(args: argparse.Namespace) -> None:
    weights_dir, _, log_dir = experiment_paths(args.rank, args.seed)
    first = entries()[0]
    if not (DATASET / first["file_name"]).is_file():
        raise FileNotFoundError(DATASET / first["file_name"])
    script = ROOT / "train_text_to_image_lora.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing vendored diffusers script: {script}")
    command = [
        "accelerate", "launch", str(script),
        "--pretrained_model_name_or_path", BASE_MODEL,
        "--train_data_dir", str(DATASET), "--caption_column", "text",
        "--resolution", "512", "--train_batch_size", "2",
        "--gradient_accumulation_steps", "2", "--max_train_steps", "5000",
        "--learning_rate", "1e-4", "--lr_scheduler", "cosine",
        "--lr_warmup_steps", "200", "--snr_gamma", "5.0",
        "--rank", str(args.rank), "--output_dir", str(weights_dir),
        "--checkpointing_steps", "400", "--seed", str(args.seed),
        "--logging_dir", str(log_dir), "--mixed_precision", "fp16",
        "--gradient_checkpointing", "--dataloader_num_workers", "0",
    ]
    print(" ".join(command), flush=True)
    if args.dry_run:
        return
    weights_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, check=True)


def generate_images(
    *, rank: int | None = None, seed: int | None = None,
    frozen: bool = False, dry_run: bool = False,
) -> Path:
    """Generate one complete run and return its output directory."""
    weights_dir, output_dir, seed_base = generation_plan(rank, seed, frozen)
    rows = entries()
    if dry_run:
        print(f"model: {BASE_MODEL}")
        print(f"weights: {weights_dir or 'frozen base model'}")
        print(f"output: {output_dir}")
        print(f"images: {len(rows)}")
        print(f"generation seeds: {seed_base}..{seed_base + len(rows) - 1}")
        return output_dir

    import torch
    from diffusers import StableDiffusionPipeline
    from tqdm import tqdm

    if weights_dir is not None:
        if not (weights_dir / "pytorch_lora_weights.safetensors").is_file():
            raise FileNotFoundError(f"LoRA weights missing in {weights_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_options = {"torch_dtype": torch.float16}
    if frozen:
        pipeline_options["variant"] = "fp16"
    else:
        pipeline_options.update(safety_checker=None, feature_extractor=None)
    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, **pipeline_options).to("cuda")
    pipe.enable_attention_slicing()
    if weights_dir is not None:
        pipe.load_lora_weights(str(weights_dir))
    mapping = []
    for index, row in enumerate(tqdm(rows, desc="Generating")):
        name = f"{index:05d}.png"
        generation_seed = seed_base + index
        mapping_row = {
            "image": name, "prompt": row["text"],
            "original_file": row.get("original_file", row["file_name"]),
        }
        if frozen and seed is not None:
            mapping_row.update(frozen_seed=seed, generation_seed=generation_seed)
        elif not frozen:
            mapping_row.update(rank=rank, training_seed=seed,
                               generation_seed=generation_seed)
        mapping.append(mapping_row)
        path = output_dir / name
        if path.exists():
            continue
        generator = torch.Generator(device="cuda").manual_seed(generation_seed)
        image = pipe(row["text"], generator=generator, num_inference_steps=30).images[0]
        image.save(path)
    write_json(output_dir / "prompt_mapping.json", mapping)
    print(f"Generated images and mapping: {output_dir}")
    return output_dir


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    temporary.replace(path)


def image_paths(folder: Path, count: int) -> list[str]:
    paths = [str(folder / f"{index:05d}.png") for index in range(count)]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} generated images missing in {folder}; first: {missing[0]}")
    return paths


def evaluate(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from cleanfid import fid
    from cleanfid.features import build_feature_extractor
    from torchmetrics.multimodal import CLIPScore
    from torchvision import transforms
    from transformers import BlipForImageTextRetrieval, BlipProcessor
    import lpips
    from tqdm import tqdm

    rows = entries()
    count = len(rows) if args.limit is None else min(args.limit, len(rows))
    real = [str(DATASET / row["file_name"]) for row in rows[:count]]
    if any(not Path(path).is_file() for path in real):
        raise FileNotFoundError("One or more real images in metadata are missing")
    selected = [(rank, seed) for rank in args.ranks for seed in args.seeds]
    generated = {(rank, seed): image_paths(experiment_paths(rank, seed)[1], count)
                 for rank, seed in selected}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_root = ROOT / "outputs-metrics" / (
        "sd15-seeds" if args.limit is None else f"sd15-validation-{count}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    extractor = build_feature_extractor(mode="clean", device=device)
    cache = output_root / f"real_fid_{count}.npz"
    if cache.exists():
        stats = np.load(cache)
        mu_real, sigma_real = stats["mu"], stats["sigma"]
    else:
        features = fid.get_files_features(real, model=extractor, num_workers=0, mode="clean")
        mu_real, sigma_real = np.mean(features, axis=0), np.cov(features, rowvar=False)
        np.savez(cache, mu=mu_real, sigma=sigma_real)

    clip = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    blip = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(device).eval()
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    to_tensor = transforms.ToTensor()
    lpips_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    all_results = []
    for rank, seed in selected:
        paths = generated[(rank, seed)]
        folder = experiment_paths(rank, seed)[1]
        print(f"Evaluating rank={rank} seed={seed}", flush=True)
        features = fid.get_files_features(paths, model=extractor, num_workers=0, mode="clean")
        fid_score = float(fid.frechet_distance(
            mu_real, sigma_real, np.mean(features, axis=0), np.cov(features, rowvar=False)))
        # KID uses all images in both directories; require the full dataset for alignment.
        kid_score = float(fid.compute_kid(str(folder), str(DATASET / "images"), mode="clean")) if count == len(rows) else None
        clip_scores, itm_scores, cos_scores = [], [], []
        for index, path in enumerate(tqdm(paths, desc=f"rank {rank} seed {seed}")):
            with Image.open(path) as source:
                image = source.convert("RGB")
                tensor = (to_tensor(image) * 255).to(torch.uint8).unsqueeze(0).to(device)
                with torch.no_grad():
                    clip_scores.append(float(clip(tensor, [rows[index]["text"]]).item()))
                    clip.reset()
                    inputs = blip_processor(image, rows[index]["text"], return_tensors="pt").to(device)
                    logits = blip(**inputs)[0]
                    itm_scores.append(float(torch.softmax(logits, dim=1)[0, 1].item()))
                    cos_scores.append(float(blip(**inputs, use_itm_head=False)[0].item()))
        rng = random.Random(42)
        sample = rng.sample(paths, min(200, len(paths)))
        tensors = []
        for path in sample:
            with Image.open(path) as source:
                tensors.append(lpips_transform(source.convert("RGB")))
        pairs = [(i, j) for i in range(len(tensors)) for j in range(i + 1, len(tensors))]
        selected_pairs = rng.sample(pairs, min(500, len(pairs)))
        diversity = []
        with torch.no_grad():
            for i, j in tqdm(selected_pairs, desc="LPIPS"):
                diversity.append(float(lpips_model(
                    tensors[i].unsqueeze(0).to(device), tensors[j].unsqueeze(0).to(device)).item()))
        result = {
            "rank": rank, "training_seed": seed, "num_samples": count,
            "fid": fid_score, "kid": kid_score,
            "clip_score": float(np.mean(clip_scores)),
            "blip_itm": float(np.mean(itm_scores)),
            "blip_cosine": float(np.mean(cos_scores)),
            "lpips_mean": float(np.mean(diversity)) if diversity else None,
            "lpips_std": float(np.std(diversity)) if diversity else None,
        }
        write_json(output_root / f"rank_{rank}" / f"seed_{seed}" / "metrics.json", result)
        all_results.append(result)
    if len(selected) == 1:
        print(f"Saved metrics for rank={selected[0][0]} seed={selected[0][1]}")
    else:
        write_json(output_root / "selected_metrics.json", all_results)
        summarize(output_root)


def summarize(output_root: Path) -> None:
    import statistics

    summary = []
    for rank in RANKS:
        results = []
        for seed in SEEDS:
            path = output_root / f"rank_{rank}" / f"seed_{seed}" / "metrics.json"
            if path.exists():
                results.append(json.loads(path.read_text(encoding="utf-8")))
        if not results:
            continue
        metrics = {}
        for key in ("fid", "kid", "clip_score", "blip_itm", "blip_cosine", "lpips_mean", "lpips_std"):
            values = [row[key] for row in results if row.get(key) is not None]
            if values:
                metrics[key] = {"mean": statistics.mean(values),
                                "std": statistics.stdev(values) if len(values) > 1 else None}
        summary.append({"rank": rank, "seeds": [row["training_seed"] for row in results],
                        "num_seeds": len(results), "complete": len(results) == len(SEEDS),
                        "metrics": metrics})
    if not summary:
        raise FileNotFoundError(f"No per-seed metrics found in {output_root}")
    write_json(output_root / "rank_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help="Train one SD 1.5 LoRA")
    training.add_argument("--rank", type=int, choices=RANKS, required=True)
    training.add_argument("--seed", type=int, choices=SEEDS, required=True)
    training.add_argument("--dry-run", action="store_true")
    generation = sub.add_parser("generate", help="Generate from one LoRA or the frozen base")
    generation.add_argument("--rank", type=int, choices=RANKS)
    generation.add_argument("--seed", type=int, choices=SEEDS)
    generation.add_argument("--frozen", action="store_true")
    generation.add_argument("--dry-run", action="store_true")
    evaluation = sub.add_parser("evaluate", help="Evaluate selected rank/seed runs")
    evaluation.add_argument("--ranks", type=int, nargs="+", choices=RANKS, default=RANKS)
    evaluation.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    evaluation.add_argument("--limit", type=int, help="Small validation run; KID omitted")
    sub.add_parser("summarize", help="Aggregate saved metrics by rank")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        if args.frozen and args.rank is not None:
            parser.error("--frozen cannot be combined with --rank")
        if not args.frozen and (args.rank is None or args.seed is None):
            parser.error("LoRA generation requires --rank and --seed")
        generate_images(rank=args.rank, seed=args.seed, frozen=args.frozen,
                        dry_run=args.dry_run)
    elif args.command == "evaluate":
        if args.limit is not None and args.limit < 2:
            parser.error("--limit must be at least 2")
        evaluate(args)
    else:
        summarize(ROOT / "outputs-metrics" / "sd15-seeds")


if __name__ == "__main__":
    main()
