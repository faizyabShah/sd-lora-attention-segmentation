"""Cluster entry point for the Cityscapes LoRA experiments.

The completed SD 1.5 sweep lives in outputs/lora/rank_*/seed_* and
outputs/lora/generated/rank_*/seed_*. GPU operations run through Slurm.
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
DATASET = ROOT / "datasets" / "cityscapes_cropped"
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def entries() -> list[dict]:
    with (DATASET / "metadata.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("No captions found in datasets/cityscapes_cropped/metadata.jsonl")
    return rows


def experiment_paths(rank: int, seed: int) -> tuple[Path, Path, Path]:
    if rank not in RANKS or seed not in SEEDS:
        raise ValueError(f"Expected rank in {RANKS} and seed in {SEEDS}")
    return (
        ROOT / "outputs" / "lora" / f"rank_{rank}" / f"seed_{seed}",
        ROOT / "outputs" / "lora" / "generated" / f"rank_{rank}" / f"seed_{seed}",
        ROOT / "outputs" / "lora" / "logs" / f"rank_{rank}" / f"seed_{seed}",
    )


def generation_plan(rank: int | None, seed: int | None, frozen: bool) -> tuple[Path | None, Path, int]:
    """Return weights, output directory, and first image seed for one run."""
    if frozen:
        if rank is not None:
            raise ValueError("Frozen generation has no LoRA rank")
        if seed is None:
            # Preserve the historical unseeded frozen baseline and its image seeds.
            return None, ROOT / "outputs" / "lora" / "cityscapes-gen-frozen", 0
        if seed not in SEEDS:
            raise ValueError(f"Expected frozen seed in {SEEDS}")
        return None, ROOT / "outputs" / "lora" / "generated" / "frozen" / f"seed_{seed}", 10000 * seed
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
    frozen: bool = False, checkpoint: int | None = None,
    caption_stride: int = 1, caption_offset: int = 0,
    num_images: int | None = None, subset_name: str | None = None,
    output_index_offset: int = 0,
    dry_run: bool = False,
) -> Path:
    """Generate one complete run and return its output directory."""
    weights_dir, output_dir, seed_base = generation_plan(rank, seed, frozen)
    rows = entries()
    if checkpoint is not None:
        if frozen:
            raise ValueError("The frozen model has no training checkpoint")
        weights_dir = weights_dir / f"checkpoint-{checkpoint}"
        output_dir = (
            ROOT / "outputs" / "lora" / "generated" / "checkpoints"
            / f"rank_{rank}" / f"seed_{seed}" / f"checkpoint_{checkpoint}"
        )
    if caption_stride < 1:
        raise ValueError("caption_stride must be at least 1")
    if caption_offset < 0 or caption_offset >= caption_stride:
        raise ValueError("caption_offset must be between 0 and caption_stride - 1")
    if output_index_offset < 0:
        raise ValueError("output_index_offset must be nonnegative")
    selected = list(enumerate(rows))[caption_offset::caption_stride]
    if num_images is not None:
        if num_images < 1:
            raise ValueError("num_images must be at least 1")
        selected = selected[:num_images]
    if caption_stride != 1 or num_images is not None:
        resolved_subset_name = subset_name or f"every_{caption_stride}_count_{len(selected)}"
        if caption_offset and subset_name is None:
            resolved_subset_name = f"every_{caption_stride}_offset_{caption_offset}_count_{len(selected)}"
        if checkpoint is None:
            family = "frozen" if frozen else f"rank_{rank}"
            output_dir = ROOT / "outputs" / "lora" / "generated" / "subsets" / family / f"seed_{seed}" / resolved_subset_name
        else:
            output_dir = output_dir / resolved_subset_name
    if weights_dir is not None and not (weights_dir / "pytorch_lora_weights.safetensors").is_file():
        raise FileNotFoundError(f"LoRA weights missing in {weights_dir}")
    if dry_run:
        print(f"model: {BASE_MODEL}")
        print(f"weights: {weights_dir or 'frozen base model'}")
        print(f"output: {output_dir}")
        print(f"checkpoint: {checkpoint if checkpoint is not None else 'final/base'}")
        print(f"images: {len(selected)}")
        print(f"caption indices: {selected[0][0]}..{selected[-1][0]} "
              f"(stride {caption_stride}, offset {caption_offset})")
        print(f"generation seeds: {seed_base + selected[0][0]}..{seed_base + selected[-1][0]}")
        print(f"output image indices: {output_index_offset}..{output_index_offset + len(selected) - 1}")
        return output_dir

    import torch
    from diffusers import StableDiffusionPipeline
    from tqdm import tqdm

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
    for output_index, (caption_index, row) in enumerate(tqdm(selected, desc="Generating")):
        name = f"{output_index_offset + output_index:05d}.png"
        generation_seed = seed_base + caption_index
        mapping_row = {
            "image": name, "prompt": row["text"],
            "original_file": row.get("original_file", row["file_name"]),
            "caption_index": caption_index,
            "checkpoint": checkpoint,
        }
        if frozen and seed is not None:
            mapping_row.update(frozen_seed=seed, generation_seed=generation_seed)
        elif not frozen:
            mapping_row.update(rank=rank, training_seed=seed,
                               generation_seed=generation_seed)
        mapping.append(mapping_row)
        path = output_dir / name
        if path.exists() and path.stat().st_size > 0:
            continue
        generator = torch.Generator(device="cuda").manual_seed(generation_seed)
        image = pipe(row["text"], generator=generator, num_inference_steps=30).images[0]
        temporary = path.with_suffix(".tmp.png")
        image.save(temporary)
        temporary.replace(path)
    mapping_path = output_dir / "prompt_mapping.json"
    if mapping_path.exists():
        previous = json.loads(mapping_path.read_text(encoding="utf-8"))
        by_image = {item["image"]: item for item in previous}
        by_image.update({item["image"]: item for item in mapping})
        mapping = [by_image[name] for name in sorted(by_image)]
    write_json(mapping_path, mapping)
    print(f"Generated images and mapping: {output_dir}")
    return output_dir


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    temporary.replace(path)


def image_paths(folder: Path, count: int) -> list[str]:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    available = {
        path.name for path in folder.iterdir()
        if path.suffix.lower() == ".png" and path.stat().st_size > 0
    }
    expected = [f"{index:05d}.png" for index in range(count)]
    missing = [name for name in expected if name not in available]
    if missing:
        raise FileNotFoundError(f"{len(missing)} generated images missing in {folder}; first: {missing[0]}")
    return [str(folder / name) for name in expected]


def metric_image_paths(folder: Path, count: int) -> tuple[list[str], list[int]]:
    """Return readable generated images and their dataset indices, skipping damage."""
    from PIL import Image, UnidentifiedImageError

    paths, indices = [], []
    for index in range(count):
        path = folder / f"{index:05d}.png"
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise OSError("missing or empty file")
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            print(f"WARNING: skipping unreadable image {path}: {error}", flush=True)
            continue
        paths.append(str(path))
        indices.append(index)
    if len(paths) < 2:
        raise ValueError(f"Fewer than two readable images in {folder}")
    return paths, indices


def mapped_metric_images(folder: Path, expected: int = 1000) -> tuple[list[str], list[int]]:
    """Load readable generated images and caption indices from their mapping."""
    from PIL import Image, UnidentifiedImageError

    mapping_path = folder / "prompt_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)
    mapping = sorted(json.loads(mapping_path.read_text(encoding="utf-8")), key=lambda row: row["image"])
    if len(mapping) != expected:
        raise ValueError(f"Expected {expected} mapping rows in {folder}, found {len(mapping)}")
    paths, indices = [], []
    for row in mapping:
        path = folder / row["image"]
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise OSError("missing or empty file")
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            print(f"WARNING: skipping unreadable image {path}: {error}", flush=True)
            continue
        paths.append(str(path))
        indices.append(int(row["caption_index"]))
    if len(paths) < 2:
        raise ValueError(f"Fewer than two readable images in {folder}")
    return paths, indices


def prepare_final_checkpoint_subsets() -> None:
    """Link the final step-5000 images matching the checkpoint evaluation subset."""
    rows = entries()
    caption_indices = list(range(0, len(rows), 10))[:500] + list(range(1, len(rows), 10))[:500]
    subset = "every_10_offsets_0_1_count_1000"
    for rank in RANKS:
        for seed in SEEDS:
            source = ROOT / "outputs" / "lora" / "generated" / f"rank_{rank}" / f"seed_{seed}"
            destination = (ROOT / "outputs" / "lora" / "generated" / "checkpoints" / f"rank_{rank}"
                           / f"seed_{seed}" / "checkpoint_5000" / subset)
            destination.mkdir(parents=True, exist_ok=True)
            mapping = []
            for output_index, caption_index in enumerate(caption_indices):
                source_image = source / f"{caption_index:05d}.png"
                if not source_image.is_file() or source_image.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing final generated image: {source_image}")
                target = destination / f"{output_index:05d}.png"
                if target.is_symlink() and target.resolve() != source_image.resolve():
                    target.unlink()
                if not target.exists():
                    target.symlink_to(source_image.resolve())
                row = rows[caption_index]
                mapping.append({
                    "image": target.name, "prompt": row["text"],
                    "original_file": row.get("original_file", row["file_name"]),
                    "caption_index": caption_index, "checkpoint": 5000,
                    "rank": rank, "training_seed": seed,
                    "generation_seed": 10000 + caption_index,
                })
            write_json(destination / "prompt_mapping.json", mapping)
            print(f"Prepared rank={rank} seed={seed}: {destination}")


def aligned_kid(generated: list[str], real: list[str]) -> float:
    """Compute KID from aligned readable subsets without exposing corrupt files."""
    import tempfile
    from cleanfid import fid

    with tempfile.TemporaryDirectory(prefix="cityscapes_kid_") as temporary:
        root = Path(temporary)
        generated_dir, real_dir = root / "generated", root / "real"
        generated_dir.mkdir()
        real_dir.mkdir()
        for index, (generated_path, real_path) in enumerate(zip(generated, real)):
            (generated_dir / f"{index:05d}.png").symlink_to(Path(generated_path).resolve())
            (real_dir / f"{index:05d}{Path(real_path).suffix}").symlink_to(Path(real_path).resolve())
        return float(fid.compute_kid(str(generated_dir), str(real_dir), mode="clean"))


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
    output_root = ROOT / "outputs" / "lora" / "outputs-metrics" / (
        "sd15-seeds" if args.limit is None else f"sd15-validation-{count}"
    )
    selected = []
    for seed in args.frozen_seeds:
        folder = ROOT / "outputs" / "lora" / "generated" / "frozen" / f"seed_{seed}"
        selected.append(("frozen", seed, folder, output_root / "frozen" / f"seed_{seed}" / "metrics.json"))
    for rank in args.ranks:
        for seed in args.seeds:
            folder = experiment_paths(rank, seed)[1]
            selected.append((f"rank_{rank}", seed, folder,
                             output_root / f"rank_{rank}" / f"seed_{seed}" / "metrics.json"))
    pending = [run for run in selected if not (args.skip_existing and run[3].is_file())]
    if not pending:
        print("All selected metrics already exist")
        summarize(output_root)
        return
    generated = {(label, seed): metric_image_paths(folder, count)
                 for label, seed, folder, _ in pending}
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    for label, seed, folder, result_path in pending:
        paths, indices = generated[(label, seed)]
        aligned_real = [real[index] for index in indices]
        print(f"Evaluating {label} seed={seed}", flush=True)
        features = fid.get_files_features(paths, model=extractor, num_workers=0, mode="clean")
        if len(indices) == count:
            run_mu_real, run_sigma_real = mu_real, sigma_real
        else:
            real_features = fid.get_files_features(
                aligned_real, model=extractor, num_workers=0, mode="clean")
            run_mu_real = np.mean(real_features, axis=0)
            run_sigma_real = np.cov(real_features, rowvar=False)
        fid_score = float(fid.frechet_distance(
            run_mu_real, run_sigma_real, np.mean(features, axis=0), np.cov(features, rowvar=False)))
        # KID uses all images in both directories; require the full dataset for alignment.
        kid_score = aligned_kid(paths, aligned_real) if count == len(rows) else None
        clip_scores, itm_scores, cos_scores = [], [], []
        for path, dataset_index in tqdm(
            zip(paths, indices), total=len(paths), desc=f"{label} seed {seed}"
        ):
            with Image.open(path) as source:
                image = source.convert("RGB")
                tensor = (to_tensor(image) * 255).to(torch.uint8).unsqueeze(0).to(device)
                with torch.no_grad():
                    text = rows[dataset_index]["text"]
                    clip_scores.append(float(clip(tensor, [text]).item()))
                    clip.reset()
                    inputs = blip_processor(image, text, return_tensors="pt").to(device)
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
            "model": "frozen" if label == "frozen" else "lora",
            "rank": None if label == "frozen" else int(label.removeprefix("rank_")),
            "seed": seed, "num_samples": len(paths),
            "skipped_images": count - len(paths),
            "fid": fid_score, "kid": kid_score,
            "clip_score": float(np.mean(clip_scores)),
            "blip_itm": float(np.mean(itm_scores)),
            "blip_cosine": float(np.mean(cos_scores)),
            "lpips_mean": float(np.mean(diversity)) if diversity else None,
            "lpips_std": float(np.std(diversity)) if diversity else None,
        }
        write_json(result_path, result)
        all_results.append(result)
    if len(selected) == 1:
        print(f"Saved metrics for {selected[0][0]} seed={selected[0][1]}")
    else:
        write_json(output_root / "last_batch_metrics.json", all_results)
        summarize(output_root)


def evaluate_checkpoint_subsets(args: argparse.Namespace) -> None:
    """Evaluate frozen and LoRA checkpoint runs on the combined 1,000 images."""
    import statistics
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
    subset = "every_10_offsets_0_1_count_1000"
    output_root = ROOT / "outputs" / "lora" / "outputs-metrics" / "sd15-checkpoints"
    runs = []
    for seed in SEEDS:
        folder = ROOT / "outputs" / "lora" / "generated" / "subsets" / "frozen" / f"seed_{seed}" / subset
        runs.append(("frozen", None, seed, folder,
                     output_root / "frozen" / f"seed_{seed}" / "metrics.json"))
    for rank in RANKS:
        for checkpoint in args.checkpoints:
            for seed in SEEDS:
                folder = (ROOT / "outputs" / "lora" / "generated" / "checkpoints" / f"rank_{rank}"
                          / f"seed_{seed}" / f"checkpoint_{checkpoint}" / subset)
                result_path = (output_root / f"rank_{rank}" / f"checkpoint_{checkpoint}"
                               / f"seed_{seed}" / "metrics.json")
                runs.append((f"rank_{rank}", checkpoint, seed, folder, result_path))
    pending = [run for run in runs if not (args.skip_existing and run[4].is_file())]
    if not pending:
        print("All checkpoint metrics already exist")
        return
    prepared = {(label, checkpoint, seed): mapped_metric_images(folder)
                for label, checkpoint, seed, folder, _ in pending}
    reference_indices = next(iter(prepared.values()))[1]
    reference_real = [str(DATASET / rows[index]["file_name"]) for index in reference_indices]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_root.mkdir(parents=True, exist_ok=True)
    extractor = build_feature_extractor(mode="clean", device=device)
    cache = output_root / "real_fid_1000.npz"
    if cache.exists():
        stats = np.load(cache)
        mu_real, sigma_real = stats["mu"], stats["sigma"]
    else:
        features = fid.get_files_features(reference_real, model=extractor, num_workers=0, mode="clean")
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
    for label, checkpoint, seed, _, result_path in pending:
        paths, indices = prepared[(label, checkpoint, seed)]
        real_paths = [str(DATASET / rows[index]["file_name"]) for index in indices]
        print(f"Evaluating {label} checkpoint={checkpoint} seed={seed}", flush=True)
        features = fid.get_files_features(paths, model=extractor, num_workers=0, mode="clean")
        if indices == reference_indices:
            run_mu, run_sigma = mu_real, sigma_real
        else:
            real_features = fid.get_files_features(real_paths, model=extractor, num_workers=0, mode="clean")
            run_mu, run_sigma = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        fid_score = float(fid.frechet_distance(
            run_mu, run_sigma, np.mean(features, axis=0), np.cov(features, rowvar=False)))
        kid_score = aligned_kid(paths, real_paths)
        clip_scores, itm_scores, cos_scores = [], [], []
        for path, dataset_index in tqdm(zip(paths, indices), total=len(paths), desc=label):
            with Image.open(path) as source:
                image = source.convert("RGB")
                tensor = (to_tensor(image) * 255).to(torch.uint8).unsqueeze(0).to(device)
                text = rows[dataset_index]["text"]
                with torch.no_grad():
                    clip_scores.append(float(clip(tensor, [text]).item()))
                    clip.reset()
                    inputs = blip_processor(image, text, return_tensors="pt").to(device)
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
        diversity = []
        with torch.no_grad():
            for i, j in tqdm(rng.sample(pairs, min(500, len(pairs))), desc="LPIPS"):
                diversity.append(float(lpips_model(
                    tensors[i].unsqueeze(0).to(device), tensors[j].unsqueeze(0).to(device)).item()))
        write_json(result_path, {
            "model": "frozen" if label == "frozen" else "lora",
            "rank": None if label == "frozen" else int(label.removeprefix("rank_")),
            "checkpoint": checkpoint, "seed": seed, "num_samples": len(paths),
            "skipped_images": 1000 - len(paths), "fid": fid_score, "kid": kid_score,
            "clip_score": float(np.mean(clip_scores)), "blip_itm": float(np.mean(itm_scores)),
            "blip_cosine": float(np.mean(cos_scores)),
            "lpips_mean": float(np.mean(diversity)), "lpips_std": float(np.std(diversity)),
        })
    results = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(output_root.glob("**/metrics.json"))]
    write_json(output_root / "all_metrics.json", results)
    groups = {}
    for result in results:
        groups.setdefault((result["rank"], result["checkpoint"]), []).append(result)
    keys = ("fid", "kid", "clip_score", "blip_itm", "blip_cosine", "lpips_mean", "lpips_std")
    summary = []
    for (rank, checkpoint), group in groups.items():
        summary.append({
            "model": "frozen" if rank is None else "lora", "rank": rank,
            "checkpoint": checkpoint, "seeds": sorted(row["seed"] for row in group),
            "num_seeds": len(group),
            "metrics": {key: {"mean": statistics.mean(row[key] for row in group),
                              "std": statistics.stdev(row[key] for row in group) if len(group) > 1 else None}
                        for key in keys},
        })
    summary.sort(key=lambda row: (-1 if row["rank"] is None else row["rank"],
                                  -1 if row["checkpoint"] is None else row["checkpoint"]))
    write_json(output_root / "checkpoint_summary.json", summary)


def summarize(output_root: Path) -> None:
    import statistics

    summary = []
    all_results = []
    for label in ("frozen", *(f"rank_{rank}" for rank in RANKS)):
        results = []
        for seed in SEEDS:
            path = output_root / label / f"seed_{seed}" / "metrics.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                results.append(result)
                all_results.append(result)
        if not results:
            continue
        metrics = {}
        for key in ("fid", "kid", "clip_score", "blip_itm", "blip_cosine", "lpips_mean", "lpips_std"):
            values = [row[key] for row in results if row.get(key) is not None]
            if values:
                metrics[key] = {"mean": statistics.mean(values),
                                "std": statistics.stdev(values) if len(values) > 1 else None}
        summary.append({"model": "frozen" if label == "frozen" else "lora",
                        "rank": None if label == "frozen" else int(label.removeprefix("rank_")),
                        "seeds": [row.get("seed", row.get("training_seed")) for row in results],
                        "num_seeds": len(results), "complete": len(results) == len(SEEDS),
                        "metrics": metrics})
    if not summary:
        raise FileNotFoundError(f"No per-seed metrics found in {output_root}")
    write_json(output_root / "rank_summary.json", summary)
    write_json(output_root / "all_metrics.json", all_results)


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
    generation.add_argument("--checkpoint", type=int)
    generation.add_argument("--caption-stride", type=int, default=1)
    generation.add_argument("--caption-offset", type=int, default=0)
    generation.add_argument("--num-images", type=int)
    generation.add_argument("--subset-name")
    generation.add_argument("--output-index-offset", type=int, default=0)
    generation.add_argument("--dry-run", action="store_true")
    evaluation = sub.add_parser("evaluate", help="Evaluate selected rank/seed runs")
    evaluation.add_argument("--ranks", type=int, nargs="+", choices=RANKS, default=RANKS)
    evaluation.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    evaluation.add_argument("--frozen-seeds", type=int, nargs="+", choices=SEEDS, default=[])
    evaluation.add_argument("--skip-existing", action="store_true")
    evaluation.add_argument("--limit", type=int, help="Small validation run; KID omitted")
    checkpoint_evaluation = sub.add_parser(
        "evaluate-checkpoints", help="Evaluate combined 1,000-image checkpoint subsets")
    checkpoint_evaluation.add_argument(
        "--checkpoints", type=int, nargs="+", default=[1200, 2400, 3600])
    checkpoint_evaluation.add_argument("--skip-existing", action="store_true")
    sub.add_parser(
        "prepare-final-subsets", help="Prepare matched 1,000-image views for step 5000")
    sub.add_parser("summarize", help="Aggregate saved metrics by rank")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        if args.frozen and args.rank is not None:
            parser.error("--frozen cannot be combined with --rank")
        if args.frozen and args.checkpoint is not None:
            parser.error("--frozen cannot be combined with --checkpoint")
        if not args.frozen and (args.rank is None or args.seed is None):
            parser.error("LoRA generation requires --rank and --seed")
        generate_images(rank=args.rank, seed=args.seed, frozen=args.frozen,
                        checkpoint=args.checkpoint, caption_stride=args.caption_stride,
                        caption_offset=args.caption_offset,
                        num_images=args.num_images,
                        subset_name=args.subset_name,
                        output_index_offset=args.output_index_offset,
                        dry_run=args.dry_run)
    elif args.command == "evaluate":
        if args.limit is not None and args.limit < 2:
            parser.error("--limit must be at least 2")
        evaluate(args)
    elif args.command == "evaluate-checkpoints":
        evaluate_checkpoint_subsets(args)
    elif args.command == "prepare-final-subsets":
        prepare_final_checkpoint_subsets()
    else:
        summarize(ROOT / "outputs" / "lora" / "outputs-metrics" / "sd15-seeds")


if __name__ == "__main__":
    main()
