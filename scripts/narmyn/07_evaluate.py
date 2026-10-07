"""Score real / frozen / LoRA image sets on the full metric suite.

Metrics, following the Kaggle evaluation notebook:

  FID   clean-fid, generated vs real          lower better
  KID   clean-fid, unbiased -- usable at N=500 lower better
  CLIP  torchmetrics CLIPScore vs the prompt   higher better
  BLIP  blip-itm-base-coco: ITM prob + cosine  higher better
  LPIPS mean pairwise distance within a set    higher = more diverse

Two things the notebook did not have to worry about, which matter here:

  * Matched N and matched indices. FID is strongly sample-size biased, so
    every set is scored on the SAME index list -- by default the intersection
    of all sets being compared. Comparing a 500-image checkpoint dir against a
    24,995-image frozen dir directly would be meaningless.
  * Real reference resolution. Crops are 1024^2, generations are 768^2. The
    reals are resized once into a cache dir so both sides enter the metric
    from the same pixel grid.

Usage:
  # rank sweep at one checkpoint, against real + frozen
  python scripts/07_evaluate.py --sets frozen rank8 rank16 rank32 rank64 rank128

  # pick the best checkpoint for one rank
  python scripts/07_evaluate.py --sets frozen rank8_ckpt3000 rank8_ckpt9000 ...

  # every checkpoint of every rank (the 50-cell sweep)
  python scripts/07_evaluate.py --sweep --metrics fid,kid
"""

import os
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image

# Data and outputs live under $TRDP_ROOT when set (the pipeline working
# tree on the cluster); otherwise they resolve relative to this repo.
PROJECT_ROOT = Path(os.environ.get("TRDP_ROOT",
                                   Path(__file__).resolve().parents[2]))
REAL_ROOT = PROJECT_ROOT / "data" / "cityscapes_cropped"
CAPTIONS = PROJECT_ROOT / "data" / "florence2_detailedcaptions" / "metadata.jsonl"
FROZEN_DIR = PROJECT_ROOT / "output" / "frozen_images"
LORA_ROOT = PROJECT_ROOT / "output" / "lora_images"
REAL_CACHE = PROJECT_ROOT / "output" / "real_resized"
RESULTS = PROJECT_ROOT / "output" / "evaluation"

SIZE = 768
CLIP_MODEL = "openai/clip-vit-large-patch14"
BLIP_MODEL = "Salesforce/blip-itm-base-coco"
LPIPS_PAIRS = 500
LPIPS_SAMPLE = 200


def load_entries():
    with CAPTIONS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_set(name: str) -> Path:
    """'frozen' | 'rank8' | 'rank8_ckpt9000' | an explicit path."""
    if name == "frozen":
        return FROZEN_DIR
    p = LORA_ROOT / name
    if p.is_dir():
        return p
    p2 = Path(name)
    if p2.is_dir():
        return p2
    print(f"ERROR: cannot resolve image set {name!r}", file=sys.stderr)
    sys.exit(1)


def indices_in(d: Path):
    return {int(p.stem) for p in d.glob("[0-9]*.png")}


def build_real_cache(indices, size: int) -> Path:
    """Real crops resized to the generation resolution, one file per index.

    Named {i:05d}.png like the generated sets so the same index list selects
    matching real and fake images.
    """
    out = REAL_CACHE / f"{size}"
    out.mkdir(parents=True, exist_ok=True)
    entries = load_entries()
    made = 0
    for i in indices:
        dst = out / f"{i:05d}.png"
        if dst.exists():
            continue
        src = REAL_ROOT / entries[i]["file_name"]
        with Image.open(src) as im:
            im.convert("RGB").resize((size, size), Image.LANCZOS).save(
                dst.with_suffix(".tmp"), format="PNG")
        dst.with_suffix(".tmp").replace(dst)
        made += 1
    print(f"  real cache: {out}  (+{made} new, {len(indices)} needed)", flush=True)
    return out


def subset_dir(src: Path, indices, tag: str) -> Path:
    """Symlink farm holding exactly `indices` from `src`.

    clean-fid consumes directories, so restricting a set to the shared index
    list means materialising that subset. Symlinks keep it free.
    """
    out = RESULTS / "subsets" / tag
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.png"):
        p.unlink()
    for i in indices:
        (out / f"{i:05d}.png").symlink_to((src / f"{i:05d}.png").resolve())
    return out


# ------------------------------------------------------------------ metrics --
def metric_fid_kid(fake_dir: Path, real_dir: Path, want_fid, want_kid):
    from cleanfid import fid as cfid
    res = {}
    if want_fid:
        res["fid"] = float(cfid.compute_fid(str(fake_dir), str(real_dir), mode="clean"))
    if want_kid:
        res["kid"] = float(cfid.compute_kid(str(fake_dir), str(real_dir), mode="clean"))
    return res


def metric_clip(paths, prompts, device, batch=16):
    """CLIPScore = 100 * max(0, cos(image_embed, text_embed)).

    Computed straight from CLIPModel rather than through
    torchmetrics.CLIPScore: on transformers 5.x, get_image_features returns a
    BaseModelOutputWithPooling instead of a tensor, which torchmetrics 1.9
    then tries to call .norm() on. The full forward still exposes the properly
    projected image_embeds/text_embeds, and this is the same definition.
    """
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    scores = []
    for s in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
        # CLIP's text encoder caps at 77 tokens; truncation is applied
        # identically to every set so comparisons stay fair.
        inp = proc(text=list(prompts[s:s + batch]), images=imgs,
                   return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model(**inp)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        scores += (100 * (ie * te).sum(-1).clamp(min=0)).tolist()
    del model
    torch.cuda.empty_cache()
    return {"clip": sum(scores) / len(scores)}


def metric_blip(paths, prompts, device, batch=16):
    """BLIP image-text matching: ITM probability and cosine head.

    NOTE: blip-itm-base-coco ships only a .bin checkpoint, and transformers
    5.x refuses torch.load unless torch >= 2.6 (CVE-2025-32434). So this needs
    the `trdp` env (torch 2.11), not `gen` (torch 2.4.1) -- i.e. renyi.
    """
    from transformers import BlipForImageTextRetrieval, BlipProcessor
    proc = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForImageTextRetrieval.from_pretrained(
        BLIP_MODEL, torch_dtype=torch.float16).to(device).eval()
    itm, cos = [], []
    for s in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
        txt = [t[:200] for t in prompts[s:s + batch]]
        inp = proc(images=imgs, text=txt, return_tensors="pt",
                   padding=True, truncation=True).to(device, torch.float16)
        with torch.no_grad():
            itm += torch.nn.functional.softmax(
                model(**inp)[0].float(), dim=1)[:, 1].tolist()
            cos += model(**inp, use_itm_head=False)[0].float().reshape(-1).tolist()
    del model
    torch.cuda.empty_cache()
    return {"blip_itm": sum(itm) / len(itm), "blip_cos": sum(cos) / len(cos)}


def metric_lpips(paths, device, seed=42):
    """Mean pairwise LPIPS inside one set -- a diversity proxy."""
    import random
    import lpips as lpips_lib
    from torchvision import transforms
    loss = lpips_lib.LPIPS(net="alex").to(device)
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    rng = random.Random(seed)
    sample = rng.sample(list(paths), min(LPIPS_SAMPLE, len(paths)))
    imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in sample])
    allp = [(i, j) for i in range(len(imgs)) for j in range(i + 1, len(imgs))]
    pairs = rng.sample(allp, min(LPIPS_PAIRS, len(allp)))
    vals = []
    with torch.no_grad():
        for i, j in pairs:
            vals.append(float(loss(imgs[i:i + 1].to(device), imgs[j:j + 1].to(device)).item()))
    del loss
    torch.cuda.empty_cache()
    import statistics as st
    return {"lpips_div": st.mean(vals), "lpips_std": st.pstdev(vals)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=[],
                    help="frozen | rank8 | rank8_ckpt9000 | explicit path")
    ap.add_argument("--sweep", action="store_true",
                    help="evaluate every rank{R}_ckpt{S} directory found")
    ap.add_argument("--metrics", default="fid,kid,clip,blip,lpips")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--max-images", type=int,
                    help="cap the shared index list (after intersection)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=RESULTS / "results.json")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA unavailable -- run on a GPU node.", file=sys.stderr)
        return 1

    names = list(args.sets)
    if args.sweep:
        found = sorted(LORA_ROOT.glob("rank*_ckpt*"),
                       key=lambda p: (int(re.search(r"rank(\d+)", p.name).group(1)),
                                      int(re.search(r"ckpt(\d+)", p.name).group(1))))
        names += [p.name for p in found]
    if not names:
        print("ERROR: nothing to evaluate (pass --sets and/or --sweep)", file=sys.stderr)
        return 1

    dirs = {n: resolve_set(n) for n in names}

    # Shared index list: every metric sees the same images from every set.
    shared = set.intersection(*(indices_in(d) for d in dirs.values()))
    if not shared:
        print("ERROR: the chosen sets share no image indices", file=sys.stderr)
        return 1
    shared = sorted(shared)
    if args.max_images:
        shared = shared[:args.max_images]
    print(f"{len(dirs)} sets, {len(shared)} shared indices "
          f"({shared[0]}..{shared[-1]})", flush=True)

    want = set(args.metrics.split(","))
    entries = load_entries()
    prompts = [entries[i]["text"] for i in shared]
    RESULTS.mkdir(parents=True, exist_ok=True)

    real_dir = None
    if {"fid", "kid"} & want:
        real_dir = subset_dir(build_real_cache(shared, args.size), shared, "real")

    results = {}
    for name, d in dirs.items():
        print(f"\n=== {name} ===", flush=True)
        paths = [d / f"{i:05d}.png" for i in shared]
        r = {"n": len(shared), "dir": str(d)}
        if {"fid", "kid"} & want:
            r.update(metric_fid_kid(subset_dir(d, shared, name), real_dir,
                                    "fid" in want, "kid" in want))
        if "clip" in want:
            r.update(metric_clip(paths, prompts, args.device))
        if "blip" in want:
            r.update(metric_blip(paths, prompts, args.device))
        if "lpips" in want:
            r.update(metric_lpips(paths, args.device))
        results[name] = r
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()
                               if isinstance(v, float)), flush=True)

    # real-vs-real is the noise floor: how low FID/KID can possibly go here
    if {"fid", "kid"} & want and len(shared) >= 200:
        half = len(shared) // 2
        a = subset_dir(Path(real_dir), shared[:half], "real_a")
        b = subset_dir(Path(real_dir), shared[half:], "real_b")
        results["_real_vs_real"] = {"n": half, **metric_fid_kid(a, b, "fid" in want, "kid" in want)}
        print(f"\n=== real vs real (noise floor, n={half}) ===")
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in results['_real_vs_real'].items()
                               if isinstance(v, float)))

    suffix = f"_{args.tag}" if args.tag else ""
    out_json = args.out.with_name(args.out.stem + suffix + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump({"n_images": len(shared), "size": args.size,
                   "indices": shared, "results": results}, f, indent=2)

    cols = [c for c in ("fid", "kid", "clip", "blip_itm", "blip_cos", "lpips_div")
            if any(c in v for v in results.values())]
    arrow = {"fid": "v", "kid": "v", "clip": "^", "blip_itm": "^",
             "blip_cos": "^", "lpips_div": "^"}
    lines = [f"n={len(shared)} images, {args.size}px",
             "",
             "| set | " + " | ".join(f"{c} {arrow[c]}" for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, r in results.items():
        lines.append("| " + name + " | " +
                     " | ".join(f"{r[c]:.4f}" if c in r else "-" for c in cols) + " |")
    table = "\n".join(lines)
    out_md = out_json.with_suffix(".md")
    out_md.write_text(table + "\n")
    print("\n" + table)
    print(f"\nwrote {out_json}\n      {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
