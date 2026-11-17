"""For every image whose stored capture differs from the official benchmark image, find the prompt text that reproduces
the official image (label-free: compares generated pixels with the official PNG; never reads the annotation).
Writes outputs/claude_scratch/regen/prompt_fix_<ds>.csv. Diagnostic only."""
import csv, random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT = Path("/home/shafa/sd-lora-attention-segmentation"); EV = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam"
ds = sys.argv[1]
D, name = {"coco": (EV / "coco_captions", "ovam_coco_cap.csv"), "voc": (EV / "voc_sim", "ovam_voc_sim.csv")}[ds]
dd = dict(np.load(ROOT / f"outputs/claude_scratch/nerve_diag/imgdiff_{ds}.npy", allow_pickle=True).tolist())
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", local_files_only=True,
                                               safety_checker=None, requires_safety_checker=False).to("cuda")
pipe.set_progress_bar_config(disable=True)
R = [r for r in csv.DictReader(open(D / name)) if r["image_id"] in dd and float(dd[r["image_id"]]) >= 5]


def cands(p):
    q = p.replace("-", " ")
    out = [p, p + ".", q, q + ".", p + " .", p + ",", p + "!", p + "?", p.replace(" ,", ","), p + "..", p + " "]
    return list(dict.fromkeys(out))


rows = []
for r in R:
    off = np.asarray(Image.open(D / r["image"]).convert("RGB")).astype(int); s = int(r["seed"]); best = None
    for p in cands(r["prompt"]):
        random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        with torch.no_grad():
            img = pipe(p, num_inference_steps=int(r["num_inferece_steps"])).images[0]
        d = float(np.abs(np.asarray(img).astype(int) - off).mean())
        if best is None or d < best[1]:
            best = (p, d)
        if d < 1:
            break
    rows.append({"image_id": r["image_id"], "classname": r["classname"], "manifest_prompt": r["prompt"], "best_prompt": best[0],
                 "mean_abs_diff": f"{best[1]:.3f}", "stored_vs_official": f"{float(dd[r['image_id']]):.3f}"})
    print(r["image_id"][:44], repr(best[0][-25:]), f"{best[1]:.2f}", flush=True)
with open(ROOT / f"outputs/claude_scratch/regen/prompt_fix_{ds}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
d = np.array([float(x["mean_abs_diff"]) for x in rows])
print(f"{ds}: {len(rows)} diverging; reproduced (<1) {(d < 1).sum()}, (<2) {(d < 2).sum()}; unresolved {(d >= 2).sum()}")
