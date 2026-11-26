"""Why does runs/19's NERVE pass differ from runs/17's on some images? Diagnostic only, no scoring of any registered cell.

env trdp:   our noised_pass (runs/19) on the stored image; fp32 pipe under autocast (as runs/19) and fp16 pipe; noise seeds.
env nerve:  NERVE's own Pipeline.refinement on the stored image (as runs/17), noise seeds 0..3.
Each run writes outputs/claude_scratch/nerve_diag/<env>.npz; `compare` prints the table (CPU).
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/shafa/sd-lora-attention-segmentation"); C = Path("/home/shafa/sg_comp")
OUT = ROOT / "outputs/claude_scratch/nerve_diag"; OUT.mkdir(parents=True, exist_ok=True)
WORST = ["ovam-coco-556255-dog-3429633183", "ovam-coco-224079-chair-3063842736", "ovam-coco-608007-person-1101767043",
         "ovam-coco-73382-dog-3660910269", "ovam-coco-158714-dog-566976589", "ovam-coco-188034-dog-3297553683"]


def ids():
    ex = sorted(f.stem for f in (C / "jc_srcrule/coco_a1_scores").glob("*.npz")
                if int(np.load(f)["img_maxdiff"]) == 0 and f.stem not in WORST)
    return WORST + ex[::40][:6]


def run_trdp():
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    import jc_srcrule as sr
    import nerve_walk_port as port
    from diffusers import StableDiffusionPipeline
    from PIL import Image
    res = {}
    for dt in ("fp32", "fp16"):
        pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None,
                                                       torch_dtype=torch.float32 if dt == "fp32" else torch.float16).to("cuda")
        for g in ids():
            img = Image.fromarray(np.load(C / f"cococap_maps/{g}.npz")["image"])
            seg = torch.from_numpy(np.load(C / f"jc_swap/nerve_seed/coco/{g}.npz")["seg"]).float().cuda()
            for seed in ((0, 1, 2, 3) if dt == "fp32" else (0, 1)):
                sr.NOISE_SEED = seed
                with torch.no_grad():
                    st = sr.noised_pass(pipe, img, sr.NERVE_LAYERS)
                    d = int(st[sr.NERVE_LAYERS[0]][0]["query"].shape[-1])
                    op = port.walk_operator(port.chained_attention_product([st[n][0] for n in sr.NERVE_LAYERS],
                                                                            torch.ones(4096, 1, device="cuda"), d=d))
                    res[f"{dt}_s{seed}|{g}"] = port.apply_walk(op, seg)[0].float().cpu().numpy()
            print(dt, g, flush=True)
        del pipe; torch.cuda.empty_cache()
    np.savez(OUT / "trdp.npz", **res)


def run_nerve():
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    import jc_swap_nerve as jn
    from PIL import Image
    pipe, cfg, _ = jn.build(refine=True)
    want = set(ids()); rows = [r for r in jn.rows_for("coco", 0, 1, 0) if r["key"] in want]
    res = {}
    for r in rows:
        pil = Image.open(r["image_local"]).convert("RGB")
        stored = np.load(C / f"cococap_maps/{r['key']}.npz")["image"]
        print(r["key"], "png == stored npz image:", bool((np.asarray(pil) == stored).all()), flush=True)
        ori = (torch.tensor(np.array(pil)).permute(2, 0, 1) / 255.0)[None]
        seg = torch.from_numpy(np.load(C / f"jc_swap/nerve_seed/coco/{r['key']}.npz")["seg"]).to(pipe.device)
        for seed in (0, 1, 2, 3):
            torch.manual_seed(seed)
            with torch.no_grad():
                res[f"nerve_s{seed}|{r['key']}"] = pipe.refinement(ori, seg)[0].float().cpu().numpy()
    np.savez(OUT / f"nerve_{sys.argv[2]}.npz", **{k.replace("nerve_", f"nerve{sys.argv[2]}_"): v for k, v in res.items()})


def compare():
    import torch, torch.nn.functional as F
    mm = lambda x: (x - x.min()) / max(x.max() - x.min(), 1e-12)
    R = {}
    for f in sorted(OUT.glob("*.npz")):
        z = np.load(f); R.update({k: z[k] for k in z.files})
    variants = sorted({k.split("|")[0] for k in R})
    print("IoU at 0.5 (bilinear to GT, min-max), per image; corr = Pearson of min-max maps vs runs/17's stored walk")
    print(f"{'image':38s} {'runs17':>7s} {'runs19':>7s} " + " ".join(f"{v:>14s}" for v in variants))
    for g in ids():
        z19 = np.load(C / f"jc_srcrule/coco_a1/{g}.npz"); truth = z19["truth"].astype(bool)
        w17 = np.load(C / f"jc_swap/walk/coco/{g}.npz")["walk_clip"].squeeze()

        def iou(m):
            u = F.interpolate(torch.from_numpy(mm(m.astype(np.float32)))[None, None], size=truth.shape, mode="bilinear")[0, 0].numpy() > .5
            return (u & truth).sum() / max((u | truth).sum(), 1)
        cells = []
        for v in variants:
            m = R[f"{v}|{g}"].squeeze()
            cells.append(f"{iou(m):.2f}/{np.corrcoef(mm(m).ravel(), mm(w17).ravel())[0, 1]:.2f}/{m.max():.2f}")
        print(f"{g[:38]:38s} {iou(w17):7.2f} {iou(z19['maps'][1, 6]):7.2f} " + " ".join(f"{c:>14s}" for c in cells))
    print("cells: IoU / corr with runs/17 / raw peak")


if __name__ == "__main__":
    {"trdp": run_trdp, "nerve": run_nerve, "compare": compare}[sys.argv[1]]()
