#!/usr/bin/env python3
"""Run 44: DiffSegmenter on run 44's real-photo items (env /home/sidna/miniconda3/envs/diffseg).

Reuses the reviewed class-word driver `comparisons/diffsegmenter/diffseg_classword.py` (Narmeen; DiffSegmenter repo copy in
/home/sidna/TRDP/external/n_side/diffseg/repo) unchanged in substance, with **caption mode `blip`** (the faithful prompt:
"a photograph of {class}++" + the tail of a BLIP-large conditional caption, Salesforce/blip-image-captioning-large) and
seed 0 per item. Its loop body is repeated here only to read run 44's items instead of a manifest and to save the 64 x 64
DiffSegmenter map (cls_all tokens; the bilinear resize to photo size and the min-max are done at scoring time, exactly as
the driver does: :537-543). One record per (photo, class) item. Ground truth is never read.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("BRACE_CACHE_DIR", str(ROOT / "outputs/sg_comp"))).expanduser()
WORK = CACHE / "jc_real44"
DRIVER = ROOT / "comparisons/diffsegmenter/diffseg_classword.py"
DS_REPO = Path("/home/sidna/TRDP/external/n_side/diffseg/repo")
BLIP = "/home/shafa/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-large/snapshots/353689b859fcf0523410b1806dace5fb46ecdf41"
COCO_ROOT = ROOT / "datasets/coco2017"
SEED = 0


def run(a):
    import importlib.util
    import numpy as np, torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    spec = importlib.util.spec_from_file_location("dsc", DRIVER); D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
    D.REPO = DS_REPO
    ds = D.load_ds_module()
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from transformers import BlipProcessor, BlipForConditionalGeneration
    device = torch.device("cuda")
    ldm = StableDiffusionPipeline.from_pretrained(D.MODEL_KEY, local_files_only=True, torch_dtype=torch.float16).to(device)
    ldm.scheduler = DDIMScheduler.from_pretrained(D.MODEL_KEY, subfolder="scheduler", local_files_only=True)
    tok, vae = ldm.tokenizer, ldm.vae
    ds.ldm_stable, ds.tokenizer, ds.device = ldm, tok, device
    ctrl = ds.AttentionStore(); g_cpu = torch.Generator(4307)
    blip = (BlipProcessor.from_pretrained(BLIP), BlipForConditionalGeneration.from_pretrained(BLIP).to(device))
    it = [x for x in json.loads((WORK / "items.json").read_text())[a.set] if x["primary"]][a.shard::a.num_shards]
    if a.limit: it = it[:a.limit]
    out = WORK / "diffseg" / a.set; out.mkdir(parents=True, exist_ok=True)
    to_tensor = transforms.ToTensor()
    with torch.no_grad():
        for x in it:
            o = out / f"{x['id']}.npz"
            if o.exists():
                continue
            from jc_real44 import query_word
            cls = query_word(a.set, x["cls"]); path = COCO_ROOT / x["path"] if a.set == "coco" else ROOT / x["path"]
            image = Image.open(path).convert("RGB"); t0 = time.time()
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            orig = to_tensor(image).unsqueeze(0).clamp(0, 1)
            rgb = F.interpolate(orig, (512, 512), mode="bilinear", align_corners=False).to(device).half()
            lat = ds.encode_imgs(rgb, vae).to(device)
            noise = torch.randn([1, 4, 64, 64]).to(device).half()
            base = f"a photograph of {cls}"
            inputs = blip[0](image, base, return_tensors="pt").to(device)
            tail = blip[0].decode(blip[1].generate(**inputs)[0], skip_special_tokens=True)[len(base):]
            lab = f"({cls})" if " " in cls else cls
            prompt = [f"a photograph of {lab}++" + tail]
            words = prompt[0].split(" ")
            cls_pos = D.class_token_positions(tok, "a photograph of", cls)
            toks = cls_pos + [cls_pos[-1] + 1] if (len(words) > 4 and words[4].endswith("ing")) else cls_pos
            maps = []
            for t in D.TIMES:
                ctrl.reset()
                noisy = ldm.scheduler.add_noise(lat, noise, torch.tensor(t, device=device)).half()
                ds.run_and_display(prompt, ctrl, latent=noisy, run_baseline=False, generator=g_cpu, t=t, noise_sample_num=1)
                cross = ds.aggregate_all_attention(prompt, ctrl, ("up", "mid", "down"), True, 0)
                selfm = ds.aggregate_all_attention(prompt, ctrl, ("up", "mid", "down"), False, 0)
                sa = selfm[3].view(64 * 64, 64 * 64).float(); outs = []
                for idx, res in enumerate([8, 16, 32, 64]):
                    ca = cross[idx][:, :, toks].mean(2).view(res, res).float()
                    if res != 64:
                        ca = F.interpolate(ca[None, None], size=(64, 64), mode="bilinear", align_corners=False).squeeze()
                    outs.append((ca - ca.min()) / (ca.max() - ca.min()) * D.WEIGHT[idx])
                maps.append(torch.matmul(sa, torch.stack(outs).sum(0).view(64 * 64, 1)).view(64, 64))
            m64 = torch.stack(maps).mean(0).cpu().numpy().astype(np.float32)
            tmp = o.with_name(o.stem + ".tmp.npz"); np.savez_compressed(tmp, m64=m64, prompt=prompt[0], toks=np.array(toks),
                                                                      size=np.array(image.size[::-1]), sec=time.time() - t0)
            os.replace(tmp, o)
            print(f"[diffseg:{a.set}] {x['id']} '{prompt[0][:70]}' {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["voc", "context", "coco"]); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    run(ap.parse_args())
