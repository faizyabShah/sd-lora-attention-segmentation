#!/usr/bin/env python3
"""NERVE as released, on run 43's regenerated PNGs (runs/43_generated_benchmarks/PREREG.md section 4). NERVE env.

prep  copy `other repositories/nerve` (kunal-mahatha/nerve @ 1ddc5f9) to <WORK>/nerve_repo with two
      infrastructure-only edits (HF mirror endpoint line disabled; SD 2.1-base loaded from the sd2-community
      re-upload because stabilityai/stable-diffusion-2-1-base was withdrawn). No method change.
run   per image: NERVE's Pipeline (ovs/pipeline_clean_mean_final_run_correct.py, imported by ovs/main_ovs.py) with
      ovs_voc21.yaml + util/args.py defaults, NERVE's 20 VOC names and background words, then main_ovs.py:61-63.
      -> <WORK>/nerve{tag}/{set}/{id}.npz: own (NERVE's binarisation == queried class), nogate, soft, final_score
"""
from __future__ import annotations

import argparse, json, os, re, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("BRACE_CACHE_DIR", str(ROOT / "outputs/sg_comp"))).expanduser()
WORK = CACHE / "jc_eval43"
SRC = ROOT / "other repositories/nerve"
REPO = WORK / "nerve_repo"
SD_ID = "sd2-community/stable-diffusion-2-1-base"
SD_PATH = "/home/sidna/.cache/huggingface/hub/models--sd2-community--stable-diffusion-2-1-base/snapshots/4e63672c03103b6c636b8fb4119ba982469b2955"   # cached snapshot (no refs/main, so load by path)
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
           "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv"]
SEED = 42


def prep(a):
    if REPO.exists():
        print("repo exists:", REPO); return
    WORK.mkdir(parents=True, exist_ok=True); tmp = WORK / "nerve_repo.tmp"; shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(SRC, tmp, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    p = tmp / "diffusion_model/stable_diffusion.py"; s = p.read_text()
    e = "os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'"
    m = 'model = "stabilityai/stable-diffusion-2-1-base"'
    assert s.count(e) == 1 and s.count(m) == 1
    s = s.replace(e, "# [run 43, infrastructure: offline local cache] " + e)
    s = s.replace(m, f'model = "{SD_PATH}"  # [run 43, infrastructure] withdrawn stabilityai repo -> {SD_ID} snapshot')
    p.write_text(s); os.replace(tmp, REPO); print("prepared", REPO)


def run(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, str(REPO))
    from util.tools import load_yaml
    from util.miou import cam_to_label
    from data.voc.voc_datasets import VOC2012_CAT_NAME
    from ovs.pipeline_clean_mean_final_run_correct import Pipeline
    from diffusion_model.Processor import DIFFUSION_LAYERS

    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False   # main_ovs.set_seed
    cfg = load_yaml(str(REPO / "scripts/config/vit-l-14/ovs_voc21.yaml"))
    for k, v in dict(fuse_feature=True, refinement="SFSA", attention_idx=1, attn_type="fused-attn", time_step=45,
                     sd_version="v2.1", size=336, log_path="log").items():          # util/args.py defaults
        setattr(cfg, k, v)
    cfg.semantic_templates = [l.strip() for l in open(REPO / "template/car_template.txt")]
    dev = torch.device("cuda:0"); pipe = Pipeline(cfg, device=dev)
    sd = pipe.attn_refine.model
    hd = {i: (lambda m: m.to_q.out_features // m.heads)(eval("unet." + DIFFUSION_LAYERS[i], {"unet": sd.unet}))
          for i in cfg.attention_layers_to_use}
    assert set(hd.values()) == {64}, hd                       # NERVE's literal d = 64 is SD 2.1's head size
    names20 = VOC2012_CAT_NAME[1:]; assert len(names20) == 20
    for c, n in zip(CLASSES, names20):
        assert n.split()[0].replace("diningtable", "dining").startswith(c.replace(" ", "")[:4]) or c == "tv", (c, n)
    bg_words = re.split(r"[\s,]+", VOC2012_CAT_NAME[0])        # voc_datasets.py
    fg = pipe.nerve.classifier(names20, cfg.semantic_templates).to(dev)
    bg = pipe.nerve.classifier(bg_words, cfg.semantic_templates).to(dev)
    sched = sd.scheduler; sched.set_timesteps(50, device=dev)
    meta = {"sd": SD_ID, "sd_snapshot": [SD_PATH], "head_dim": hd, "names20": names20,
        "bg_words": bg_words, "t_index": cfg.time_step, "seed": SEED, "cfg": {k: str(v) for k, v in vars(cfg).items()}}
    out = WORK / f"nerve{a.tag}" / a.set; out.mkdir(parents=True, exist_ok=True)
    (out.parent / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    pngs = sorted((WORK / f"png{a.tag}" / a.set).glob("*.png"))[a.shard::a.num_shards]
    labels = json.loads((WORK / f"labels{a.tag}.json").read_text())
    for f in pngs:
        o = out / (f.stem + ".npz")
        if o.exists():
            continue
        c = CLASSES.index(labels[f.stem]); pil = Image.open(f).convert("RGB"); W, H = pil.size
        ori = (torch.tensor(np.array(pil)).permute(2, 0, 1) / 255.0)[None]          # voc_datasets.__getitem__
        img = pipe.nerve.preprocess(pil)[None].to(dev)
        torch.manual_seed(SEED); t0 = time.time()
        with torch.no_grad():
            pred, score = pipe(ori, img, fg, bg)
            pred = F.interpolate(pred[None], size=(H, W), mode="bilinear")[0]
            soft = pred[c].clone()
            pm = pred / (pred.amax((-1, -2), keepdim=True) + 1e-5) * score[..., None, None]   # main_ovs.py:61
            _, arg = cam_to_label(pm[None].clone(), cls_label=score[None], bkg_thre=cfg.bkg_thre,
                                  cls_thre=cfg.score_threshold, is_normalize=cfg.is_normalize)
            _, ng = cam_to_label(pm[c:c + 1][None].clone(), cls_label=torch.ones(1, 1, device=dev),
                                 bkg_thre=cfg.bkg_thre, cls_thre=0.0, is_normalize=cfg.is_normalize)
        own = (arg[0] == c + 1).cpu().numpy(); nog = (ng[0] == 1).cpu().numpy(); s = soft.float().cpu().numpy()
        if not np.isfinite(s).all():
            raise ValueError("non-finite NERVE map for " + f.stem)
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, own=own, nogate=nog, soft=s.astype(np.float16), final_score=score.float().cpu().numpy(), cls=c,
                 sec=time.time() - t0)
        os.replace(tmp, o)
        print(f"[nerve:{a.set}] {f.stem} {CLASSES[c]} score {float(score[c]):.3f} own {own.mean():.3f} "
              f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    p = sub.add_parser("run"); p.add_argument("--set", required=True); p.add_argument("--tag", default="")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args(); {"prep": prep, "run": run}[a.cmd](a)
