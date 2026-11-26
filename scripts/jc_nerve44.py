#!/usr/bin/env python3
"""Run 44: NERVE as released on run 44's real photos (NERVE env), one config per column.

Uses run 43's prepared repo copy (<CACHE>/jc_eval43/nerve_repo: only the HF mirror line disabled and SD 2.1-base loaded
from the sd2-community snapshot). Per column, NERVE's own YAML + util/args.py defaults (as main_ovs.parse_args merges them):
  voc      scripts/config/vit-l-14/ovs_voc21.yaml   + data/voc/cls_voc21.txt      (background words + 20 classes)
  context  scripts/config/vit-l-14/ovs_context.yaml + data/voc/cls_context60.txt  ("background" + 59 classes, alphabetical)
  coco     scripts/config/vit-l-14/ovs_coco5k.yaml  + data/coco14/cls_coco81.txt  (background words + 80 classes)
Per photo, exactly main_ovs.py's loop body: Pipeline(ori, img, fg, bg) -> bilinear to photo size -> /amax * final_score ->
cam_to_label(bkg_thre, score_threshold, is_normalize). Saved: the label map (0 = background, i = NERVE class i) and
final_score, plus the soft maps of the classes queried on this photo (secondary read-out).
Seed: torch.manual_seed(42) before each photo (main_ovs seeds 42 once per run; per photo keeps every photo reproducible).
"""
from __future__ import annotations

import argparse, json, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("BRACE_CACHE_DIR", str(ROOT / "outputs/sg_comp"))).expanduser()
WORK = CACHE / "jc_real44"
REPO = CACHE / "jc_eval43" / "nerve_repo"
YAML = {"voc": "ovs_voc21.yaml", "context": "ovs_context.yaml", "coco": "ovs_coco5k.yaml"}
NAMES = {"voc": "data/voc/cls_voc21.txt", "context": "data/voc/cls_context60.txt", "coco": "data/coco14/cls_coco81.txt"}
ARGS_DEFAULTS = dict(fuse_feature=True, refinement="SFSA", attention_idx=1, attn_type="fused-attn", time_step=45,
                     sd_version="v2.1", size=336, log_path="log")              # util/args.py non-None defaults
COCO_ROOT = ROOT / "datasets/coco2017"
SEED = 42


def photos(s):
    it = json.loads((WORK / "items.json").read_text())[s]; P = {}
    for x in it:
        if x["primary"]:
            p = str(COCO_ROOT / x["path"]) if s == "coco" else str(ROOT / x["path"])
            P.setdefault(str(x["img"]), [p, set()])[1].add(x["cls"])
    return sorted(P.items())


def run(a):
    import numpy as np, torch
    import torch.nn.functional as F
    from PIL import Image
    os.chdir(REPO / "ovs"); sys.path.insert(0, str(REPO))
    from util.tools import load_yaml
    from util.miou import cam_to_label
    from ovs.pipeline_clean_mean_final_run_correct import Pipeline
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    cfg = load_yaml(str(REPO / "scripts/config/vit-l-14" / YAML[a.set]))
    for k, v in ARGS_DEFAULTS.items():
        setattr(cfg, k, v)
    cfg.semantic_templates = [l.strip() for l in open(REPO / "ovs" / cfg.semantic_templates)]
    lines = open(REPO / NAMES[a.set]).read().splitlines()
    bg_words, cats = re.split(r"[\s,]+", lines[0]), lines[1:]                 # as the dataset classes do
    dev = torch.device("cuda:0"); pipe = Pipeline(cfg, device=dev)
    fg = pipe.nerve.classifier(cats, cfg.semantic_templates).to(dev); bg = pipe.nerve.classifier(bg_words, cfg.semantic_templates).to(dev)
    out = WORK / "nerve" / a.set; out.mkdir(parents=True, exist_ok=True)
    (out.parent / f"meta_{a.set}.json").write_text(json.dumps({"yaml": YAML[a.set], "categories": cats, "bg_words": bg_words,
        "cfg": {k: str(v) for k, v in vars(cfg).items()}, "seed": SEED}, indent=1, default=str))
    canon = json.loads(a.canon) if a.canon else None
    P = photos(a.set)[a.shard::a.num_shards]
    if a.limit: P = P[:a.limit]
    for img, (path, qcls) in P:
        o = out / f"{img}.npz"
        if o.exists():
            continue
        pil = Image.open(path).convert("RGB"); W, H = pil.size
        ori = (torch.tensor(np.array(pil)).permute(2, 0, 1) / 255.0)[None]
        x = pipe.nerve.preprocess(pil)[None].to(dev)
        torch.manual_seed(SEED); t0 = time.time()
        with torch.no_grad():
            pred, score = pipe(ori, x, fg, bg)
            pred = F.interpolate(pred[None], size=(H, W), mode="bilinear")[0]
            pm = pred / (pred.amax((-1, -2), keepdim=True) + 1e-5) * score[..., None, None]
            _, arg = cam_to_label(pm[None].clone(), cls_label=score[None], bkg_thre=cfg.bkg_thre,
                                  cls_thre=cfg.score_threshold, is_normalize=cfg.is_normalize)
        lab = arg[0].cpu().numpy().astype(np.uint8)
        soft = {f"soft_{i}": pred[i].float().cpu().numpy().astype(np.float16) for i in a_idx(cats, qcls, a.set)}
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez_compressed(tmp, label=lab, final_score=score.float().cpu().numpy(), sec=time.time() - t0, **soft); os.replace(tmp, o)
        print(f"[nerve:{a.set}] {img} labels {np.unique(lab).tolist()[:10]} {time.time() - t0:.1f}s", flush=True)


def a_idx(cats, qcls, s):
    """NERVE channel indices (0-based over `cats`) of the queried classes, matched by name (see NAME_MAP)."""
    from jc_names44 import method_index
    return [method_index("nerve", s, cats, c) for c in sorted(qcls)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(YAML)); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--canon", default="")
    sys.path.insert(0, str(ROOT / "scripts")); run(ap.parse_args())
