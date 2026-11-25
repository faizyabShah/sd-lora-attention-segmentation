#!/usr/bin/env python3
"""Run 44: SCLIP / ClearCLIP / NACLIP / ProxyCLIP as released, on run 44's photos (env `clipseg`).

For each column the repo's own config is used unchanged (VOC -> cfg_voc21.py, Context -> cfg_context60.py,
COCO-Object -> cfg_coco_object.py: class-name file with background, prob_thd, resize, slide settings, backbone).
Each photo goes through the config's test pipeline without LoadAnnotations (labels are never read here) and the
segmentor's mmseg `test_step` (its SegDataPreProcessor + predict + postprocess), exactly as `eval.py` runs it.
Saved per photo: the predicted label map (uint8, original size) -> <WORK>/clip_<repo>/<set>/<img>.npz.
The class index -> name table of the repo's name file is saved once per (repo, set) for scoring by name.
"""
from __future__ import annotations

import argparse, importlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("BRACE_CACHE_DIR", str(ROOT / "outputs/sg_comp"))).expanduser()
WORK = CACHE / "jc_real44"
REPOS = {"sclip": "clip_segmentor", "clearclip": "clearclip_segmentor", "naclip": "naclip", "proxyclip": "proxyclip_segmentor"}
CFG = {"voc": "cfg_voc21.py", "context": "cfg_context60.py", "coco": "cfg_coco_object.py"}
# ClearCLIP's released base_config.py sets model_type='vanilla'; that reproduces 32.0 VOC21 mIoU, while the paper's
# q-q attention ('ClearCLIP') reproduces the published 51.8 exactly (jc_repro44_voc21.py, job 263464). Use the paper's.
OVERRIDE = {"clearclip": {"model_type": "ClearCLIP"}}
COCO_ROOT = ROOT / "datasets/coco2017"


def photos(s):
    it = json.loads((WORK / "items.json").read_text())[s]
    P = {}
    for x in it:
        if x["primary"]:
            P[str(x["img"])] = str(COCO_ROOT / x["path"]) if s == "coco" else str(ROOT / x["path"])
    return sorted(P.items())


def run(a):
    import numpy as np, torch
    repo = ROOT / "other repositories" / a.repo
    os.chdir(repo); sys.path.insert(0, str(repo))
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmseg.registry import MODELS
    from mmengine.registry import init_default_scope
    init_default_scope("mmseg")                       # as the runner does (default_scope='mmseg' in base_config)
    importlib.import_module(REPOS[a.repo]); importlib.import_module("custom_datasets")
    cfg = Config.fromfile(str(repo / "configs" / CFG[a.set]))
    for k, v in OVERRIDE.get(a.repo, {}).items():
        cfg.model[k] = v
    model = MODELS.build(cfg.model).cuda(); model.eval()      # the runner moves the whole model (incl. preprocessor) to the GPU
    pipe = Compose([t for t in cfg.test_pipeline if t["type"] != "LoadAnnotations"])
    names = [l.strip() for l in open(cfg.model.name_path) if l.strip()]       # line i = class index i (synonyms comma-separated)
    out = WORK / f"clip_{a.repo}" / a.set; out.mkdir(parents=True, exist_ok=True)
    (out.parent / f"names_{a.set}.json").write_text(json.dumps({"names": names, "config": cfg.filename,
                                                                "model": cfg.model.to_dict()}, default=str, indent=1))
    P = photos(a.set)[a.shard::a.num_shards]
    if a.limit: P = P[:a.limit]
    for img, path in P:
        o = out / f"{img}.npz"
        if o.exists():
            continue
        t0 = time.time(); d = pipe(dict(img_path=path))
        with torch.no_grad():
            r = model.test_step(dict(inputs=[d["inputs"]], data_samples=[d["data_samples"]]))[0]
        pred = r.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez_compressed(tmp, pred=pred, sec=time.time() - t0); os.replace(tmp, o)
        print(f"[{a.repo}:{a.set}] {img} {pred.shape} classes {np.unique(pred).tolist()[:8]} {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, choices=list(REPOS)); ap.add_argument("--set", required=True, choices=list(CFG))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    run(ap.parse_args())
