#!/usr/bin/env python3
"""Run 44: iSeg (linsun449/iseg.code @ 57e9a17) as released, on run 44's real photos (env `iseg`).

prep    copy the repo to <WORK>/iseg_repo with one infrastructure edit: base/stable_difusion.py loads SD 2.1-base from the
        local sd2-community snapshot (stabilityai's repo id was withdrawn). Link COCO photos as <root>/val2014/<12-digit>.jpg
        because iSeg's COCO code reads that layout.
labels  iSeg's own image-label step, `ovs/predict_labels.py` (TagCLIP), run unchanged via runpy with the column's argument
        group (ovs/arguments.py: add_voc_dataset_args / add_voc_context_dataset_args / add_coco_object_dataset_args, which
        the authors switch by commenting lines in init_args) and our photo list -> <WORK>/iseg/labels_<set>/<name>.npy
seg     transcription of ovs/diff_for_openvocabulary.py DiffForOpenVocabulary.test_step (released settings per column:
        iterations, enhancement 1.6, entropy, att_mean, cam_bg_thr; t = [100]; rand_seed 1101 inside iSeg's SD wrapper):
        image as the datasets load it (RGB / 255, cv2.resize to 512 x 512, CHW), labels from `labels` (primary, TagCLIP) or
        the queried classes (--given, secondary), prompt "a photograph of <c1> and <c2> ... and other object and background"
        + bg_context, cross-attention enhancement, get_masks, bilinear to the photo size (the VOC/Context loaders keep the
        label at photo size), cam_to_label(bkg_thre=cam_bg_thr). -> label map (0 = background, k + 1 = iSeg class k).
"""
from __future__ import annotations

import argparse, json, os, runpy, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("BRACE_CACHE_DIR", str(ROOT / "outputs/sg_comp"))).expanduser()
WORK = CACHE / "jc_real44"
SRC = ROOT / "other repositories/iseg"
REPO = WORK / "iseg_repo"
SD_PATH = "/home/sidna/.cache/huggingface/hub/models--sd2-community--stable-diffusion-2-1-base/snapshots/4e63672c03103b6c636b8fb4119ba982469b2955"
GROUP = {"voc": "add_voc_dataset_args", "context": "add_voc_context_dataset_args", "coco": "add_coco_object_dataset_args"}
VOC_ROOT = ROOT / "datasets/voc2012/VOCdevkit/VOC2012"
COCO_ROOT = ROOT / "datasets/coco2017"


def photos(s):
    it = json.loads((WORK / "items.json").read_text())[s]; P = {}
    for x in it:
        if x["primary"]:
            name = f"{int(x['img']):012d}" if s == "coco" else str(x["img"])
            p = COCO_ROOT / x["path"] if s == "coco" else ROOT / x["path"]
            P.setdefault(name, [str(p), set()])[1].add(x["cls"])
    return sorted(P.items())


def prep(a):
    if not REPO.exists():
        tmp = WORK / "iseg_repo.tmp"; shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(SRC, tmp, ignore=shutil.ignore_patterns(".git", "__pycache__", "assert"))
        p = tmp / "base/stable_difusion.py"; s = p.read_text(); m = 'model_key = "stabilityai/stable-diffusion-2-1-base"'
        assert s.count(m) == 1
        p.write_text(s.replace(m, f'model_key = "{SD_PATH}"  # [run 44, infrastructure] withdrawn stabilityai repo -> sd2-community snapshot'))
        os.replace(tmp, REPO); print("prepared", REPO)
    link = WORK / "iseg_coco_root/val2014"; link.mkdir(parents=True, exist_ok=True)
    for name, (p, _) in photos("coco"):
        q = link / f"{name}.jpg"
        if not q.exists():
            os.symlink(p, q)
    for s in GROUP:
        (WORK / "iseg").mkdir(parents=True, exist_ok=True)
        (WORK / "iseg" / f"list_{s}.txt").write_text("\n".join(n for n, _ in photos(s)) + "\n")
    print("lists and COCO links written")


def config(s):
    sys.path.insert(0, str(REPO)); os.chdir(REPO / "ovs")
    import ovs.arguments as A
    ap = argparse.ArgumentParser()
    for f in ("add_base_args", GROUP[s], "add_train_args", "add_test_args", "add_classifier_args", "add_shell_args"):
        ap = getattr(A, f)(ap)
    cfg = ap.parse_args([])
    cfg.root_datasets_path = str(WORK / "iseg_coco_root") if s == "coco" else str(VOC_ROOT) + "/"
    cfg.test_data_dir = str(WORK / "iseg" / f"list_{s}.txt")
    cfg.classified_result_dir = str(WORK / "iseg" / f"labels_{s}")
    return A, cfg


def labels(a):
    A, cfg = config(a.set); os.makedirs(cfg.classified_result_dir, exist_ok=True)
    if a.limit:                                                                # smoke: first photos of this shard only
        L = [n for n, _ in photos(a.set)][a.shard::a.num_shards][:a.limit]
        cfg.test_data_dir = str(WORK / "iseg" / f"list_{a.set}_smoke.txt"); Path(cfg.test_data_dir).write_text("\n".join(L) + "\n")
    A.init_args = lambda: cfg                                                  # the authors' switch, done by argument
    sys.argv = ["predict_labels.py"]
    runpy.run_path(str(REPO / "ovs/predict_labels.py"), run_name="__main__")


def seg(a):
    import numpy as np, torch, cv2, imageio
    import torch.nn.functional as F
    from functools import reduce
    from operator import add
    A, cfg = config(a.set)
    sys.path.append(str(REPO / "base"))
    from wsss.iSeg import iSeg
    from util.cam import cam_to_label
    model = iSeg(cfg).cuda().eval(); model.on_test_start()
    names = model.class_name
    from jc_names44 import method_index
    out = WORK / "iseg" / (f"{a.set}_given" if a.given else a.set); out.mkdir(parents=True, exist_ok=True)
    (out.parent / f"names_{a.set}.json").write_text(json.dumps({"names": names, "cfg": {k: str(v) for k, v in vars(cfg).items()}}, indent=1))
    P = photos(a.set)[a.shard::a.num_shards]
    if a.limit: P = P[:a.limit]
    for name, (path, qcls) in P:
        o = out / f"{name}.npz"
        if o.exists():
            continue
        t0 = time.time()
        img = np.asarray(imageio.imread(path))
        if img.ndim == 2:
            img = np.stack([img] * 3, 2)
        H, W = img.shape[:2]
        x = cv2.resize(img[..., :3].astype(np.float32) / 255.0, dsize=(cfg.test_mask_size, cfg.test_mask_size))
        image = torch.from_numpy(x.transpose(2, 0, 1))[None].cuda().half()
        if a.given:
            ids = sorted(method_index("iseg", a.set, names, c) for c in qcls)
        else:
            ids = sorted(int(i) for i in np.atleast_1d(np.load(Path(cfg.classified_result_dir) / f"{name}.npy")))
        model.cls_label = torch.tensor(ids, device="cuda", dtype=torch.int64)
        # --- diff_for_openvocabulary.py test_step, from "self.token_start_ids = []" to cam_to_label ---
        model.token_start_ids = []; model.token_sel_ids = []
        img_label = [model.class_name[i] for i in model.cls_label]
        text = "a photograph of "; last_cls = None
        for idx, cls in enumerate(img_label):
            text += cls + " and "
            model.token_start_ids.append(model.token_start_ids[idx - 1] + 1 + model.all_tokens[last_cls][0] if idx > 0 else 4)
            model.token_sel_ids.append([model.token_start_ids[-1] + sel_id for sel_id in model.all_tokens[cls][1]])
            last_cls = cls
        text = text[:-5] + " and other object and background" + model.bg_context
        with torch.no_grad():
            model.test_t_embedding = model.get_text_embedding(text)
            meaning_index = reduce(add, model.token_sel_ids)
            model.test_t_embedding[:, meaning_index] *= cfg.enhanced if cfg.no_use_cross_enh else 1
            split_mask, fam = model.get_masks(image, cfg.test_mask_size)
            fam = F.interpolate(fam[None], size=(H, W), mode="bilinear", align_corners=False)[0]
            cls_label = F.one_hot(model.cls_label, model.num_parts).sum(0)
            _, pred = cam_to_label(fam[None], cls_label=cls_label[None], bkg_thre=cfg.cam_bg_thr)
        model.stable_diffusion.feature_maps = {}; model.stable_diffusion.toq_maps = {}; model.stable_diffusion.attention_maps = {}
        lab = pred[0].cpu().numpy().astype(np.uint8)
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez_compressed(tmp, label=lab, cls_ids=np.array(ids), text=text, sec=time.time() - t0)
        os.replace(tmp, o)
        print(f"[iseg:{a.set}{'_given' if a.given else ''}] {name} labels {[names[i] for i in ids][:6]} -> {np.unique(lab).tolist()[:8]} "
              f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    for c in ("labels", "seg"):
        p = sub.add_parser(c); p.add_argument("--set", required=True, choices=list(GROUP))
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--limit", type=int, default=0); p.add_argument("--given", action="store_true")
    a = ap.parse_args(); sys.path.insert(0, str(ROOT / "scripts"))
    {"prep": prep, "labels": labels, "seg": seg}[a.cmd](a)
