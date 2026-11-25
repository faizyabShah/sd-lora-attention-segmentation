#!/usr/bin/env python3
"""Run 44: OVDiff (karazijal/ovdiff @ 218566b, ECCV 2024) as released, on run 44's real photos (env `ovdiff`).

prep   copy the repo to <WORK>/ovdiff_repo with infrastructure-only edits:
       - "runwayml/stable-diffusion-v1-5" -> the cached stable-diffusion-v1-5 snapshot (same weights, offline);
       - argparse `choices=["voc"]` -> also "voc44", "context44", "coco44" (the release only implements VOC);
       - mmdata.py: append `run44_ext` (below), which adds those keys: class lists in the release's own format
         (voc44 = the release's VOC CLASSES / CLASS_TO_PROMPT; context44 = "__background__" + the 59 PASCAL Context names
         exactly as written in the release's own thing/stuff table CHAT_GPT_REPLY_LONGER; coco44 = "__background__" + the
         80 COCO names), and a dataset over run 44's photos (image, all-zero mask: labels are never read);
       - CutLER cascade checkpoint downloaded to CutLER/ (README).
steps  the README's invocation, per column key K (`python <script> ... K <out>`):
       sample_support_set -> gen_vit_features (clip ViT-B/16 layer -2; dino_vitb8) -> gen_proto_vit (dino, clip)
       -> gen_proto_sd -> predict (--nonlocal so per-photo predictions are saved; run as a SLURM array so its built-in
       mmseg evaluation is skipped). Prediction = argmax over [background, classes] (threshold 0.0, its default).
"""
from __future__ import annotations

import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("BRACE_CACHE_DIR", str(ROOT / "outputs/sg_comp"))).expanduser()
WORK = CACHE / "jc_real44"
SRC = ROOT / "other repositories/ovdiff"
REPO = WORK / "ovdiff_repo"
SD15 = "/home/shafa/sg_comp/jc_real44/sd15_bin"   # .bin copy of the cached SD 1.5 snapshot 451f4fe (transformers 4.25 reads no safetensors; /home/shafa/sg_comp/tmp/r44/sd15_bin.py)
CUTLER = "http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth"
KEY = {"voc": "voc44", "context": "context44", "coco": "coco44"}
SCRIPTS = ["sample_support_set.py", "gen_vit_features.py", "gen_proto_vit.py", "gen_proto_sd.py", "predict.py"]

EXT = r'''

# ----------------------------------------------------------------------------------------------- run44_ext (run 44)
import json as _json, os as _os
import numpy as _np
from PIL import Image as _Image
COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"]
_CTX = [l.split(":")[0].strip() for l in CHAT_GPT_REPLY_LONGER.strip().splitlines()][2:61]
assert len(_CTX) == 59 and _CTX[0] == "airplane" and _CTX[-1] == "wood", _CTX
_R44 = {"voc44": (PascalVOCDataset.CLASSES, PascalVOCDataset.CLASS_TO_PROMPT),
        "context44": (tuple(["__background__"] + _CTX), dict({"__background__": "background"}, **{c: c for c in _CTX})),
        "coco44": (tuple(["__background__"] + COCO80), dict({"__background__": "background"}, **{c: c for c in COCO80}))}


class _Photos44:
    def __init__(self, key):
        self.items = _json.loads(open(_os.environ["RUN44_PHOTOS_" + key]).read())

    def __getitem__(self, i):
        img = _Image.open(self.items[i]).convert("RGB")
        return img, _Image.fromarray(_np.zeros(img.size[::-1], _np.uint8))

    def __len__(self):
        return len(self.items)


def _overlay(r, apply_overlay):
    if not apply_overlay:
        return r
    return r[0], {k: CLASS_TO_PROMPT_OVERLAY.get(k, v) for k, v in r[1].items()}


_orig_get_dataset, _orig_get_data_classes = get_dataset, get_data_classes


def get_data_classes(dataset, apply_overlay=False):
    if dataset in _R44:
        return _overlay(_R44[dataset], apply_overlay)
    return _orig_get_data_classes(dataset, apply_overlay)


def get_dataset(dataset, apply_overlay=False):
    if dataset in _R44:
        c, p = _overlay(_R44[dataset], apply_overlay)
        ds = _Photos44(dataset)

        class _W:
            dataset = ds
            def __getitem__(self, i): return ds[i]
            def __len__(self): return len(ds)
        return _W(), c, p
    return _orig_get_dataset(dataset, apply_overlay)
'''


def photos(s):
    it = json.loads((WORK / "items.json").read_text())[s]; P = {}
    for x in it:
        if x["primary"]:
            P[str(x["img"])] = str(ROOT / "datasets/coco2017" / x["path"]) if s == "coco" else str(ROOT / x["path"])
    return sorted(P.items())


def prep(a):
    if not REPO.exists():
        tmp = WORK / "ovdiff_repo.tmp"; shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(SRC, tmp, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        for f in list(tmp.glob("*.py")) + list((tmp / "utils").glob("*.py")):
            s = f.read_text(); s2 = s.replace("runwayml/stable-diffusion-v1-5", SD15)
            if f.name in SCRIPTS:
                n = s2.count('choices=["voc"]'); assert n == 1, (f, n)
                s2 = s2.replace('choices=["voc"]', 'choices=["voc", "voc44", "context44", "coco44"]')
            if s2 != s:
                f.write_text(s2)
        m = tmp / "mmdata.py"; m.write_text(m.read_text() + EXT)
        subprocess.run(["wget", "-q", "-c", "-P", str(tmp / "CutLER"), CUTLER], check=True)
        os.replace(tmp, REPO); print("prepared", REPO)
    txt = (SRC / "mmdata.py").read_text()                                     # class lists for scoring (by name)
    voc = re.findall(r'"([^"]+)",', re.sub(r"#.*", "", txt[txt.index("CLASSES = (") : txt.index("PALETTE")]))
    blk = txt[txt.index('CHAT_GPT_REPLY_LONGER = """') + 27:]; blk = blk[: blk.index('"""')]
    ctx = [l.split(":")[0].strip() for l in blk.strip().splitlines()][2:61]
    coco = re.findall(r'"([^"]+)"', EXT[EXT.index("COCO80 = [") : EXT.index("_CTX =")])
    assert len(voc) == 21 and len(ctx) == 59 and len(coco) == 80, (len(voc), len(ctx), len(coco))
    (WORK / "ovdiff").mkdir(parents=True, exist_ok=True)
    for k, L in (("voc44", voc), ("context44", ["__background__"] + ctx), ("coco44", ["__background__"] + coco)):
        (WORK / "ovdiff" / f"classes_{k}.json").write_text(json.dumps(L))
    for s, k in KEY.items():
        P = photos(s); d = WORK / "ovdiff"; d.mkdir(parents=True, exist_ok=True)
        (d / f"photos_{k}.json").write_text(json.dumps([p for _, p in P]))
        (d / f"ids_{k}.json").write_text(json.dumps([i for i, _ in P]))
        print(k, len(P), "photos")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    a = ap.parse_args(); {"prep": prep}[a.cmd](a)
