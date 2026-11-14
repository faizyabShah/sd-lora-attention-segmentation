#!/usr/bin/env python3
"""Training-free dense CLIP (SCLIP port) for the real-photo track (exploratory; does not touch BRACE).

Port of SCLIP (Wang et al., ECCV 2024; github.com/wangf3014/SCLIP, main branch read 2026-11-08) onto
HF transformers CLIP, without mmseg:
  * last vision block: correlative self-attention  softmax(q q^T s) + softmax(k k^T s)  (s = head_dim^-0.5),
    residual and MLP kept, as in the repo's VisionTransformer.forward; then post-LN and projection on all tokens;
  * text: 80 OpenAI ImageNet templates per name, each L2-normalised, averaged, re-normalised;
    class-name files cls_voc20.txt / cls_voc21.txt copied verbatim (synonyms max-pooled after softmax);
  * inference: resize keep-ratio to short side 336 (long <= 2048), sliding 224 crops, stride 112,
    CLIP mean/std, patch logits bilinear to the crop, averaged, bilinear to the original size;
    logit_scale 40 (VOC20) / 65 with prob_thd 0.1 and area_thd 0.1 (VOC21), no PAMR (repo defaults).

faith:   VOC2012 val (1449), VOC20 (bg ignored) and VOC21 mIoU, dataset-level confusion like mmseg IoUMetric.
         Repo reference with ViT-B/16: VOC20 81.54, VOC21 59.63.
capture: the 200 real D photos. Per photo and backbone: raw cosine logits of every query (VOC20 names incl.
         synonyms, the 26 VOC21 background names, the 8 generic BRACE words) averaged to 64x64 (the SD maps cover
         the whole photo stretched to 512, so 64x64 cells align), the VOC21 standalone prediction at full size,
         and per-class VOC20/VOC21 probabilities at 64x64. Label-free; masks are only read by later scoring.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
from brace_paths import CACHE  # noqa: E402

BACKBONES = {"B16": "openai/clip-vit-base-patch16", "L14": "openai/clip-vit-large-patch14"}
GENERIC = ["wall", "floor", "ground", "sky", "grass", "tree", "building", "road"]
VOC20_NAMES = ["aeroplane", "bicycle", "bird", "ship", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog",
               "horse", "motorbike",
               "person, person in shirt, person in jeans, person in dress, person in sweater, person in skirt, person in jacket",
               "pottedplant", "sheep", "sofa", "train",
               "television monitor, tv monitor, monitor, television, screen"]
VOC21_BG = ("sky, wall, tree, wood, grass, road, sea, river, mountain, sands, desk, bed, building, cloud, lamp, door, "
            "window, wardrobe, ceiling, shelf, curtain, stair, floor, hill, rail, fence")
MEAN = torch.tensor([122.771, 116.746, 104.094]).view(1, 3, 1, 1)
STD = torch.tensor([68.501, 66.632, 70.323]).view(1, 3, 1, 1)
TEMPLATES = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.',
    'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.',
    'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
    'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.',
    'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.',
    'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.',
    'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
    'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.',
    'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.',
    'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.',
    'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
    'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.',
    'a photo of a small {}.', 'a tattoo of the {}.']
assert len(TEMPLATES) == 80


def split(lines):
    names, idx = [], []
    for i, line in enumerate(lines):
        for n in line.split(", "):
            names.append(n.strip()); idx.append(i)
    return names, np.array(idx)


class DenseCLIP:
    def __init__(self, name, device):
        from transformers import CLIPModel, CLIPTokenizer
        self.m = CLIPModel.from_pretrained(BACKBONES[name]).to(device).eval()
        self.tok = CLIPTokenizer.from_pretrained(BACKBONES[name])
        self.dev = device
        self.patch = self.m.vision_model.config.patch_size
        self.crop = self.m.vision_model.config.image_size          # 224 for both backbones
        self._text = {}

    @torch.inference_mode()
    def text(self, names):
        out = []
        for n in names:
            if n not in self._text:
                t = self.tok([tp.format(n) for tp in TEMPLATES], padding="max_length", max_length=77,
                             truncation=True, return_tensors="pt").to(self.dev)
                f = self.m.get_text_features(**t); f = f / f.norm(dim=-1, keepdim=True)
                f = f.mean(0); self._text[n] = f / f.norm()
            out.append(self._text[n])
        return torch.stack(out)

    @torch.inference_mode()
    def patch_features(self, x):
        vm = self.m.vision_model
        h = vm.pre_layrnorm(vm.embeddings(x))
        for layer in vm.encoder.layers[:-1]:
            h = layer(h, None, None)[0]
        L = vm.encoder.layers[-1]; att = L.self_attn
        y = L.layer_norm1(h); B, N, D = y.shape; H = att.num_heads; hd = D // H; s = hd ** -0.5
        q = att.q_proj(y).view(B, N, H, hd).transpose(1, 2)
        k = att.k_proj(y).view(B, N, H, hd).transpose(1, 2)
        v = att.v_proj(y).view(B, N, H, hd).transpose(1, 2)
        w = (q @ q.transpose(-1, -2) * s).softmax(-1) + (k @ k.transpose(-1, -2) * s).softmax(-1)
        o = att.out_proj((w @ v).transpose(1, 2).reshape(B, N, D))
        h = h + o
        h = h + L.mlp(L.layer_norm2(h))
        f = self.m.visual_projection(vm.post_layernorm(h))[:, 1:]
        return f / f.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def logits(self, img, T):
        """img: HxWx3 uint8 RGB -> raw cosine logits [Q, H, W] at the original size (SCLIP forward_slide)."""
        H0, W0 = img.shape[:2]
        sf = min(2048 / max(H0, W0), 336 / min(H0, W0))
        h1, w1 = int(H0 * sf + 0.5), int(W0 * sf + 0.5)
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(h1, w1), mode="bilinear", align_corners=False)
        x = ((x - MEAN) / STD).to(self.dev)
        c, sd = self.crop, 112
        preds = torch.zeros((T.shape[0], h1, w1), device=self.dev); cnt = torch.zeros((1, h1, w1), device=self.dev)
        hg = max(h1 - c + sd - 1, 0) // sd + 1; wg = max(w1 - c + sd - 1, 0) // sd + 1
        for i in range(hg):
            for j in range(wg):
                y2 = min(i * sd + c, h1); x2 = min(j * sd + c, w1); y1 = max(y2 - c, 0); x1 = max(x2 - c, 0)
                cr = x[:, :, y1:y2, x1:x2]
                if cr.shape[-2:] != (c, c):
                    raise ValueError("crop smaller than the native input; positional interpolation not ported")
                f = self.patch_features(cr)[0] @ T.T                       # [g*g, Q]
                g = c // self.patch
                lg = f.T.reshape(1, -1, g, g)
                preds[:, y1:y2, x1:x2] += F.interpolate(lg, size=(y2 - y1, x2 - x1), mode="bilinear")[0]
                cnt[:, y1:y2, x1:x2] += 1
        preds = preds / cnt
        return F.interpolate(preds[None], size=(H0, W0), mode="bilinear")[0]


def postprocess(lg, idx, scale, prob_thd=0.0, area_thd=None):
    """SCLIP postprocess_result: softmax over queries, max over synonyms, area/prob thresholds -> (pred, prob)."""
    p = (lg * scale).softmax(0)
    ncls = int(idx.max()) + 1
    if ncls != len(idx):
        I = torch.as_tensor(idx, device=p.device)
        p = torch.stack([p[I == c].max(0).values for c in range(ncls)])
    if area_thd is not None:
        oh = F.one_hot(p.argmax(0), ncls).to(p.dtype)
        area = oh[:, :, 1:].sum((0, 1))
        keep = (area > area_thd * area.sum()).to(p.dtype)
        p = torch.cat([p[:1], p[1:] * keep[:, None, None]])
    pred = p.argmax(0)
    pred[p.max(0).values < prob_thd] = 0
    return pred, p


def faith(a):
    dev = torch.device("cuda")
    ids = (st.VOC / "ImageSets/Segmentation/val.txt").read_text().split()
    if a.max_images:
        ids = ids[: a.max_images]
    res = {}
    for bb in a.backbones:
        M = DenseCLIP(bb, dev)
        n20, i20 = split(VOC20_NAMES); n21, i21 = split([VOC21_BG] + VOC20_NAMES)
        T20, T21 = M.text(n20), M.text(n21)
        C20 = np.zeros((20, 20), np.int64); C21 = np.zeros((21, 21), np.int64); t0 = time.perf_counter()
        for n, iid in enumerate(ids):
            img = np.asarray(Image.open(st.VOC / f"JPEGImages/{iid}.jpg").convert("RGB"))
            lab = np.asarray(Image.open(st.VOC / f"SegmentationClass/{iid}.png")).astype(np.int64)
            lg20 = M.logits(img, T20); p20, _ = postprocess(lg20, i20, 40)
            lg21 = M.logits(img, T21); p21, _ = postprocess(lg21, i21, 65, 0.1, 0.1)
            p20 = p20.cpu().numpy(); p21 = p21.cpu().numpy()
            v = (lab != 255) & (lab != 0)
            C20 += np.bincount((lab[v] - 1) * 20 + p20[v], minlength=400).reshape(20, 20)
            v = lab != 255
            C21 += np.bincount(lab[v] * 21 + p21[v], minlength=441).reshape(21, 21)
            if (n + 1) % 200 == 0:
                print(f"[faith {bb}] {n + 1}/{len(ids)} {(time.perf_counter() - t0) / (n + 1):.2f}s/img", flush=True)
        iou = lambda C: np.diag(C) / np.maximum(C.sum(0) + C.sum(1) - np.diag(C), 1)
        res[bb] = {"VOC20": float(iou(C20).mean() * 100), "VOC21": float(iou(C21).mean() * 100),
                   "VOC20_per_class": (iou(C20) * 100).round(1).tolist(), "n": len(ids),
                   "s_per_img_both_protocols": (time.perf_counter() - t0) / len(ids)}
        print(f"[faith {bb}] VOC20 {res[bb]['VOC20']:.2f} (repo B16 81.54)  VOC21 {res[bb]['VOC21']:.2f} "
              f"(repo B16 59.63)", flush=True)
        del M; torch.cuda.empty_cache()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


def capture(a):
    dev = torch.device("cuda")
    photos = sorted({e["image"] for e in json.loads(st.MANIFEST.read_text())[a.set]})
    if a.max_images:
        photos = photos[: a.max_images]
    a.out.mkdir(parents=True, exist_ok=True)
    n20, i20 = split(VOC20_NAMES); n21, i21 = split([VOC21_BG] + VOC20_NAMES)
    queries = list(dict.fromkeys(n21 + GENERIC))
    for bb in a.backbones:
        M = DenseCLIP(bb, dev); T = M.text(queries); qi = {q: i for i, q in enumerate(queries)}
        t0 = time.perf_counter(); done = 0
        for iid in photos:
            out = a.out / f"{iid}__{bb}.npz"
            if out.exists():
                continue
            img = np.asarray(Image.open(st.VOC / f"JPEGImages/{iid}.jpg").convert("RGB"))
            lg = M.logits(img, T)                                           # [Q, H, W] raw cosine
            pred21, prob21 = postprocess(lg[[qi[q] for q in n21]], i21, 65, 0.1, 0.1)
            _, prob20 = postprocess(lg[[qi[q] for q in n20]], i20, 40)
            to64 = lambda x: F.adaptive_avg_pool2d(x[None].float(), (64, 64))[0].cpu().numpy().astype(np.float32)
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez(tmp, queries=np.array(queries), logit64=to64(lg), prob20_64=to64(prob20),
                     prob21_64=to64(prob21), pred21=pred21.cpu().numpy().astype(np.uint8),
                     classes20=np.array([n.split(",")[0] for n in VOC20_NAMES]), backbone=bb)
            os.replace(tmp, out); done += 1
            if done <= 3 or done % 50 == 0:
                print(f"[capture {bb}] {iid} {(time.perf_counter() - t0) / done:.2f}s/photo", flush=True)
        del M; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("faith"); p.add_argument("--backbones", nargs="+", default=["B16", "L14"])
    p.add_argument("--max-images", type=int)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/segmentation/sg_real_atlas/analysis/sclip_faith.json")
    p = sub.add_parser("capture"); p.add_argument("--set", default="D"); p.add_argument("--max-images", type=int)
    p.add_argument("--backbones", nargs="+", default=["B16", "L14"])
    p.add_argument("--out", type=Path, default=CACHE / "clip_D")
    a = ap.parse_args()
    {"faith": faith, "capture": capture}[a.cmd](a)
