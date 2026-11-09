#!/usr/bin/env python3
"""Standard-protocol VOC2012 val (1449) for SCLIP + SD self-attention propagation (real-photo track, 2026-11-09).

Frozen pipeline (pre-registered in outputs/segmentation/sg_real_atlas/analysis/voc21_prereg.md before any val
capture). Label-free: no class-presence list, all 20 classes (+background for VOC21) compete in every photo.
  1. SCLIP ViT-B/16 (scripts/sg_clip_dense.py port; reproduces the repo on VOC val): raw logits, softmax at the
     repo's logit scale (65 VOC21 / 40 VOC20), synonyms max-pooled -> per-class probabilities p [C, H, W].
  2. One SD1.5 run per photo: empty-prompt DDIM inversion (30 steps), then the frozen live pipeline on that
     trajectory with the prompt "a photograph of a {w}", w = the foreground class with the largest total SCLIP
     VOC21 probability (label-free; the SA graphs are what is used, see the D prompt-dependence check).
  3. C16 propagation of every class map at once (identical to sg27_fusion.base_paths C16 per channel):
     p64 = area-pool(p, 64); C16 = mean over 11 calls of  sum_h A64_h · up16→64( mean_h A16_h · down64→16(p64) ) / 8.
     Attention rows sum to 1, so propagated maps stay on SCLIP's probability scale.
  4. Bilinear back to the photo size, then SCLIP's own post-processing unchanged: VOC21 area_thd 0.1 and
     prob_thd 0.1; VOC20 argmax.
Comparators under the same protocol: SCLIP standalone, SCLIP + DenseCRF (multi-label, the R2 parameters
(10, 80, 13, 3, 3, 5) on the area-thresholded, renormalised probabilities); secondary: C16 + DenseCRF.

capture (GPU): per photo npz with full-size uint8 predictions of every pipeline and the 64x64 maps (fp16).
score  (CPU): dataset-level confusion (mmseg IoUMetric style), VOC21/VOC20 mIoU, per class, photo-bootstrap CIs.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_clip_dense as cd  # noqa: E402

CRF_PARAMS = (10.0, 80, 13, 3.0, 3, 5.0)   # = ovam_full.OVAM_CRF_PARAMETERS (R2)
PIPES = ("sclip", "sclip_crf", "c16", "c16_crf")


def probs(lg, idx, scale):
    """SCLIP postprocess up to (excluding) the thresholds: softmax over queries, max over synonyms."""
    p = (lg * scale).softmax(0)
    I = torch.as_tensor(idx, device=p.device)
    return torch.stack([p[I == c].max(0).values for c in range(int(idx.max()) + 1)])


def thresholds(p, prob_thd, area_thd):
    """The rest of SCLIP's postprocess_result, applied to (possibly propagated) probabilities."""
    ncls = p.shape[0]
    if area_thd is not None:
        oh = F.one_hot(p.argmax(0), ncls).to(p.dtype)
        area = oh[:, :, 1:].sum((0, 1))
        keep = (area > area_thd * area.sum()).to(p.dtype)
        p = torch.cat([p[:1], p[1:] * keep[:, None, None]])
    pred = p.argmax(0)
    pred[p.max(0).values < prob_thd] = 0
    return pred, p


def crf(img, p):
    from src.segmentation.ovam_full import _import_densecrf
    dc = _import_densecrf(st.CRF_SITE)
    q = p.clamp_min(1e-8); q = (q / q.sum(0, keepdim=True)).permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
    return np.asarray(dc.densecrf(img, q, CRF_PARAMS)).astype(np.uint8)


def c16_multi(sa, P):
    """P [C,64,64] -> C16 of every channel [C,64,64]; equals sg27_fusion.base_paths(sa, P[c])['C16']."""
    from src.segmentation.sg25_sa import _attention, _resize
    C = P.shape[0]; c16 = _resize(P.reshape(C, -1).float(), 64, 16)
    out = torch.zeros(C, 4096, device=P.device)
    for call in range(11):
        a16 = _attention(sa[16][call], 16)
        co = torch.einsum("hij,cj->hci", a16, c16).mean(0); del a16
        a64 = _attention(sa[64][call], 64)
        out += torch.einsum("hij,cj->ci", a64, _resize(co, 16, 64)); del a64
    return (out / 88).reshape(C, 64, 64)


def ids_of(split):
    if split == "val":
        return (st.VOC / "ImageSets/Segmentation/val.txt").read_text().split()
    return sorted({e["image"] for e in json.loads(st.MANIFEST.read_text())[split]})


def capture(a):
    import run_sg_real_atlas as ra
    from scripts import sg27_runtime
    from src.segmentation import sg27_fusion
    from sg_class_comp import voc_classes
    dev = torch.device("cuda")
    ids = ids_of(a.split)[a.shard::a.num_shards]
    if a.max_images:
        ids = ids[: a.max_images]
    a.out.mkdir(parents=True, exist_ok=True)
    M = cd.DenseCLIP("B16", dev)
    n20, i20 = cd.split(cd.VOC20_NAMES); n21, i21 = cd.split([cd.VOC21_BG] + cd.VOC20_NAMES)
    T20, T21 = M.text(n20), M.text(n21)
    runtime = sg27_runtime.make_runtime(a.model); pipe = runtime["pipe"]
    pipe.scheduler.set_timesteps(30, device=pipe.device)
    ts_unique = sorted(set(pipe.scheduler.timesteps.tolist()))
    empty = pipe.text_encoder(pipe.tokenizer([""], padding="max_length", max_length=77,
                                             return_tensors="pt").input_ids.to(pipe.device))[0].to(pipe.unet.dtype)
    words = voc_classes()
    scratch = a.out / f".gt_{a.split}_{a.shard}.png"
    t_all = time.perf_counter(); done = 0
    for iid in ids:
        out = a.out / f"{iid}.npz"
        if out.exists():
            continue
        t0 = time.perf_counter()
        img = np.asarray(Image.open(st.VOC / f"JPEGImages/{iid}.jpg").convert("RGB")); H, W = img.shape[:2]
        p21 = probs(M.logits(img, T21), i21, 65); p20 = probs(M.logits(img, T20), i20, 40)
        sclip21, p21t = thresholds(p21, 0.1, 0.1); sclip20 = p20.argmax(0)
        t_clip = time.perf_counter() - t0
        k = int(p21[1:].sum((1, 2)).argmax()) + 1; w = words[k - 1]
        Image.fromarray(np.zeros((H, W), np.uint8)).save(scratch)          # only satisfies the row loader
        z0 = ra.encode(pipe, ra.to512(img))
        table, _ = ra.ddim_invert(pipe, z0, ts_unique, empty)
        prompt = f"a photograph of a {w}"
        row = {"image_id": f"{iid}/{w}", "classname": w, "prompt": prompt, "attribution_prompt": prompt,
               "seed": "0", "num_inferece_steps": "30", "annotation": str(scratch)}
        P64 = torch.cat([F.adaptive_avg_pool2d(p21[None], (64, 64))[0], F.adaptive_avg_pool2d(p20[None], (64, 64))[0]])
        with ra.trajectory(ra.table_fn(table)), sg27_runtime.live_maps(runtime, row, "voc") as (_s, sa, _seed, _b, _p):
            prop = c16_multi(sa, P64.to(dev))
            if done == 0:                                                   # replay vs the frozen per-seed path
                ref = sg27_fusion.base_paths(sa, P64[k].to(dev))["C16"].reshape(64, 64)
                rel = float((ref - prop[k]).abs().max() / ref.abs().max())
                print(f"[voc21] replay C16 multi vs base_paths rel err {rel:.2e}", flush=True)
                assert rel < 1e-4, rel
        t_sd = time.perf_counter() - t0 - t_clip
        up = F.interpolate(prop[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        c16_21, q21t = thresholds(up[:21], 0.1, 0.1); c16_20 = up[21:].argmax(0)
        preds = {"sclip21": sclip21, "c16_21": c16_21, "sclip20": sclip20, "c16_20": c16_20}
        preds = {n: v.cpu().numpy().astype(np.uint8) for n, v in preds.items()}
        preds["sclip_crf21"] = crf(img, p21t); preds["c16_crf21"] = crf(img, q21t)
        preds["sclip_crf20"] = crf(img, p20); preds["c16_crf20"] = crf(img, up[21:])
        tmp = out.with_name(out.stem + ".tmp.npz")
        extra = {"p21_full": p21.cpu().numpy().astype(np.float16), "p20_full": p20.cpu().numpy().astype(np.float16)} if a.save_full else {}
        np.savez_compressed(tmp, **{f"pred/{n}": v for n, v in preds.items()}, **extra,
                            p64=P64.cpu().numpy().astype(np.float16), prop64=prop.cpu().numpy().astype(np.float16),
                            prompt_word=w, t_clip=t_clip, t_sd=t_sd, t_total=time.perf_counter() - t0)
        os.replace(tmp, out); done += 1
        del table, prop, up; torch.cuda.empty_cache()
        print(f"[voc21:{a.split}] {iid} '{w}' clip {t_clip:.1f}s sd {t_sd:.1f}s total {time.perf_counter() - t0:.1f}s "
              f"(avg {(time.perf_counter() - t_all) / done:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)",
              flush=True)


NAMES21 = ["background"] + [n.split(",")[0] for n in cd.VOC20_NAMES]


def score(a):
    ids = ids_of(a.split)
    files = [a.maps / f"{i}.npz" for i in ids]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} photos missing, e.g. {missing[0]}")
    keys = [f"{p}{n}" for n in ("21", "20") for p in ("sclip", "sclip_crf", "c16", "c16_crf")]
    Cs = {k: np.zeros((len(ids), 21, 21), np.int64) for k in keys}; t = []; words = []
    for n, (iid, f) in enumerate(zip(ids, files)):
        z = np.load(f); lab = np.asarray(Image.open(st.VOC / f"SegmentationClass/{iid}.png")).astype(np.int64)
        t.append(float(z["t_total"])); words.append(str(z["prompt_word"]))
        for k in keys:
            pr = z[f"pred/{k}"].astype(np.int64)
            if k.endswith("20"):
                v = (lab != 255) & (lab != 0); g, p = lab[v], pr[v] + 1     # 20-way: classes 1..20
            else:
                v = lab != 255; g, p = lab[v], pr[v]
            Cs[k][n] = np.bincount(g * 21 + p, minlength=441).reshape(21, 21)
    def miou(C, k):
        C = C.sum(0) if C.ndim == 3 else C
        iou = np.diag(C) / np.maximum(C.sum(0) + C.sum(1) - np.diag(C), 1)
        return iou[1:] if k.endswith("20") else iou
    rng = np.random.default_rng(0); B = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
    def boot(ka, kb):
        d = [miou(Cs[ka][b], ka).mean() - miou(Cs[kb][b], kb).mean() for b in B]
        return np.quantile(d, [.025, .975]) * 100
    res = {k: float(miou(Cs[k], k).mean() * 100) for k in keys}
    L = []; w = L.append
    w(f"# VOC2012 val ({len(ids)} photos), standard protocol: SCLIP ViT-B/16 + SD1.5 C16 propagation\n")
    w("Pre-registered in `voc21_prereg.md` (committed before the val capture). Dataset-level confusion (mmseg "
      "IoUMetric style); VOC21 = 21 classes incl. background; VOC20 = 20 classes, background pixels ignored. "
      "No class-presence information is used; nothing was tuned on val.\n")
    w("| pipeline | VOC21 mIoU | VOC20 mIoU |\n|---|---:|---:|")
    for p, lbl in (("sclip", "SCLIP standalone (repo B16: 59.63 / 81.54)"), ("sclip_crf", "SCLIP + DenseCRF"),
                   ("c16", "**SCLIP → SD C16** (primary)"), ("c16_crf", "SCLIP → SD C16 + DenseCRF")):
        w(f"| {lbl} | {res[p + '21']:.2f} | {res[p + '20']:.2f} |")
    w("\n## Registered comparisons (photo-bootstrap 95% CI, 1000 resamples)\n")
    w("| comparison | Δ VOC21 | CI | Δ VOC20 | CI |\n|---|---:|---|---:|---|")
    for x, y in (("c16", "sclip"), ("c16", "sclip_crf"), ("c16_crf", "sclip_crf")):
        a21 = boot(x + "21", y + "21"); a20 = boot(x + "20", y + "20")
        w(f"| {x} − {y} | {res[x + '21'] - res[y + '21']:+.2f} | {a21[0]:+.2f}…{a21[1]:+.2f} | "
          f"{res[x + '20'] - res[y + '20']:+.2f} | {a20[0]:+.2f}…{a20[1]:+.2f} |")
    w("\n## Per-class VOC21 IoU (SCLIP / SCLIP+CRF / C16 / C16+CRF)\n")
    I = {k: miou(Cs[k], k) * 100 for k in keys}
    w("| class | SCLIP | SCLIP+CRF | C16 | C16+CRF |\n|---|---:|---:|---:|---:|")
    for c in range(21):
        w(f"| {NAMES21[c]} | {I['sclip21'][c]:.1f} | {I['sclip_crf21'][c]:.1f} | {I['c16_21'][c]:.1f} | {I['c16_crf21'][c]:.1f} |")
    w("\n## Background behaviour (VOC21, dataset level)\n")
    for k in ("sclip21", "c16_21"):
        C = Cs[k].sum(0); g_bg = C[0].sum(); g_fg = C[1:].sum()
        w(f"- {k}: background pixels predicted as an object {C[0, 1:].sum() / g_bg * 100:.1f}%; object pixels "
          f"predicted as background {C[1:, 0].sum() / g_fg * 100:.1f}%; object pixels given the wrong object "
          f"{(C[1:, 1:].sum() - np.trace(C[1:, 1:])) / g_fg * 100:.1f}%")
    w(f"\nRuntime: {np.mean(t):.1f} s/photo mean (median {np.median(t):.1f}) on the GPU node, incl. SCLIP, one DDIM "
      f"inversion + one SD run, propagation and CRF. Prompt word = SCLIP's top class.")
    a.out.write_text("\n".join(L) + "\n"); print("\n".join(L))
    (a.out.with_suffix(".json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--split", default="val"); p.add_argument("--max-images", type=int)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--save-full", action="store_true", help="also store full-size SCLIP probabilities (dev sets only)")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5"); p.add_argument("--out", type=Path, default=Path("/home/shafa/sg_comp/voc21_val"))
    p = sub.add_parser("score"); p.add_argument("--split", default="val")
    p.add_argument("--maps", type=Path, default=Path("/home/shafa/sg_comp/voc21_val"))
    p.add_argument("--out", type=Path, default=ROOT / "outputs/segmentation/sg_real_atlas/analysis/voc21_val.md")
    a = ap.parse_args()
    {"capture": capture, "score": score}[a.cmd](a)
