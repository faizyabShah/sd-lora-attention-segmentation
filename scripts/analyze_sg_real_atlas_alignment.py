#!/usr/bin/env python3
"""SG-real-atlas analyses 2-4 (Section 137), per pair, exploratory on D. Diagnostics only: the real mask is
used to *measure* the stored attention, never to build a prediction.

Per pair and per condition (renoise/n0, ddim):
  CA atlas   AUROC against the mask of every CA module x call x head for four signals of the class token:
             prob = exp(logit - lse), logit, cos = logit / (scale |q| |k|), |q|; R-precision of prob; value
             norms |v W_o| per head; AUROC of every prompt token's head-mean probability (SOT ... pads).
  SA atlas   GT-mode keep / leak per SA module x call x head for exact / eroded / dilated masks (stored by the
             capture) plus the mask area at each side, so lifts over chance can be formed.
  Diffusion  AUROC and AP (vs the 64 mask) of y_t = A_s^t seed_s, t = 0..8, for s = 16, 32, 64, with A_s the
             head-mean SA operator of each of the 11 BRACE calls (maps averaged over calls), plus the per-head
             AUROC at t = 2; and the composite C16 path A64 up(A16^t c16) for t = 1..4.
Error breakdown (renoise/n0 V0+R2 @0.65, ddim C16 @0.675, ddim V0 @0.625 -- the OOF taus of analysis 1):
  FP pixels split into other-object / background within 8 px of the object / far background; FN pixels
  into the 8 px boundary band vs interior; per-instance coverage (VOC SegmentationObject) by instance size;
  FP pixels per other VOC class.
Usage: analyze_sg_real_atlas_alignment.py CAPTURE_DIR OUT_DIR [--set D] [--workers 12]
"""
import argparse, json, math, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VOC = ROOT / "datasets/voc2012/VOCdevkit/VOC2012"
CRF_SITE = ROOT / "outputs/dependencies/simplecrf_py310"
SIDE = {"down_blocks.0": 64, "down_blocks.1": 32, "down_blocks.2": 16, "mid_block": 8,
        "up_blocks.1": 16, "up_blocks.2": 32, "up_blocks.3": 64}
HEAD_DIM = {64: 40, 32: 80, 16: 160, 8: 160}
CONDS = ("renoise/n0", "ddim")
ROLES = ("SOT", "a", "photograph", "of", "a2", "cls", "EOT", "pad1", "pad2", "pad3", "pad4")
TMAX = 8
ERR = (("n0_V0R2", "renoise/n0", "V0", 0.65, True), ("ddim_C16", "ddim", "C16", 0.675, False),
       ("ddim_V0", "ddim", "V0", 0.625, False))
BAND = 8


def side_of(name):
    return next(v for k, v in SIDE.items() if name.startswith(k))


def down(mask, side):
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(t, size=(side, side), mode="area")[0, 0].numpy().ravel()


def auroc(X, y):
    """X [B, N] scores, y [N] bool -> [B] AUROC (average ranks for ties)."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(len(X), np.nan, np.float32)
    r = rankdata(X, axis=1)
    return ((r[:, y].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)).astype(np.float32)


def ap(X, y):
    n1 = int(y.sum())
    if n1 == 0 or n1 == len(y):
        return np.full(len(X), np.nan, np.float32)
    o = np.argsort(-X, axis=1); hits = y[o].astype(np.float64)
    prec = np.cumsum(hits, 1) / np.arange(1, X.shape[1] + 1)
    return ((prec * hits).sum(1) / n1).astype(np.float32)


def rprec(X, y):
    n1 = int(y.sum())
    if n1 == 0:
        return np.full(len(X), np.nan, np.float32)
    top = np.argpartition(-X, n1 - 1, axis=1)[:, :n1]
    return y[top].mean(1).astype(np.float32)


def gt_at(truth, valid, side):
    t, v = down(truth & valid, side), down(valid, side)
    keep = v >= .5
    return (t / np.maximum(v, 1e-6) >= .5), keep


def ca_atlas(atlas, n_all, truth, valid):
    mods = sorted({k.split("/", 2)[2] for k in atlas.files if k.startswith("ca/logit/")})
    auc = np.full((4, len(mods), 31, 8), np.nan, np.float32); rp = np.full((len(mods), 31, 8), np.nan, np.float32)
    vn = np.zeros((len(mods), 31, 8), np.float32); kn = np.zeros((len(mods), 31, 8), np.float32)
    tok = np.full((len(mods), 31, len(ROLES)), np.nan, np.float32)
    for i, m in enumerate(mods):
        s = side_of(m); y, keep = gt_at(truth, valid, s); y = y[keep]
        lg = atlas[f"ca/logit/{m}"].astype(np.float32)                  # [31,T,8,N]
        lse = atlas[f"ca/lse/{m}"].astype(np.float32)                   # [31,8,N]
        qn = atlas[f"ca/qnorm/{m}"].astype(np.float32)
        knm = atlas[f"ca/knorm/{m}"].astype(np.float32)                 # [31,T,8]
        vnm = atlas[f"ca/vnorm/{m}"].astype(np.float32)
        prob = np.exp(lg - lse[:, None]).sum(1)                          # [31,8,N]
        cos = (lg / (qn[:, None] * knm[..., None] / math.sqrt(HEAD_DIM[s]) + 1e-8)).mean(1)
        sig = np.stack([prob, lg.mean(1), cos, qn])[..., keep]           # [4,31,8,N']
        auc[:, i] = auroc(sig.reshape(-1, sig.shape[-1]), y).reshape(4, 31, 8)
        rp[i] = rprec(sig[0].reshape(-1, sig.shape[-1]), y).reshape(31, 8)
        vn[i], kn[i] = vnm.mean(1), knm.mean(1)
        tk = atlas[f"tok/{m.replace('attn2', 'attn2')}"].astype(np.float32)[..., keep]   # [31,n,N']
        rows = {"SOT": tk[:, 0], "a": tk[:, 1], "photograph": tk[:, 2], "of": tk[:, 3], "a2": tk[:, 4],
                "cls": tk[:, 5:n_all - 1].sum(1), "EOT": tk[:, n_all - 1]}
        for j in range(4):
            if n_all + j < tk.shape[1]:
                rows[f"pad{j + 1}"] = tk[:, n_all + j]
        for j, r in enumerate(ROLES):
            if r in rows:
                tok[i, :, j] = auroc(rows[r], y)
    return mods, auc, rp, vn, kn, tok


def attention(q, k):
    q, k = torch.from_numpy(q).float(), torch.from_numpy(k).float()
    return torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1]), -1)


def resize(v, s, t):
    mode = "bilinear" if t > s else "area"
    kw = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(v.reshape(1, 1, s, s), size=(t, t), mode=mode, **kw).reshape(-1)


def diffusion(qk, seed64, y64, keep64):
    seed = torch.from_numpy(seed64).float().reshape(-1)
    curves, heads2, c16p = {}, {}, np.zeros((4, 2), np.float32)
    acc = {s: torch.zeros(TMAX + 1, 4096) for s in (16, 32, 64)}
    acc_h = {s: torch.zeros(8, 4096) for s in (16, 64)}
    acc_c = torch.zeros(4, 4096)
    for c in range(11):
        A = {}
        for s in (16, 32, 64):
            a = attention(qk[f"s{s}"][c, 0], qk[f"s{s}"][c, 1]); A[s] = a
            P = a.mean(0); y = resize(seed, 64, s) if s < 64 else seed.clone()
            for t in range(TMAX + 1):
                acc[s][t] += (resize(y, s, 64) if s < 64 else y) / 11
                y = P @ y
            if s in acc_h:
                y0 = resize(seed, 64, s) if s < 64 else seed
                y2 = torch.bmm(a, torch.bmm(a, y0[None].expand(8, -1).unsqueeze(-1))).squeeze(-1)
                acc_h[s] += torch.stack([resize(v, s, 64) if s < 64 else v for v in y2]) / 11
        P16, P64 = A[16].mean(0), A[64].mean(0); y = resize(seed, 64, 16)
        for t in range(4):
            y = P16 @ y; acc_c[t] += (P64 @ resize(y, 16, 64)) / 11
        del A
    yk = y64[keep64]
    for s in (16, 32, 64):
        X = acc[s].numpy()[:, keep64]; curves[s] = np.stack([auroc(X, yk), ap(X, yk)])       # [2, T+1]
    for s in (16, 64):
        heads2[s] = auroc(acc_h[s].numpy()[:, keep64], yk)
    X = acc_c.numpy()[:, keep64]; c16p = np.stack([auroc(X, yk), ap(X, yk)])
    return curves, heads2, c16p


def mm(x):
    return (x - x.min()) / max(x.max() - x.min(), 1e-12)


def up(x, hw):
    return F.interpolate(torch.from_numpy(np.ascontiguousarray(x, np.float32))[None, None], size=hw,
                         mode="bilinear", align_corners=False)[0, 0].numpy()


def errors(img, lab, inst, k, maps):
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(CRF_SITE)
    truth, valid = lab == k, lab != 255
    dist_out = distance_transform_edt(~truth)            # distance of background pixels to the object
    dist_in = distance_transform_edt(truth)              # distance of object pixels to the background
    ids = [i for i in np.unique(inst[truth]) if i not in (0, 255)]
    out = {}
    for name, cond, m, tau, crf in ERR:
        p = up(mm(maps[f"{cond}/{m}"]), truth.shape) >= tau
        if crf:
            p = apply_ovam_densecrf(img, p, CRF_SITE).astype(bool)
        p &= valid
        fp = p & ~truth; fn = ~p & truth & valid; tp = p & truth
        other = fp & (lab != 0)
        bg = fp & (lab == 0)
        r = {"tp": int(tp.sum()), "fp_other": int(other.sum()), "fp_bg_near": int((bg & (dist_out <= BAND)).sum()),
             "fp_bg_far": int((bg & (dist_out > BAND)).sum()), "fn_band": int((fn & (dist_in <= BAND)).sum()),
             "fn_interior": int((fn & (dist_in > BAND)).sum()),
             "fp_other_by_class": {int(c): int((other & (lab == c)).sum()) for c in np.unique(lab[other])},
             "instances": [(int((inst == i).sum()), float((p & (inst == i)).sum() / max((inst == i).sum(), 1)))
                           for i in ids]}
        out[name] = r
    return out


def work(job):
    pdir, out = Path(job[0]), Path(job[1])
    if out.exists():
        return
    torch.set_num_threads(1)
    st, iid, k = pdir.parts[-3], pdir.parts[-2], int(pdir.parts[-1])
    img = np.asarray(Image.open(VOC / f"JPEGImages/{iid}.jpg").convert("RGB"))
    lab = np.asarray(Image.open(VOC / f"SegmentationClass/{iid}.png"))
    inst = np.asarray(Image.open(VOC / f"SegmentationObject/{iid}.png"))
    truth, valid = lab == k, lab != 255
    meta = json.loads((pdir / "meta.json").read_text()); maps = np.load(pdir / "maps.npz")
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
    n_all = len(tok(meta["prompts"]["ref"], padding=False)["input_ids"])
    res = {}
    y64, keep64 = gt_at(truth, valid, 64)
    for cond in CONDS:
        tag = cond.replace("/", "_")
        atlas = np.load(pdir / f"atlas_{tag}.npz")
        mods, auc, rp, vn, kn, tk = ca_atlas(atlas, n_all, truth, valid)
        res.update({f"{tag}/ca_auc": auc, f"{tag}/ca_rprec": rp, f"{tag}/ca_vnorm": vn, f"{tag}/ca_knorm": kn,
                    f"{tag}/tok_auc": tk})
        sa_mods = sorted(k[3:] for k in atlas.files if k.startswith("rl/"))
        res[f"{tag}/rl"] = np.stack([atlas[f"rl/{m}"].astype(np.float32) for m in sa_mods])   # [16,31,8,3,2]
        qk = np.load(pdir / f"qk_{tag}.npz")
        curves, heads2, c16p = diffusion(qk, maps[f"{cond}/A3"], y64, keep64)
        for s in curves:
            res[f"{tag}/diff{s}"] = curves[s]
        for s in heads2:
            res[f"{tag}/heads2_{s}"] = heads2[s]
        res[f"{tag}/c16path"] = c16p
    # mask areas per side for the R/L chance level (same construction as the capture's rl_masks)
    t = torch.as_tensor(truth & valid, dtype=torch.float32)[None, None]
    t64 = F.interpolate(t, size=(64, 64), mode="area"); base = (t64 >= .5).float()
    st3 = torch.cat([t64, 1 - F.max_pool2d(1 - base, 5, 1, 2), F.max_pool2d(base, 5, 1, 2)], 1)
    res["area"] = np.array([[float((F.interpolate(st3, size=(s, s), mode="area") if s < 64 else st3)[0, j].mean())
                             for j in range(3)] for s in (8, 16, 32, 64)], np.float32)
    err = errors(img, lab, inst, k, maps)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **res, mods=np.array(mods), sa_mods=np.array(sa_mods), err=json.dumps(err),
             area_full=float((truth & valid).sum() / valid.sum()), k=k)
    os.replace(tmp, out)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("capture", type=Path); ap_.add_argument("out", type=Path)
    ap_.add_argument("--set", default="D"); ap_.add_argument("--workers", type=int, default=12)
    ap_.add_argument("--limit", type=int)
    a = ap_.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    jobs = []
    for d in sorted((a.capture / a.set).glob("*/*/done.json")):
        p = d.parent; o = a.out / a.set / f"{p.parts[-2]}__{p.parts[-1]}.npz"; o.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((str(p), str(o)))
    jobs = jobs[: a.limit] if a.limit else jobs
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(work, jobs), 1):
            if n % 10 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
