#!/usr/bin/env python3
"""Standard multi-class protocol: D development + frozen val scoring (2026-11-09).

Reads sg_voc21.py `capture --save-full` outputs and scores label-free, parameter-free ways of combining SCLIP's
full-size probabilities p with the SD C16 propagation of their 64x64 pooled version:
  sclip     p                                           (= SCLIP standalone)
  pool      bilinear(area-pool64(p))                    (no SA: isolates the 64-grid resolution loss; control)
  c16       bilinear(C16(p64))
  resid     p + bilinear(C16(p64) - p64)                (SA correction added to the full-size detail)
  geo       sqrt(p * bilinear(C16(p64)))
then SCLIP's own thresholds (VOC21: area 0.1, prob 0.1; VOC20 argmax), each with and without multi-label DenseCRF
(R2 parameters). VOC21 = 21 classes incl. background; VOC20 = 20 classes, background pixels ignored.

dev  D_DIR : the D photos (VOC train). Writes voc21_dev_D.md and voc21_choice.json. Selection rule (registered in
             voc21_prereg.md): the candidate in {c16, resid, geo} x {"", +crf} with the highest D VOC21 mIoU.
val  VAL_DIR: the 1449 VOC2012 val photos. Primary = chosen − (sclip+crf) VOC21, photo-bootstrap 95% CI > 0.
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
torch.set_num_threads(1)
import sg_voc21 as v  # noqa: E402

AN = ROOT / "outputs/segmentation/sg_real_atlas/analysis"
VAR = ("sclip", "pool", "c16", "resid", "geo")
CANDIDATES = [f"{n}{t}" for n in ("c16", "resid", "geo") for t in ("", "+crf")]


def one(f):
    z = np.load(f); iid = f.stem
    img = np.asarray(Image.open(v.st.VOC / f"JPEGImages/{iid}.jpg").convert("RGB"))
    lab = np.asarray(Image.open(v.st.VOC / f"SegmentationClass/{iid}.png")).astype(np.int64); H, W = lab.shape
    up = lambda x: F.interpolate(torch.from_numpy(x.astype(np.float32))[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    out = {"t": float(z["t_total"])}
    for pro, sl, full in (("21", slice(0, 21), "p21_full"), ("20", slice(21, 41), "p20_full")):
        p = torch.from_numpy(z[full].astype(np.float32)); P64 = z["p64"][sl]; Q64 = z["prop64"][sl]
        maps = {"sclip": p, "pool": up(P64), "c16": up(Q64), "resid": (p + up(Q64 - P64)).clamp_min(0),
                "geo": (p * up(Q64).clamp_min(0)).sqrt()}
        for n, m in maps.items():
            if pro == "21":
                pr, mt = v.thresholds(m.clone(), 0.1, 0.1)
            else:
                pr, mt = m.argmax(0), m
            pr = pr.numpy().astype(np.int64); cr = v.crf(img, mt).astype(np.int64)
            for tag, x in (("", pr), ("+crf", cr)):
                if pro == "20":
                    ok = (lab != 255) & (lab != 0); g, q = lab[ok], x[ok] + 1
                else:
                    ok = lab != 255; g, q = lab[ok], x[ok]
                out[f"{n}{tag}|{pro}"] = np.bincount(g * 21 + q, minlength=441).reshape(21, 21)
    return out


def ious(C, pro):
    iou = np.diag(C) / np.maximum(C.sum(0) + C.sum(1) - np.diag(C), 1)
    return iou[1:] if pro == "20" else iou


def miou(C, pro):
    return ious(C, pro).mean() * 100


def evaluate(d, ids=None):
    files = sorted(d.glob("*.npz")) if ids is None else [d / f"{i}.npz" for i in ids]
    miss = [f for f in files if not f.exists()]
    if miss:
        raise SystemExit(f"{len(miss)} photos missing, e.g. {miss[0]}")
    with Pool(16) as pool:
        R = pool.map(one, files, chunksize=4)
    S = {k: np.stack([r[k] for r in R]) for k in R[0] if k != "t"}
    return S, np.array([r["t"] for r in R])


def tables(S, t, title, chosen=None):
    rng = np.random.default_rng(0); n = len(next(iter(S.values())))
    B = [rng.integers(0, n, n) for _ in range(1000)]
    ci = lambda a, b, pro: np.quantile([miou(S[f"{a}|{pro}"][x].sum(0), pro) - miou(S[f"{b}|{pro}"][x].sum(0), pro) for x in B], [.025, .975])
    d = lambda a, b, pro: miou(S[f"{a}|{pro}"].sum(0), pro) - miou(S[f"{b}|{pro}"].sum(0), pro)
    L = [title, "| variant | VOC21 | VOC20 | VOC21 +CRF | VOC20 +CRF |", "|---|---:|---:|---:|---:|"]
    for nm in VAR:
        L.append(f"| {nm} | {miou(S[nm + '|21'].sum(0), '21'):.2f} | {miou(S[nm + '|20'].sum(0), '20'):.2f} | "
                 f"{miou(S[nm + '+crf|21'].sum(0), '21'):.2f} | {miou(S[nm + '+crf|20'].sum(0), '20'):.2f} |")
    if chosen:
        lo, hi = ci(chosen, "sclip+crf", "21")
        L.append(f"\n**Primary (registered):** {chosen} − sclip+crf VOC21 = {d(chosen, 'sclip+crf', '21'):+.2f} "
                 f"(photo-bootstrap 95% CI {lo:+.2f}…{hi:+.2f}) → **{'PASS' if lo > 0 else 'FAIL'}**")
    L.append("\nΔ vs SCLIP + CRF and vs SCLIP standalone (photo-bootstrap 95% CI)" + (" — secondary" if chosen else "") + ":\n")
    L.append("| variant | Δ VOC21 vs sclip+crf | CI | Δ VOC21 vs sclip | CI | Δ VOC20 vs sclip | CI |\n|---|---:|---|---:|---|---:|---|")
    for k in [f"{nm}{tg}" for nm in VAR[1:] for tg in ("", "+crf")]:
        a, b, c = ci(k, "sclip+crf", "21"), ci(k, "sclip", "21"), ci(k, "sclip", "20")
        L.append(f"| {k} | {d(k, 'sclip+crf', '21'):+.2f} | {a[0]:+.2f}…{a[1]:+.2f} | {d(k, 'sclip', '21'):+.2f} | "
                 f"{b[0]:+.2f}…{b[1]:+.2f} | {d(k, 'sclip', '20'):+.2f} | {c[0]:+.2f}…{c[1]:+.2f} |")
    L.append("\nVOC21 error types (dataset level):\n")
    for k in ["sclip", "sclip+crf", "pool", "c16", "resid"] + ([chosen] if chosen and chosen not in ("c16", "resid") else []):
        C = S[k + "|21"].sum(0)
        L.append(f"- {k}: background→object {C[0, 1:].sum() / C[0].sum() * 100:.1f}% of bg pixels; object→background "
                 f"{C[1:, 0].sum() / C[1:].sum() * 100:.1f}% of object pixels; wrong object "
                 f"{(C[1:, 1:].sum() - np.trace(C[1:, 1:])) / C[1:].sum() * 100:.1f}%")
    show = ["sclip", "sclip+crf"] + ([chosen] if chosen else ["c16", "resid"])
    L.append("\nPer-class VOC21 IoU:\n\n| class | " + " | ".join(show) + " |\n|---|" + "---:|" * len(show))
    I = {k: ious(S[k + "|21"].sum(0), "21") * 100 for k in show}
    for c in range(21):
        L.append(f"| {v.NAMES21[c]} | " + " | ".join(f"{I[k][c]:.1f}" for k in show) + " |")
    L.append(f"\nRuntime: {t.mean():.1f} s/photo mean (median {np.median(t):.1f}): SCLIP + one DDIM inversion + one SD run + "
             "propagation + all CRFs, on one GPU node.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    mode, d = sys.argv[1], Path(sys.argv[2])
    if mode == "dev":
        S, t = evaluate(d)
        chosen = max(CANDIDATES, key=lambda k: miou(S[k + "|21"].sum(0), "21"))
        (AN / "voc21_choice.json").write_text(json.dumps({"chosen": chosen, "rule": "max D VOC21 mIoU over " + ", ".join(CANDIDATES),
                                                          "D_VOC21": {k: miou(S[k + "|21"].sum(0), "21") for k in CANDIDATES}}, indent=1))
        txt = tables(S, t, f"# D development, standard multi-class protocol ({len(t)} D photos, VOC train; val untouched)\n") + \
            f"\n**Chosen for val by the registered rule:** `{chosen}`\n"
        (AN / "voc21_dev_D.md").write_text(txt); print(txt)
    else:
        chosen = json.loads((AN / "voc21_choice.json").read_text())["chosen"]
        S, t = evaluate(d, v.ids_of("val"))
        txt = tables(S, t, f"# VOC2012 val ({len(t)} photos), standard protocol: SCLIP ViT-B/16 + SD1.5 self-attention\n\n"
                     "Pre-registered in `voc21_prereg.md`; the variant was chosen on D by the registered rule "
                     f"(`voc21_choice.json`: **{chosen}**) before val was scored. Nothing tuned on val. No class-presence "
                     "information used. Dataset-level confusion (mmseg IoUMetric style). Our SCLIP port reproduces the repo's "
                     "B16 numbers (59.63 / 81.54).\n", chosen)
        (AN / "voc21_val.md").write_text(txt); print(txt)
