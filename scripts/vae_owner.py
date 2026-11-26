#!/usr/bin/env python3
"""Pilot (pre-registered in runs/07_vae_owner/PREREG.md): does the SD VAE decoder carry sub-cell boundary ownership
beyond its own spatial footprint and RGB? 30 screening images (rich capture), BRACE frozen, ground truth only measures.

python scripts/vae_owner.py select                     CPU  -> runs/07_vae_owner/ids.json
python scripts/vae_owner.py run [--limit k] [--tag t]  GPU  regenerate z, gate 0, perturbation decodes -> CACHE/vae_owner*/
python scripts/vae_owner.py score [--tag t]            CPU  kernel, footprint, gates A-C, report
"""
from __future__ import annotations

import argparse, json, sys, time, types, zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402
import sa_boundary_diag as sbd  # noqa: E402

sbd.NBOOT = 2000
RUN = ROOT / "runs/07_vae_owner"
DUMP = CACHE / "ana_dump"
N_PER = 15; EPS = 0.05; NDRAW = 8; NSETS = 2; RING = 4; BAND = 16; MINCELLS = 4
LAT = 16; LAT_OFF = ((0, 0), (8, 0), (0, 8), (8, 8)); KDRAW = 2; W0, W1 = -64, 72          # kernel window
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}


def crc(s):
    return zlib.crc32(s.encode())


def work(tag):
    return CACHE / ("vae_owner" + (f"_{tag}" if tag else ""))


# ------------------------------------------------------------------ selection (CPU)
def select(a):
    from ideas_grouped import screen_ids
    pilot = json.loads((ROOT / "runs/04_regen_pilot/pilot_ids.json").read_text())
    excl = {r["id"] for v in pilot.values() for r in v}; out = {}
    for ds in ("voc", "coco"):
        rows = []
        for i in sorted(screen_ids(ds), key=crc):
            if i in excl:
                continue
            z = np.load(DUMP / ds / f"{i}.npz", allow_pickle=True); area = float(z["truth"].mean())
            if 0.01 <= area <= 0.90:
                rows.append({"id": i, "cls": str(z["cls"]), "area": area})
        pick, used = [], set()
        for r in rows:
            if r["cls"] not in used and len(pick) < N_PER:
                pick.append(r); used.add(r["cls"])
        for r in rows:
            if len(pick) < N_PER and r not in pick:
                pick.append(r)
        out[ds] = pick
    RUN.mkdir(parents=True, exist_ok=True); (RUN / "ids.json").write_text(json.dumps(out, indent=1))
    print({k: [(r["id"], r["cls"]) for r in v] for k, v in out.items()})


# ------------------------------------------------------------------ label-free geometry (shared)
def geometry(z):
    """BRACE mask M (frozen), band, matched cell sets F/B. Label-free."""
    import ana_ops_common as oc
    from scipy.ndimage import distance_transform_edt, binary_dilation
    oc.crf_init()
    img = z["image"]; v = st.mm(z["v3"]); M = oc.crf(img, st.up(v, (512, 512)) >= 0.55)
    din, dout = distance_transform_edt(M), distance_transform_edt(~M)
    band = (M & (din <= BAND)) | (~M & (dout <= BAND)); sdist = np.where(M, -din, dout)
    bcell = sbd.pool(band.astype(float), 64) > 0
    ring = binary_dilation(bcell, np.ones((2 * RING + 1, 2 * RING + 1), bool)) & ~bcell
    # amendment 1: match F and B ring by ring (equal counts at each Chebyshev distance 1..RING from the band cells)
    from scipy.ndimage import distance_transform_cdt
    cheb = distance_transform_cdt(~bcell, metric="chessboard")
    rng = np.random.default_rng(crc(str(z["prompt"])) if "prompt" in z.files else 0); Fl, Bl = [], []
    for d in range(1, RING + 1):
        f, b = np.argwhere((v >= 0.9) & ring & (cheb == d)), np.argwhere((v <= 0.1) & ring & (cheb == d)); m = min(len(f), len(b))
        Fl.append(f[np.sort(rng.choice(len(f), m, replace=False))]); Bl.append(b[np.sort(rng.choice(len(b), m, replace=False))])
    Fc, Bc = np.concatenate(Fl), np.concatenate(Bl); n = len(Fc)
    dist_c = distance_transform_edt(~bcell)
    return dict(M=M, band=band, sdist=sdist, F=Fc, B=Bc, n=n,
                dF=float(dist_c[tuple(Fc.T)].mean()) if n else np.nan, dB=float(dist_c[tuple(Bc.T)].mean()) if n else np.nan)


# ------------------------------------------------------------------ GPU run
def run(a):
    import torch
    from scripts import sg27_runtime
    import idea1_knockout as k1
    from diffusers.models.attention_processor import AttnProcessor
    ids = json.loads((RUN / "ids.json").read_text())
    runtime = sg27_runtime.make_runtime(k1.MODEL); pipe = runtime["pipe"]; vae = pipe.vae.to(torch.float32)
    sf = vae.config.scaling_factor; dev = pipe.device
    W = work(a.tag); W.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def dec(Z):
        out = []
        for i in range(0, len(Z), 4):
            out.append(((vae.decode(Z[i:i + 4].float() / sf).sample.clamp(-1, 1) + 1) / 2))
        return torch.cat(out)                                                   # [n,3,512,512] in [0,1]

    todo = [(ds, r["id"]) for ds in ("voc", "coco") for r in ids[ds]][: a.limit or None]
    for ds, iid in todo:
        o = W / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); z = np.load(DUMP / ds / f"{iid}.npz", allow_pickle=True)
        row = next(r for r in cc.gen_rows(ds) if r["image_id"] == iid)
        saved = dict(pipe.unet.attn_processors); pipe.unet.set_attn_processor(AttnProcessor())
        try:
            pe, ne, noise = k1.embeds(pipe, row)
            zb = k1.denoise(pipe, types.SimpleNamespace(ko_rows=()), k1.new_sched(pipe, int(row["num_inferece_steps"])), noise, 0, ne, pe)
        finally:
            pipe.unet.set_attn_processor(saved)
        zb = zb.float(); x0 = dec(zb)
        rec = np.asarray((x0[0].permute(1, 2, 0).cpu().numpy() * 255).round(), np.uint8)
        g0 = float(np.abs(rec.astype(int) - z["image"].astype(int)).mean())
        G = geometry(z)
        res = dict(gate0=g0, n=G["n"], dF=G["dF"], dB=G["dB"], F=G["F"], B=G["B"], ds=ds, cls=str(z["cls"]))
        rbar = float(zb[0].norm(dim=0).mean())
        gen = torch.Generator(device="cpu").manual_seed(crc(iid))

        def pert(cells, k):
            V = torch.zeros((k, 4, 64, 64))
            if len(cells):
                d = torch.randn((k, 4, len(cells)), generator=gen); d = d / d.norm(dim=1, keepdim=True) * EPS * rbar
                V[:, :, cells[:, 0], cells[:, 1]] = d
            return V.to(dev)

        if G["n"] >= MINCELLS and g0 <= 1.0:
            for nm, cells in (("F", G["F"]), ("B", G["B"])):
                for s in range(NSETS):
                    dD = dec(zb + pert(cells, NDRAW)) - x0
                    res[f"e{nm}{s}"] = (dD ** 2).sum(1).mean(0).cpu().numpy().astype(np.float32)
        # kernel windows (isolated lattice cells), sum and count in [W0, W1) around each cell's top-left
        ks = np.zeros((W1 - W0, W1 - W0)); kn = np.zeros_like(ks)
        if g0 <= 1.0:
            for oy, ox in LAT_OFF:
                cells = np.array([(oy + LAT * i, ox + LAT * j) for i in range(64 // LAT) for j in range(64 // LAT)])
                E = (dec(zb + pert(cells, KDRAW)) - x0).pow(2).sum(1).cpu().numpy()            # [KDRAW,512,512]
                P = np.pad(E, ((0, 0), (-W0, W1), (-W0, W1))); V = np.pad(np.ones((512, 512)), ((-W0, W1), (-W0, W1)))
                for cy, cx in cells:
                    y, x = 8 * cy, 8 * cx
                    ks += P[:, y:y + W1 - W0, x:x + W1 - W0].sum(0); kn += KDRAW * V[y:y + W1 - W0, x:x + W1 - W0]
        res.update(ksum=ks, kcnt=kn, sec=time.perf_counter() - t0)
        np.savez(o, **res)
        print(f"[vae:{ds}] {iid} {res['cls']} gate0 {g0:.3f} cells {G['n']} (dist F {G['dF']:.1f}, B {G['dB']:.1f}) "
              f"{res['sec']:.0f}s", flush=True)
        if g0 > 1.0:
            sys.exit(f"GATE 0 FAILED on {iid}: {g0:.3f}")


# ------------------------------------------------------------------ CPU scoring
def footprint(cells, K):
    E = np.zeros((512 - W0 + W1, 512 - W0 + W1))
    for cy, cx in cells:
        y, x = 8 * cy, 8 * cx; E[y:y + W1 - W0, x:x + W1 - W0] += K
    return E[-W0:-W0 + 512, -W0:-W0 + 512]


def score(a):
    from scipy.ndimage import gaussian_filter, sobel
    ids = json.loads((RUN / "ids.json").read_text()); W = work(a.tag)
    todo = [(ds, r["id"]) for ds in ("voc", "coco") for r in ids[ds] if (W / f"{r['id']}.npz").exists()]
    R = {iid: dict(np.load(W / f"{iid}.npz", allow_pickle=True)) for _, iid in todo}
    K = sum(r["ksum"] for r in R.values()) / np.maximum(sum(r["kcnt"] for r in R.values()), 1)
    yy, xx = np.mgrid[W0:W1, W0:W1] - 3.5; rad = np.hypot(yy, xx)
    prof = [float(K[(rad >= r0) & (rad < r0 + 4)].mean()) for r0 in range(0, 64, 4)]
    per = []; Xb, Xt, Yp, Gp = [], [], [], []; cells_rows = []
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    for ds, iid in todo:
        r = R[iid]
        if "eF0" not in r:
            per.append(dict(id=iid, ds=ds, included=False, n=int(r["n"]))); continue
        z = np.load(DUMP / ds / f"{iid}.npz", allow_pickle=True); truth = z["truth"].astype(bool); img = z["image"]
        G = geometry(z); assert G["n"] == int(r["n"])
        eF = (r["eF0"] + r["eF1"]) / 2; eB = (r["eB0"] + r["eB1"]) / 2; o = eF / (eF + eB + 1e-12)
        o_a = r["eF0"] / (r["eF0"] + r["eB0"] + 1e-12); o_b = r["eF1"] / (r["eF1"] + r["eB1"] + 1e-12)
        fF, fB = footprint(G["F"], K), footprint(G["B"], K); ofp = fF / (fF + fB + 1e-12)
        bd = G["band"]; y = truth[bd]; M = G["M"]
        from scipy.stats import spearmanr
        rep = float(spearmanr(o_a[bd], o_b[bd])[0])
        A = sbd.auc(y, o[bd]); Afp = sbd.auc(y, ofp[bd])
        iou = lambda m: float((m & truth).sum() / max((m | truth).sum(), 1))
        Mc = np.where(bd, o > 0.5, M); Mf = np.where(bd, ofp > 0.5, M)
        L = sbd.lab(img); Ls = gaussian_filter(L, (1, 1, 0))
        Gm = np.sqrt(sum(sobel(Ls[..., c], 0) ** 2 + sobel(Ls[..., c], 1) ** 2 for c in range(3)))
        cellmask = lambda C: np.kron(np.isin(np.arange(4096), C[:, 0] * 64 + C[:, 1]).reshape(64, 64), np.ones((8, 8))).astype(bool)
        mF, mB = L[cellmask(G["F"])].mean(0), L[cellmask(G["B"])].mean(0)
        dFc, dBc = np.linalg.norm(L - mF, axis=-1), np.linalg.norm(L - mB, axis=-1)
        rng = np.random.default_rng(crc(iid)); ix = np.flatnonzero(bd.ravel())
        ix = rng.choice(ix, min(4000, len(ix)), replace=False)
        base = np.c_[dFc.ravel()[ix], dBc.ravel()[ix], Gm.ravel()[ix], G["sdist"].ravel()[ix], lg(ofp.ravel()[ix])]
        Xb.append(base); Xt.append(np.c_[base, lg(o.ravel()[ix])]); Yp.append(truth.ravel()[ix]); Gp.append(np.full(len(ix), iid))
        # within-cell (cells crossed by the GT boundary, centre in band)
        for cy in range(64):
            for cx in range(64):
                sl = (slice(8 * cy, 8 * cy + 8), slice(8 * cx, 8 * cx + 8)); t = truth[sl]
                if t.sum() >= 8 and (~t).sum() >= 8 and bd[8 * cy + 4, 8 * cx + 4]:
                    wrong = int((M[sl] != t).sum())
                    cells_rows.append(dict(ds=ds, wrong=wrong, o=sbd.auc(t.ravel(), o[sl].ravel()),
                                           fp=sbd.auc(t.ravel(), ofp[sl].ravel()),
                                           col=sbd.auc(t.ravel(), (dBc - dFc)[sl].ravel())))
        per.append(dict(id=iid, ds=ds, included=True, n=int(r["n"]), dF=float(r["dF"]), dB=float(r["dB"]), gate0=float(r["gate0"]),
                        rep=rep, auc_o=A, auc_fp=Afp, dA=A - Afp, iou_M=iou(M), iou_o=iou(Mc), iou_fp=iou(Mf)))
    inc = [p for p in per if p["included"]]; nI = len(inc)
    dA = sbd.mean_ci([p["dA"] for p in inc])
    passA = bool(dA[0] >= 0.02 and dA[1] > 0)
    B = sbd.compare(np.concatenate(Xb), np.concatenate(Xt), np.concatenate(Yp), np.concatenate(Gp))
    passB = bool(B["delta"] >= 0.05 and B["lo"] > 0)
    g = np.array([p["iou_o"] - p["iou_M"] for p in inc]); gf = np.array([p["iou_fp"] - p["iou_M"] for p in inc])
    need = int(np.ceil(2 * nI / 3)); wins = int((g > 0).sum())
    passC = bool(wins >= need and g.mean() > 0 and g.mean() > gf.mean())
    verdict = "PASS (all three gates)" if passA and passB and passC else "CLOSE"
    out = dict(verdict=verdict, passA=passA, passB=passB, passC=passC, dA=dA, B=B, C=dict(wins=wins, need=need, n=nI,
               mean_gain=float(g.mean()), mean_gain_fp=float(gf.mean())), kernel_profile=prof, per_image=per)
    f4 = lambda x: f"{x:+.4f}"
    Lr = ["# VAE-decoder ownership pilot (pre-registered in `PREREG.md`)\n",
          f"{len(todo)} images scored, {nI} included (≥ {MINCELLS} matched cells per set). ε {EPS}, {NDRAW * NSETS} draws per set. "
          f"Gate 0 (decode reproduces the stored image): max mean |Δ| {max(p.get('gate0', 0) for p in inc):.3f} grey levels.\n",
          f"## Verdict: **{verdict}**\n",
          "| gate | statistic | bar | result |", "|---|---|---|---|",
          f"| A. beyond footprint | mean per-image AUROC(o) − AUROC(o_fp), band pixels | ≥ +0.02, CI > 0 | {f4(dA[0])} [{f4(dA[1])}, {f4(dA[2])}] → {'pass' if passA else 'fail'} |",
          f"| B. beyond RGB + footprint | OOF ΔAUROC ({B['rgb']:.3f} → {B['rgb_sa']:.3f}) | ≥ +0.05, CI > 0 | {f4(B['delta'])} [{f4(B['lo'])}, {f4(B['hi'])}] → {'pass' if passB else 'fail'} |",
          f"| C. label-free correction (o > 0.5 in band) | wins vs R2; mean IoU gain; vs footprint rule | ≥ {need}/{nI}; > 0; > fp | {wins}/{nI}; {f4(g.mean())}; fp {f4(gf.mean())} → {'pass' if passC else 'fail'} |",
          "\n## Per set\n", "| set | images | AUROC o | AUROC o_fp | IoU R2 | IoU o-rule | IoU fp-rule | repeatability ρ |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in ("voc", "coco"):
        P = [p for p in inc if p["ds"] == ds]
        if P:
            m = lambda k: float(np.nanmean([p[k] for p in P]))
            Lr.append(f"| {NAME[ds]} | {len(P)} | {m('auc_o'):.3f} | {m('auc_fp'):.3f} | {m('iou_M'):.4f} | {m('iou_o'):.4f} | {m('iou_fp'):.4f} | {m('rep'):.3f} |")
    Lr += ["\n## Within-cell discrimination (cells crossed by the ground-truth boundary, centre in band; reported, not gating)\n",
           "| subset | cells | AUROC o | AUROC o_fp | AUROC colour |", "|---|---:|---:|---:|---:|"]
    for nm, sel in (("all", lambda c: True), ("BRACE wrong on ≥ 8 px", lambda c: c["wrong"] >= 8)):
        C = [c for c in cells_rows if sel(c)]
        if C:
            Lr.append(f"| {nm} | {len(C)} | {np.nanmean([c['o'] for c in C]):.3f} | {np.nanmean([c['fp'] for c in C]):.3f} | {np.nanmean([c['col'] for c in C]):.3f} |")
    Lr += ["\n## Matched cell sets and kernel\n",
           f"- Mean matched cells per set: {np.mean([p['n'] for p in inc]):.1f}; mean distance to band (cells): F {np.nanmean([p['dF'] for p in inc]):.2f}, B {np.nanmean([p['dB'] for p in inc]):.2f}.",
           f"- Excluded images (< {MINCELLS} cells): {', '.join(p['id'] for p in per if not p['included']) or 'none'}.",
           "- Kernel radial profile (mean ΔD², 4-px rings from the cell centre): " + ", ".join(f"{x:.2e}" for x in prof[:10]) + "."]
    (RUN / (f"result_{a.tag}.md" if a.tag else "result.md")).write_text("\n".join(Lr) + "\n")
    (RUN / (f"result_{a.tag}.json" if a.tag else "result.json")).write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(Lr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    for n in ("run", "score"):
        p = sub.add_parser(n); p.add_argument("--limit", type=int); p.add_argument("--tag", default="")
    a = ap.parse_args(); {"select": select, "run": run, "score": score}[a.cmd](a)
