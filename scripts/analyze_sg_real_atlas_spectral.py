#!/usr/bin/env python3
"""SG-real-atlas analysis 3b (Section 137): spectral structure of the SA graphs vs the real mask.

Pairs whose image stored all-module Q/K (the first 50 D images), renoise/n0, calls 10..20. For every SA module
at sides 16 / 32 / 64, the symmetrised graph W = (A + A^T)/2 of each head and of the head mean (A row-stochastic
softmax(q k^T / sqrt d)), normalised Laplacian L = I - D^-1/2 W D^-1/2 and its lowest eigenvectors u_0..u_32.
  E_k     share of the mask's energy (f = D^1/2 m, u_0 component removed) in u_1..u_k, k = 1,2,4,8,16,32;
          chance = the same mask rolled by half the grid; also for the A3 seed.
  eig     AUROC of each generalised eigenvector D^-1/2 u_j (j = 1..16, sign-free) against the mask; the
          label-free pick = the eigenvector most correlated with the seed (sign from that correlation).
  lowpass AUROC of the seed projected onto u_0..u_k (graph low-pass filter, label-free), k = 4,8,16,32,64.
Sides 16 and 32: head mean and the 8 heads at calls 10, 15, 20; side 64: head mean at calls 10, 15, 20.
Usage: analyze_sg_real_atlas_spectral.py CAPTURE_DIR OUT_DIR [--workers 12]
"""
import argparse, json, math, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_sg_real_atlas_alignment import SIDE, VOC, auroc, gt_at, side_of  # noqa: E402

MANIFEST = ROOT / "outputs/segmentation/sg_real_atlas/manifest.json"
KS = (1, 2, 4, 8, 16, 32)
LP = (4, 8, 16, 32, 64)
CALLS = (0, 5, 10)                       # calls 10, 15, 20 within the stored window
NEIG = 65


def spectrum(A):
    W = (A + A.T) / 2; d = W.sum(1); dm = 1 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(len(W)) - dm[:, None] * W * dm[None]
    w, U = eigh(L, subset_by_index=[0, min(NEIG, len(W)) - 1], driver="evr")
    return w, U, d


def energy(U, d, m):
    f = np.sqrt(d) * m; u0 = U[:, 0]; f = f - (u0 @ f) * u0; tot = max(f @ f, 1e-12)
    c = (U[:, 1:max(KS) + 1].T @ f) ** 2
    return np.array([c[:k].sum() / tot for k in KS], np.float32)


def analyse(A, y, keep, seed, side):
    w, U, d = spectrum(A)
    m = y.astype(np.float64)
    roll = np.roll(m.reshape(side, side), (side // 2, side // 2), (0, 1)).ravel()
    out = {"E": energy(U, d, m), "E_roll": energy(U, d, roll), "E_seed": energy(U, d, seed), "lam": w[:17]}
    V = U[:, 1:17] / np.sqrt(np.maximum(d, 1e-12))[:, None]          # generalised eigenvectors
    au = auroc(V.T[:, keep], y[keep]); out["eig_auc"] = np.maximum(au, 1 - au)
    cor = np.array([np.corrcoef(V[:, j], seed)[0, 1] for j in range(V.shape[1])])
    j = int(np.nanargmax(np.abs(cor))); v = V[:, j] * np.sign(cor[j])
    out["eig_pick_auc"] = float(auroc(v[keep][None], y[keep])[0]); out["eig_pick_j"] = j + 1
    f = np.sqrt(d) * seed; lp = []
    for k in LP:
        Uk = U[:, :min(k, U.shape[1])]; g = Uk @ (Uk.T @ f) / np.sqrt(np.maximum(d, 1e-12))
        lp.append(float(auroc(g[keep][None], y[keep])[0]))
    out["lowpass_auc"] = np.array(lp, np.float32)
    out["seed_auc"] = float(auroc(seed[keep][None], y[keep])[0])
    return out


def work(job):
    pdir, outp = Path(job[0]), Path(job[1])
    if outp.exists():
        return
    torch.set_num_threads(1)
    iid, k = pdir.parts[-2], int(pdir.parts[-1])
    lab = np.asarray(Image.open(VOC / f"SegmentationClass/{iid}.png")); truth, valid = lab == k, lab != 255
    maps = np.load(pdir / "maps.npz"); seed64 = torch.from_numpy(maps["renoise/n0/A3"]).float()[None, None]
    atlas = np.load(pdir / "atlas_renoise_n0.npz")
    res = {}
    for key in sorted(x for x in atlas.files if x.startswith("qk/")):
        mod = key[3:]; s = side_of(mod)
        if s == 8:
            continue
        y, keep = gt_at(truth, valid, s)
        seed = (F.interpolate(seed64, size=(s, s), mode="area") if s < 64 else seed64)[0, 0].numpy().ravel().astype(np.float64)
        qk = atlas[key]                                                    # [11,2,8,N,d]
        for c in CALLS:
            q, kk = torch.from_numpy(qk[c, 0]).float(), torch.from_numpy(qk[c, 1]).float()
            A = torch.softmax(torch.bmm(q, kk.transpose(1, 2)) / math.sqrt(q.shape[-1]), -1).double().numpy()
            graphs = {"mean": A.mean(0)}
            if s < 64:
                graphs.update({f"h{h}": A[h] for h in range(8)})
            for g, Ag in graphs.items():
                for name, v in analyse(Ag, y, keep, seed, s).items():
                    res[f"{mod}|{c}|{g}|{name}"] = np.asarray(v)
    tmp = outp.with_name(outp.stem + ".tmp.npz")
    np.savez(tmp, **res, area=float((truth & valid).sum() / valid.sum()), k=k); os.replace(tmp, outp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path); ap.add_argument("out", type=Path); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    first = {e["image"] for e in json.loads(MANIFEST.read_text())["D"] if e.get("all_qk")}
    jobs = []
    for d in sorted((a.capture / "D").glob("*/*/done.json")):
        p = d.parent
        if p.parts[-2] in first:
            o = a.out / f"{p.parts[-2]}__{p.parts[-1]}.npz"; o.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((str(p), str(o)))
    jobs = jobs[: a.limit] if a.limit else jobs
    print(len(jobs), "pairs", flush=True)
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(work, jobs), 1):
            if n % 5 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
