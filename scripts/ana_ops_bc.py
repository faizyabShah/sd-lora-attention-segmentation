#!/usr/bin/env python3
"""Operator batches B and C (GPU capture + CPU CRF, exploratory) on the class-stratified 150/150 screening images.

Each image is regenerated once with the frozen live pipeline (A3 seed, 11 SA calls at 16/32/64); every operator
uses that capture, so all maps are paired with BRACE (V3) in the same capture. s = min-max A3 seed; A = head- and
call-mean 64x64 self-attention; W = (A + A^T) / 2.

B  6  personalised PageRank x = (1-a) A x + a s, a in {0.1, 0.2, 0.4}; PPR(0.2) as the seed of V3
   7  foreground / background competition: PPR of s vs PPR of a label-free background seed (image border x (1 - s),
      or 1 - s); random walker (Grady 2006) with fg markers s >= 0.9 and bg markers on the border with s <= 0.1
   8  graph cut: V3 as unary, kNN-16 sparsified W as pairwise, lambda in {0.5, 2} (scipy max-flow)
   9  low-pass graph filter: s projected on the top-k eigenvectors of D^-1/2 W D^-1/2, k in {8, 16, 32}; k=16 into V3
  10  attention sinks removed (columns with mass > 5x mean, rows renormalised) for one-step propagation; softmax
      temperature T in {0.5, 2} for one-step propagation and for V3 (q / T at every SA level)
  11  mean-field CRF with W as the pairwise kernel, unary V3, weight in {1, 3}
  12  attention rollout over the 11 calls: prod (0.5 A_i + 0.5 I) s
C 13  re-noising ensemble: the generated image's VAE latent noised to t in {200, 400, 600} (fixed noise), one
      conditional UNet call each, its SA at the three frozen modules -> V3 with the A3 seed; mean of the four V3 maps
      (live + three) and mean - 0.5 sd

python scripts/ana_ops_bc.py run --dataset voc|coco [--crf-workers 8]   -> CACHE/ops_bc/<ds>/<id>.npz
python scripts/ana_ops_bc.py report                                   -> runs/analysis_cluster/ops_bc.md
"""
from __future__ import annotations

import argparse, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402
import ana_ops_common as oc  # noqa: E402

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
WORK = CACHE / "ops_bc"; OUT = ROOT / "runs/analysis_cluster"
RENOISE_T = (200, 400, 600)
GROUPS = {
    "6 PageRank": ("ppr01", "ppr02", "ppr04", "ppr02_v3"),
    "7 fg/bg competition": ("comp_border", "comp_inv", "rw"),
    "8 graph cut": ("gc_l05", "gc_l2", "gc_l05+R2", "gc_l2+R2"),
    "9 low-pass": ("lp8", "lp16", "lp32", "lp16_v3"),
    "10 sinks / temperature": ("p1", "desink_p1", "temp05_p1", "temp2_p1", "temp05_v3", "temp2_v3"),
    "11 SA-kernel CRF": ("sacrf_w1", "sacrf_w3"),
    "12 rollout": ("rollout",),
    "13 re-noising ensemble": ("rn200_v3", "rn400_v3", "rn600_v3", "ens_mean", "ens_conf"),
}


def heads_attn(pair, T=1.0):
    q, k = pair; q, k = q.float(), k.float()
    return torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (math.sqrt(q.shape[-1]) * T), -1)   # [8, N, N]


def ppr(A, s, a, it=60):
    x = s.clone()
    for _ in range(it):
        x = (1 - a) * (A @ x) + a * s
    return x


def norm01(x):
    x = x - x.min(); return x / (x.max() + 1e-12)


def graph_cut(Wk, p, lam):
    """Binary min-cut: unary -log p / -log(1-p), pairwise lam * W (kNN, symmetric). Returns fg mask [N]."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_flow, breadth_first_order
    N = len(p); S, T = N, N + 1; sc = 1000.0
    p = np.clip(p, 0.01, 0.99)
    Wc = Wk.tocoo(); r, c, w = Wc.row, Wc.col, Wc.data * lam
    rows = np.r_[r, np.full(N, S), np.arange(N)]; cols = np.r_[c, np.arange(N), np.full(N, T)]
    cap = np.r_[w, -np.log(1 - p), -np.log(p)]
    G = coo_matrix((np.maximum(np.round(cap * sc), 0).astype(np.int32), (rows, cols)), shape=(N + 2, N + 2)).tocsr()
    G.sum_duplicates(); res = maximum_flow(G, S, T)
    R = G - res.flow; R.data = np.where(R.data > 0, 1, 0); R.eliminate_zeros()
    reach = breadth_first_order(R, S, directed=True, return_predecessors=False)
    m = np.zeros(N + 2, bool); m[reach] = True
    return m[:N]


class QKCap:
    """AttnProcessor arithmetic that also stores the per-head (q, k) of one self-attention module."""

    def __init__(self, store, side):
        self.store, self.side = store, side

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        from diffusers.models.attention_processor import AttnProcessor
        q = attn.head_to_batch_dim(attn.to_q(hidden_states)); k = attn.head_to_batch_dim(attn.to_k(hidden_states))
        self.store[self.side] = (q[-attn.heads:].detach(), k[-attn.heads:].detach())
        return AttnProcessor()(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


@torch.no_grad()
def renoise_sa(pipe, modules, img, t):
    """SA (q, k) at the three frozen modules from one conditional UNet call on the noised VAE latent of img."""
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float().to(pipe.device) / 127.5 - 1
    z0 = pipe.vae.encode(x.to(pipe.vae.dtype)).latent_dist.mean * pipe.vae.config.scaling_factor
    g = torch.Generator(device="cpu").manual_seed(0); eps = torch.randn(z0.shape, generator=g).to(z0.device, z0.dtype)
    ac = pipe.scheduler.alphas_cumprod.to(z0.device)[t]; zt = ac.sqrt() * z0 + (1 - ac).sqrt() * eps
    pe, _ = pipe.encode_prompt(prompt=PROMPT[0], device=pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=False)
    saved = dict(pipe.unet.attn_processors); new = {}; store = {}
    for name, p in saved.items():
        orig = getattr(p, "original", p); mod = name.removesuffix(".processor")
        side = next((s for s, m in modules.items() if m == mod), None)
        new[name] = QKCap(store, side) if side else orig
    pipe.unet.set_attn_processor(new)
    try:
        pipe.unet(zt, torch.tensor(t, device=z0.device), encoder_hidden_states=pe, return_dict=False)
    finally:
        pipe.unet.set_attn_processor(saved)
    return {s: [store[s]] * 11 for s in (16, 32, 64)}


PROMPT = [""]


def maps_for(sa, seed64, img, pipe, modules):
    from src.segmentation import sg_combined_v2
    V3 = lambda sa_, sd: norm01(sg_combined_v2.aggregate(sa_, sd.reshape(64, 64))[0]["V3"].float().reshape(-1))
    dev = seed64.device; s = norm01(seed64.float().reshape(-1)); out = {}
    calls = [heads_attn(p).mean(0) for p in sa[64]]
    A = torch.stack(calls).mean(0); W = (A + A.T) / 2
    out["brace"] = V3(sa, seed64.float()); out["p1"] = norm01(A @ s)
    # 6 PageRank
    for a, nm in ((0.1, "ppr01"), (0.2, "ppr02"), (0.4, "ppr04")):
        out[nm] = norm01(ppr(A, s, a))
    out["ppr02_v3"] = V3(sa, out["ppr02"])
    # 7 competition
    yy, xx = torch.meshgrid(torch.arange(64, device=dev), torch.arange(64, device=dev), indexing="ij")
    border = ((yy < 4) | (yy > 59) | (xx < 4) | (xx > 59)).reshape(-1).float()
    fg = ppr(A, s / s.sum(), 0.2)
    for nm, b in (("comp_border", border * (1 - s)), ("comp_inv", 1 - s)):
        bg = ppr(A, b / (b.sum() + 1e-12), 0.2); out[nm] = fg / (fg + bg + 1e-12)
    fgm = s >= 0.9
    if fgm.sum() < 5:
        fgm = s >= torch.quantile(s, 0.99)
    bgm = (border > 0) & (s <= 0.1)
    if bgm.sum() < 5:
        bgm = (border > 0) & (s <= torch.quantile(s[border > 0], 0.5))
    bgm &= ~fgm; mk = fgm | bgm; u = ~mk
    Lp = torch.diag(W.sum(1)) - W
    xu = torch.linalg.solve(Lp[u][:, u].double() + 1e-6 * torch.eye(int(u.sum()), device=dev, dtype=torch.float64),
                            -(Lp[u][:, mk].double() @ fgm[mk].double()))
    rw = torch.zeros(4096, device=dev, dtype=torch.float64); rw[fgm] = 1; rw[u] = xu; out["rw"] = rw.float().clamp(0, 1)
    # 9 low-pass
    d = W.sum(1).clamp_min(1e-12).rsqrt(); Sm = d[:, None] * W * d[None]
    try:
        if not Sm.is_cuda:
            raise RuntimeError("CPU MKL eigh fails at 4096")
        V = torch.linalg.eigh(Sm.float())[1].flip(1)[:, :32]
    except RuntimeError:
        from scipy.linalg import eigh
        V = torch.from_numpy(eigh(Sm.double().cpu().numpy(), subset_by_index=[4096 - 32, 4095])[1][:, ::-1].copy()).float().to(dev)
    for k in (8, 16, 32):
        out[f"lp{k}"] = norm01((V[:, :k] @ (V[:, :k].T @ s)).clamp_min(0))
    out["lp16_v3"] = V3(sa, out["lp16"])
    # 10 sinks / temperature
    cs = A.sum(0); keep = cs <= 5 * cs.mean(); Ad = A * keep[None]; Ad = Ad / Ad.sum(1, keepdim=True).clamp_min(1e-12)
    out["desink_p1"] = norm01(Ad @ s); out["n_sinks"] = torch.tensor(float((~keep).sum()))
    for T, nm in ((0.5, "temp05"), (2.0, "temp2")):
        AT = torch.stack([heads_attn(p, T).mean(0) for p in sa[64]]).mean(0); out[f"{nm}_p1"] = norm01(AT @ s)
        saT = {r: [(q.float() / T, k) for q, k in sa[r]] for r in (16, 32, 64)}; out[f"{nm}_v3"] = V3(saT, seed64.float())
    # 11 mean-field CRF with the SA kernel
    Wr = W / W.sum(1, keepdim=True).clamp_min(1e-12); p0 = out["brace"].clamp(0.01, 0.99); L0 = torch.log(p0 / (1 - p0))
    for w, nm in ((1.0, "sacrf_w1"), (3.0, "sacrf_w3")):
        Q = p0.clone()
        for _ in range(10):
            Q = torch.sigmoid(L0 + w * (Wr @ (2 * Q - 1)))
        out[nm] = Q
    # 12 rollout
    x = s.clone()
    for Ai in calls:
        x = 0.5 * (Ai @ x) + 0.5 * x
    out["rollout"] = norm01(x)
    # 8 graph cut (CPU)
    Wc = W.cpu().numpy(); kk = np.argpartition(-Wc, 17, axis=1)[:, :17]
    from scipy.sparse import coo_matrix
    rr = np.repeat(np.arange(4096), 17); Wk = coo_matrix((Wc[rr, kk.ravel()], (rr, kk.ravel())), shape=(4096, 4096)).tocsr()
    Wk.setdiag(0); Wk.eliminate_zeros(); Wk = Wk.maximum(Wk.T); Wk = Wk / Wk.data.mean()
    pb = out["brace"].cpu().numpy()
    for lam, nm in ((0.5, "gc_l05"), (2.0, "gc_l2")):
        out[nm] = torch.from_numpy(graph_cut(Wk, pb, lam).astype(np.float32))
    del calls, A, W, Sm, V, Lp
    # 13 re-noising ensemble
    ens = [out["brace"]]
    for t in RENOISE_T:
        sat = renoise_sa(pipe, modules, img, t); out[f"rn{t}_v3"] = V3(sat, seed64.float()); ens.append(out[f"rn{t}_v3"])
    E = torch.stack(ens); out["ens_mean"] = norm01(E.mean(0)); out["ens_conf"] = norm01((E.mean(0) - 0.5 * E.std(0)).clamp_min(0))
    return {k: (v.reshape(64, 64).float().cpu().numpy() if v.numel() == 4096 else float(v)) for k, v in out.items()}


def run(a):
    import multiprocessing as mp
    pool = ProcessPoolExecutor(a.crf_workers, mp_context=mp.get_context("forkserver"), initializer=oc.crf_init)
    from scripts import sg27_runtime
    from ideas_grouped import screen_ids
    ids = screen_ids(a.dataset); rows = [r for r in cc.gen_rows(a.dataset) if r["image_id"] in ids][a.shard::a.num_shards]
    if a.max_images:
        rows = rows[: a.max_images]
    runtime = sg27_runtime.make_runtime(MODEL); pipe = runtime["pipe"]; modules = runtime["modules"]
    out = WORK / a.dataset; out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        o = out / f"{row['image_id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); PROMPT[0] = row["prompt"]
        with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, _b, _p):
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            M = maps_for(sa, seed.float().clone(), img, pipe, modules)
        t1 = time.perf_counter(); hw = truth.shape
        res = {"cls": row["classname"], "area": float(truth.mean()), "n_sinks": M.pop("n_sinks")}
        grids = {k: st.up(v, hw) for k, v in M.items() if not k.startswith("gc_")}
        names = list(grids); masks = list(pool.map(oc.crf_grid, [(img, grids[k]) for k in names]))
        for k, mk in zip(names, masks):
            res[f"{k}|raw"] = oc.grid(grids[k], truth); res[f"{k}+R2|raw"] = oc.cnt(mk, truth)
        for k in ("gc_l05", "gc_l2"):
            mk = st.up(M[k], hw) >= 0.5
            res[f"{k}|mask"] = oc.cnt(mk, truth); res[f"{k}+R2|mask"] = oc.cnt(oc.crf(img, mk), truth)
        res["sec_gpu"] = t1 - t0; res["sec"] = time.perf_counter() - t0
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)
        print(f"[ops:{a.dataset}] {row['image_id']} {row['classname']} ops {t1 - t0:.1f}s total {res['sec']:.1f}s "
              f"sinks {res['n_sinks']:.0f}", flush=True)
        torch.cuda.empty_cache()
    pool.shutdown()


def report(a):
    S = {ds: oc.load(WORK / ds) for ds in ("voc", "coco")}
    L = ["# Operator batches B and C on the class-stratified 150/150 screening images (exploratory)\n",
         "`scripts/ana_ops_bc.py`. One frozen live capture per image; every map is paired with BRACE (V3) from the same "
         "capture. Ground truth only scores. Small subsets: treat differences below ~0.01 as noise. Nothing here is a "
         "registered test.\n"]
    for ds, s in S.items():
        if s is None:
            continue
        ns = np.array([float(z["n_sinks"]) for z in s["_Z"]]); sec = np.array([float(z["sec"]) for z in s["_Z"]])
        L += [f"## {oc.NAME[ds]} ({len(s['grp'])} images; sinks per image mean {ns.mean():.1f}; {sec.mean():.0f} s/image)\n"]
        L += oc.table(s, None, groups=GROUPS) + [""]
    txt = "\n".join(L) + "\n"; OUT.mkdir(parents=True, exist_ok=True); (OUT / "ops_bc.md").write_text(txt); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-images", type=int); p.add_argument("--crf-workers", type=int, default=8)
    sub.add_parser("report")
    a = ap.parse_args(); {"run": run, "report": report}[a.cmd](a)
