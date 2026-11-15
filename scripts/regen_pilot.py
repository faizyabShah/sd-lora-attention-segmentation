#!/usr/bin/env python3
"""Exploratory pilot: does attention-constrained regeneration rank candidate masks by correctness?
Plan and fixed pass bars: runs/04_regen_pilot/PILOT.md (committed before any GPU run).

Per image (20 screening images: 10 VOC-Sim + 10 COCO-Cap, chosen by `select`):
  1. frozen live capture -> BRACE V3 map -> DenseCRF R2 masks at 7 thresholds (the candidate masks)
  2. base trajectory with a manual PNDM loop equal to pipe() (keeps z_K0 and the scheduler state)
  3. for each candidate mask M: replay steps K0..30 from z_K0 with the class token's cross-attention constrained
     to M (conditional branch only, every attn2 layer and head, renormalised):
        outside M  class-token probability -> 0, the other tokens rescaled to sum to 1
        inside  M  class-token probability -> max(p, rho), rho = 90th percentile of p over the layer's pixels
                   (per head, per call, from the row's own unedited probabilities); others rescaled to 1 - p'
  4. record whole-image squared error of the decoded result vs the unconstrained replay image (primary),
     final-latent squared error (secondary), mean KL(p' || p) of the edit, the change in class-token mass,
     candidate IoU and area.
  Diagnostic candidates (ground truth used ONLY to build controls and to score; never by the method):
     gt, gt dilated / eroded by 8 and 16 px, gt shifted by a quarter image, gt flipped (same area elsewhere),
     empty, full, and an unconstrained replay (must reproduce the base image).

select   (CPU)  python scripts/regen_pilot.py select          -> runs/04_regen_pilot/pilot_ids.json
unittest (CPU)  python scripts/regen_pilot.py unittest        processor edit on random tensors
run      (GPU)  python scripts/regen_pilot.py run --dataset voc|coco   -> CACHE/regen_pilot/<ds>/<id>.npz
report   (CPU)  python scripts/regen_pilot.py report          -> runs/04_regen_pilot/result.md, fig/
"""
from __future__ import annotations

import argparse, copy, json, math, os, sys, time, zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402

RUN = ROOT / "runs/04_regen_pilot"
OUT = CACHE / "regen_pilot"
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
K0 = 8                     # intervention starts before step 8 of 31 PNDM timesteps (fixed, not tuned)
RHO_Q = 0.90               # inside-mask floor: 90th percentile of the class-token probability
TAUS = (0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75)
CFG = 7.5
BATCH = 4                  # candidates per UNet batch (x2 for classifier-free guidance)
N_PER = 10                 # images per dataset
CONTROLS = ("gt", "gt_dil8", "gt_dil16", "gt_ero8", "gt_ero16", "gt_shift", "gt_flip", "empty", "full")
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
STORED = {"voc": CACHE / "vocsim_scores", "coco": CACHE / "cococap_scores"}
crc = lambda s: zlib.crc32(str(s).encode())


# ------------------------------------------------------------------ the constrained cross-attention processor
class Ctl:
    def __init__(self):
        self.tokens = (); self.masks = None; self.active = None
        self.record = False; self.tok16 = None; self.n16 = 0
        self.kl = None; self.dmass = None; self.ncall = 0


def edit_probs(probs, H, ctl):
    """probs [2B*H, n, 77] (rows: uncond batch 0..B-1, then cond B..2B-1; head-minor). Returns edited probs and
    per-candidate (mean KL(p'||p), mean |delta class mass|) for this call."""
    B = probs.shape[0] // H // 2; n = probs.shape[1]; s = math.isqrt(n)
    P = probs.view(2 * B, H, n, probs.shape[-1])
    c = P[B:].float()                                                   # [B, H, n, 77]
    T = list(ctl.tokens)
    pt = c[..., T]; mt = pt.sum(-1, keepdim=True)                        # class-token mass [B, H, n, 1]
    rho = torch.quantile(mt[..., 0], RHO_Q, dim=2)[..., None, None]     # [B, H, 1, 1]
    inside = ctl.masks[s][:, None, :, None].to(c.device)                 # [B, 1, n, 1] bool
    new_t = torch.where(inside, torch.maximum(mt, rho), torch.zeros_like(mt))
    share = torch.where(mt > 0, pt / mt.clamp_min(1e-20), torch.full_like(pt, 1.0 / len(T)))
    rest = (1 - new_t) / (1 - mt).clamp_min(1e-12)
    out = c * rest
    out[..., T] = share * new_t
    act = ctl.active.to(c.device)[:, None, None, None]
    out = torch.where(act, out, c)
    kl = (out * (torch.log(out.clamp_min(1e-20)) - torch.log(c.clamp_min(1e-20)))).sum(-1).mean((1, 2))
    dm = (out[..., T].sum(-1) - mt[..., 0]).abs().mean((1, 2))
    P = P.clone(); P[B:] = out.to(P.dtype)
    return P.view_as(probs), kl, dm


class RegenProc:
    def __init__(self, ctl):
        self.ctl = ctl

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        batch_size, seq_len, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states); value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query); key = attn.head_to_batch_dim(key); value = attn.head_to_batch_dim(value)
        probs = attn.get_attention_scores(query, key, attention_mask)
        ctl, H = self.ctl, attn.heads
        if ctl.record and probs.shape[0] == 2 * H and probs.shape[1] == 256:
            m = probs[H:].mean(0).float(); ctl.tok16 = m if ctl.tok16 is None else ctl.tok16 + m; ctl.n16 += 1
        if ctl.masks is not None and ctl.tokens:
            probs, kl, dm = edit_probs(probs, H, ctl)
            ctl.kl += kl; ctl.dmass += dm; ctl.ncall += 1
        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states); hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install(pipe, ctl):
    saved = dict(pipe.unet.attn_processors); new = {}
    for name, p in saved.items():
        orig = getattr(p, "original", p)
        new[name] = RegenProc(ctl) if name.endswith("attn2.processor") else orig
    pipe.unet.set_attn_processor(new)
    return saved


def pooled_masks(M512):
    """[B, 512, 512] bool -> {side: [B, side*side] bool} by area pooling and >= 0.5 (same rule for every candidate)."""
    x = torch.from_numpy(np.asarray(M512, np.float32))[:, None]
    return {s: (F.adaptive_avg_pool2d(x, s)[:, 0] >= 0.5).reshape(len(M512), -1) for s in (8, 16, 32, 64)}


# ------------------------------------------------------------------ candidates
def controls(t):
    H, W = t.shape; r = lambda k: np.ones((2 * k + 1, 2 * k + 1), bool)
    return {"gt": t, "gt_dil8": binary_dilation(t, r(8)), "gt_dil16": binary_dilation(t, r(16)),
            "gt_ero8": binary_erosion(t, r(8)), "gt_ero16": binary_erosion(t, r(16)),
            "gt_shift": np.roll(t, (H // 4, W // 4), (0, 1)), "gt_flip": t[::-1, ::-1].copy(),
            "empty": np.zeros_like(t), "full": np.ones_like(t)}


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


# ------------------------------------------------------------------ batched replay from z_K0
def expand_sched(s, B):
    s = copy.deepcopy(s)
    s.ets = [e.repeat(B, 1, 1, 1) for e in s.ets]
    if isinstance(getattr(s, "cur_sample", None), torch.Tensor):
        s.cur_sample = s.cur_sample.repeat(B, 1, 1, 1)
    return s


@torch.no_grad()
def replay(pipe, sched, z, i0, ne, pe):
    B = z.shape[0]; enc = torch.cat([ne.expand(B, -1, -1), pe.expand(B, -1, -1)])
    for i in range(i0, len(sched.timesteps)):
        t = sched.timesteps[i]
        eps = pipe.unet(sched.scale_model_input(torch.cat([z] * 2), t), t, encoder_hidden_states=enc, return_dict=False)[0]
        eu, ec = eps.chunk(2)
        z = sched.step(eu + CFG * (ec - eu), t, z, return_dict=False)[0]
    return z


# ------------------------------------------------------------------ select (CPU)
def stored_iou(f):
    z = np.load(f, allow_pickle=True); C = z["base|V3+R2|raw"]
    return C[:, 0] / np.maximum(C[:, 0] + C[:, 1] + C[:, 2], 1), float(z["area"]), str(z["cls"])


def select(a):
    """Deterministic, diverse 10 per dataset from the class-stratified screening subset, using the stored paper
    capture (diagnostic selection; these 20 images must be excluded from any later confirmation). Only objects
    covering 1-90% of the image (shift / flip / full controls are degenerate otherwise). Then:
    5 with the largest oracle-tau gain over 0.55 (distinct classes), then 5 at the 10/30/50/70/90th percentile of
    object area among the rest (distinct classes where possible). Ties by crc32(image_id)."""
    from ideas_grouped import screen_ids
    j55 = int(np.flatnonzero(np.isclose(cc.THR_CRF, 0.55))[0]); out = {}
    for ds in ("voc", "coco"):
        rows = []
        for i in sorted(screen_ids(ds), key=crc):
            u, area, cls = stored_iou(STORED[ds] / f"{i}.npz")
            if 0.01 <= area <= 0.90:                                        # controls degenerate otherwise
                rows.append(dict(id=i, cls=cls, area=area, gain=float(u.max() - u[j55]), iou55=float(u[j55])))
        pick, used = [], set()
        for r in sorted(rows, key=lambda r: -r["gain"]):
            if r["cls"] not in used and len(pick) < 5:
                pick.append(r); used.add(r["cls"])
        rest = sorted([r for r in rows if r not in pick], key=lambda r: r["area"])
        for q in (0.1, 0.3, 0.5, 0.7, 0.9):
            j0 = int(q * (len(rest) - 1))
            near = [r for p, r in enumerate(rest) if abs(p - j0) <= 6 and r not in pick]   # nearest 13 by area rank
            r = min(near, key=lambda r: (r["cls"] in used, abs(rest.index(r) - j0)))
            pick.append(r); used.add(r["cls"])
        assert len(pick) == N_PER and len({r["id"] for r in pick}) == N_PER
        out[ds] = pick
    RUN.mkdir(parents=True, exist_ok=True); (RUN / "pilot_ids.json").write_text(json.dumps(out, indent=1))
    for ds in out:
        print(ds); [print(f"  {r['id']:40s} {r['cls']:12s} area {r['area']:.3f} iou@.55 {r['iou55']:.3f} oracle gain {r['gain']:+.3f}") for r in out[ds]]


# ------------------------------------------------------------------ unittest (CPU)
def unittest(a):
    torch.manual_seed(0); H, B, n = 8, 3, 256
    logits = torch.randn(2 * B * H, n, 77); probs = logits.softmax(-1)
    ctl = Ctl(); ctl.tokens = (5, 6)
    m = torch.zeros(B, 16, 16, dtype=torch.bool); m[:, 4:10, 4:10] = True
    ctl.masks = {16: m.reshape(B, -1)}; ctl.active = torch.tensor([True, True, False])
    out, kl, dm = edit_probs(probs, H, ctl)
    P = out.view(2 * B, H, n, 77); P0 = probs.view(2 * B, H, n, 77); inside = ctl.masks[16]
    ok = []
    ok.append(torch.allclose(P.sum(-1), torch.ones(2 * B, H, n), atol=1e-5))           # rows still sum to 1
    ok.append(torch.equal(P[:B], P0[:B]))                                                # unconditional untouched
    ok.append(torch.equal(P[B + 2], P0[B + 2]))                                          # inactive candidate untouched
    mt = P[B:B + 2][..., [5, 6]].sum(-1)
    ok.append(float(mt[:, :, ~inside[0]].abs().max()) < 1e-7)                             # outside -> 0
    rho = torch.quantile(P0[B:B + 2][..., [5, 6]].sum(-1), RHO_Q, dim=2)
    ok.append(bool((mt[:, :, inside[0]] >= rho[..., None] - 1e-6).all()))               # inside >= rho
    ratio = P[B:B + 2][..., :5] / P0[B:B + 2][..., :5]                                   # others rescaled uniformly per pixel
    ok.append(float((ratio - ratio[..., :1]).abs().max()) < 1e-4)
    ok.append(bool(kl[2] == 0) and bool(dm[2] == 0) and bool((kl[:2] > 0).all()))
    print("checks", ok, "kl", kl.tolist(), "dmass", dm.tolist())
    t = np.zeros((512, 512), bool); t[100:300, 150:350] = True; C = controls(t)
    print({k: (round(float(v.mean()), 3), round(iou(v, t), 3)) for k, v in C.items()})
    print({s: v.float().mean(1).tolist() for s, v in pooled_masks(np.stack([t, C['gt_dil16']])).items()})
    print("UNITTEST PASS" if all(ok) else "UNITTEST FAIL"); sys.exit(0 if all(ok) else 1)


# ------------------------------------------------------------------ run (GPU)
def run(a):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    from idea1_knockout import embeds, new_sched, denoise, decode, to_uint8
    from idea2_dose import choose_tokens
    _import_densecrf(st.CRF_SITE)
    ids = [r["id"] for r in json.loads((RUN / "pilot_ids.json").read_text())[a.dataset]]
    rows = [r for r in cc.gen_rows(a.dataset) if r["image_id"] in ids]
    assert len(rows) == len(ids), (len(rows), len(ids))
    runtime = sg27_runtime.make_runtime(MODEL); pipe = runtime["pipe"]
    out = OUT / a.dataset; out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter()
        with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, _b, _p):
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            seed64 = seed.float().clone()
            v3 = sg_combined_v2.aggregate(sa, seed64)[0]["V3"].float().cpu().numpy()
        xb = st.up(st.mm(v3), truth.shape)
        cand = {f"tau{t:.2f}": apply_ovam_densecrf(img, xb >= t, st.CRF_SITE).astype(bool) for t in TAUS}
        cand.update(controls(truth))
        names = ["replay"] + list(cand)
        masks = np.stack([np.zeros_like(truth)] + [cand[k] for k in names[1:]])
        steps = int(row["num_inferece_steps"])
        ctl = Ctl(); saved = install(pipe, ctl)
        try:
            pe, ne, noise = embeds(pipe, row); snap = {}

            def keep(i, z, sched):
                if i == K0:
                    snap["z"] = z.clone(); snap["s"] = copy.deepcopy(sched)
            ctl.record = True; zb = denoise(pipe, ctl, new_sched(pipe, steps), noise, 0, ne, pe, keep=keep); ctl.record = False
            tokens, how = choose_tokens(pipe, row, a.dataset, seed64, ctl.tok16 / max(ctl.n16, 1)); ctl.tokens = tokens
            xbase = decode(pipe, zb); base_img = to_uint8(xbase)
            pm = pooled_masks(masks)
            D_img, D_lat, KL, DM, thumbs = [], [], [], [], []
            for b0 in range(0, len(names), BATCH):
                sl = slice(b0, min(b0 + BATCH, len(names))); nb = sl.stop - sl.start
                ctl.masks = {s: v[sl] for s, v in pm.items()}
                ctl.active = torch.tensor([names[j] != "replay" for j in range(sl.start, sl.stop)])
                ctl.kl = torch.zeros(nb, device=pipe.device); ctl.dmass = torch.zeros(nb, device=pipe.device); ctl.ncall = 0
                zf = replay(pipe, expand_sched(snap["s"], nb), snap["z"].repeat(nb, 1, 1, 1), K0, ne, pe)
                KL += (ctl.kl / max(ctl.ncall, 1)).tolist(); DM += (ctl.dmass / max(ctl.ncall, 1)).tolist()
                for j in range(nb):
                    x = decode(pipe, zf[j:j + 1])
                    D_img.append(float(((x.float() - xbase.float()) ** 2).sum()))
                    D_lat.append(float(((zf[j].float() - zb[0].float()) ** 2).sum()))
                    thumbs.append(np.asarray(F.interpolate(x[None].float(), 128, mode="area")[0].permute(1, 2, 0).cpu().numpy() * 255, np.uint8))
                ctl.masks = None
        finally:
            pipe.unet.set_attn_processor(saved)
        res = dict(names=np.array(names), D_img=np.array(D_img), D_lat=np.array(D_lat), kl=np.array(KL), dmass=np.array(DM),
                   iou=np.array([iou(m, truth) if k != "replay" else np.nan for k, m in zip(names, masks)]),
                   area=masks.reshape(len(names), -1).mean(1), cls=cls, tokens=np.array(tokens), token_rule=how,
                   img_meandiff=float(np.abs(base_img.astype(int) - img.astype(int)).mean()),
                   thumbs=np.stack(thumbs), image128=np.asarray(F.interpolate(torch.from_numpy(img).permute(2, 0, 1)[None].float(), 128, mode="area")[0].permute(1, 2, 0).numpy(), np.uint8),
                   masks128=np.stack([np.asarray(F.adaptive_avg_pool2d(torch.from_numpy(m.astype(np.float32))[None, None], 128)[0, 0] >= 0.5) for m in masks]),
                   sec=time.perf_counter() - t0)
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)
        j = names.index("tau0.55")
        print(f"[regen:{a.dataset}] {iid} {cls} tokens {tokens} ({how}) replay D {D_img[0]:.3g}  tau0.55 D {D_img[j]:.3g}  "
              f"gt D {D_img[names.index('gt')]:.3g}  {res['sec']:.0f}s", flush=True)
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ report (CPU)
def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def report(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Z = {ds: [dict(np.load(f, allow_pickle=True)) for f in sorted((OUT / ds).glob("*.npz")) if not f.stem.endswith(".tmp")] for ds in ("voc", "coco")}
    allz = [(ds, z) for ds in Z for z in Z[ds]]
    if not allz:
        print("no results"); return
    L = ["# Regeneration pilot: does attention-constrained regeneration rank masks by correctness? (exploratory)\n",
         f"Plan and pass bars: `PILOT.md`. {len(allz)} images. Intervention from step {K0}, class-token floor at the "
         f"{RHO_Q:.0%} quantile, conditional branch only. D = whole-image squared error vs the unconstrained replay.\n"]
    rows, wins, sp_tau, sp_res, sel_gain, sel_better = [], {k: [] for k in ("gt_shift", "gt_flip", "gt_dil8", "gt_ero8", "gt_dil16", "gt_ero16", "empty", "full")}, [], [], [], []
    for ds, z in allz:
        nm = [str(x) for x in z["names"]]; D = z["D_img"]; U = z["iou"]; idx = {k: i for i, k in enumerate(nm)}
        for k in wins:
            wins[k].append(D[idx["gt"]] < D[idx[k]])
        ti = [idx[f"tau{t:.2f}"] for t in TAUS]
        sp_tau.append(spearman(D[ti], U[ti]))
        c = [i for i, k in enumerate(nm) if k != "replay"]
        X = np.c_[np.ones(len(c)), (z["area"][c] - z["area"][c].mean()) / (z["area"][c].std() + 1e-12),
                  (z["kl"][c] - z["kl"][c].mean()) / (z["kl"][c].std() + 1e-12)]
        y = (D[c] - D[c].mean()) / (D[c].std() + 1e-12); r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        sp_res.append(spearman(r, U[c]))
        js = ti[int(np.argmin(D[ti]))]; g = U[js] - U[idx["tau0.55"]]; sel_gain.append(g); sel_better.append(g > 1e-9)
        rows.append(f"| {NAME[ds]} | {str(z['cls'])} | {z['area'][idx['gt']]:.3f} | {D[0]:.2g} | {U[idx['tau0.55']]:.3f} | "
                    f"{nm[js][3:]} → {U[js]:.3f} | {max(U[ti]):.3f} | {sp_tau[-1]:+.2f} | {sp_res[-1]:+.2f} | "
                    + " ".join("✓" if wins[k][-1] else "·" for k in wins) + " |")
    n = len(allz)
    L += ["| set | class | object area | replay D | IoU @0.55 | selected τ → IoU | oracle τ IoU | ρ(D, IoU) over τ | ρ(resid D, IoU) all | gt beats: " + " ".join(wins) + " |",
          "|---|---|---:|---:|---:|---|---:|---:|---:|---|"] + rows + [""]
    bars = {
        "1. gt beats shift, flip, dil8, ero8 on >= 15/20 each": all(sum(wins[k]) >= 15 for k in ("gt_shift", "gt_flip", "gt_dil8", "gt_ero8")),
        "2. median within-image Spearman(D, IoU) over the 7 tau masks <= -0.5": float(np.nanmedian(sp_tau)) <= -0.5,
        "3. selected tau beats 0.55: mean IoU gain > 0 and strictly better on >= 12/20": float(np.mean(sel_gain)) > 0 and sum(sel_better) >= 12,
        "4. median Spearman(D residual after area + KL, IoU) over all candidates <= -0.3": float(np.nanmedian(sp_res)) <= -0.3}
    L += [f"gt beats each control (of {n}): " + ", ".join(f"{k} {sum(v)}" for k, v in wins.items()),
          f"\nmedian ρ(D, IoU) over τ masks {np.nanmedian(sp_tau):+.2f}; median ρ(resid D, IoU) {np.nanmedian(sp_res):+.2f}; "
          f"selected τ vs 0.55: mean IoU gain {np.mean(sel_gain):+.3f}, better on {sum(sel_better)}/{n}\n",
          "## Pass bars (fixed in PILOT.md before the run)\n"] + [f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in bars.items()]
    L.append(f"\n**Verdict: {'viable — freeze the rule and screen' if all(bars.values()) else 'not viable as specified — close without tuning'}**"
             + (" (fewer than 20 images: provisional)" if n < 20 else "") + "\n")
    # figure: per image, IoU vs D coloured by candidate type
    fig, ax = plt.subplots(4, 5, figsize=(20, 15)); ax = ax.ravel()
    col = lambda k: "tab:blue" if k.startswith("tau") else ("tab:green" if k == "gt" else ("tab:orange" if k.startswith("gt_") else "tab:gray"))
    for i, (ds, z) in enumerate(allz[:20]):
        nm = [str(x) for x in z["names"]]
        for j, k in enumerate(nm):
            if k == "replay":
                continue
            ax[i].scatter(z["D_img"][j], z["iou"][j], c=col(k), s=30); ax[i].annotate(k.replace("tau", ""), (z["D_img"][j], z["iou"][j]), fontsize=6)
        ax[i].set_title(f"{NAME[ds]} {z['cls']}", fontsize=9); ax[i].set_xlabel("D (whole-image sq. error)", fontsize=7); ax[i].set_ylabel("IoU", fontsize=7)
    fig.tight_layout(); (RUN / "fig").mkdir(parents=True, exist_ok=True); fig.savefig(RUN / "fig/scatter.png", dpi=80); plt.close(fig)
    # figure: thumbnails for the first 6 images
    sel = ["tau0.35", "tau0.55", "tau0.75", "gt", "gt_dil16", "gt_shift", "empty", "full"]
    fig, ax = plt.subplots(min(6, n), len(sel) + 1, figsize=(2 * (len(sel) + 1), 2 * min(6, n)), squeeze=False)
    for i, (ds, z) in enumerate(allz[:6]):
        nm = [str(x) for x in z["names"]]; ax[i, 0].imshow(z["image128"]); ax[i, 0].set_title(str(z["cls"]), fontsize=8)
        for j, k in enumerate(sel):
            q = nm.index(k); ax[i, j + 1].imshow(z["thumbs"][q]); ax[i, j + 1].contour(z["masks128"][q], [0.5], colors="r", linewidths=0.6)
            ax[i, j + 1].set_title(f"{k} D={z['D_img'][q]:.0f}", fontsize=7)
    [x.axis("off") for x in ax.ravel()]; fig.tight_layout(); fig.savefig(RUN / "fig/regen_examples.jpg", dpi=70); plt.close(fig)
    L.append("Figures: `fig/scatter.png` (IoU vs D per image; blue τ masks, green gt, orange gt controls, grey empty/full), "
             "`fig/regen_examples.jpg` (regenerated images under several masks; red = the imposed mask).")
    txt = "\n".join(L) + "\n"; (RUN / "result.md").write_text(txt); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    for c in ("select", "unittest", "report"):
        sub.add_parser(c)
    a = ap.parse_args(); {"run": run, "select": select, "unittest": unittest, "report": report}[a.cmd](a)
