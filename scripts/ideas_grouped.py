#!/usr/bin/env python3
"""Grouped run of ideas 1 (knockout), 3 (chrono), 4 (Jacobian), 6 (OT), 7 (denoising contrast) on VOC-Sim / COCO-Cap.
Pre-registered in runs/03_grouped/PREREG.md. Only per-image score counts are written; no maps are stored.

Per image, one shared generation:
  1. base trajectory (manual PNDM loop == pipe(), no safety checker): z_i, scheduler states at k, eps_u / eps_c,
     x0 predictions, 16x16 and 32x32 conditional token attention
  2. frozen live capture -> BRACE (A3 seed -> V3 -> R2); inside it: target tokens, OT seed -> V3 with the live SA
  3. knockout (instantaneous along the base, and branches from z_k), Jacobian VJPs at steps 10 / 20
  4. chrono k-means, CFG contrast maps
  5. every map -> counts on the tau grid, raw and + R2 (CRF in a CPU process pool)

selftest (GPU)  python scripts/ideas_grouped.py selftest
run      (GPU)  python scripts/ideas_grouped.py run --dataset voc|coco [--shard i --num-shards n] [--max-images k] [--crf-workers w]
report   (CPU)  python scripts/ideas_grouped.py report
"""
from __future__ import annotations

import argparse, copy, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402
from idea2_dose import choose_tokens  # noqa: E402
import idea1_knockout as k1  # noqa: E402  (manual loop, embeddings, swap, post-processing: unchanged)

RUN = ROOT / "runs/03_grouped"
MODEL = k1.MODEL
KS = k1.KS
JAC_STEPS = (10, 20); N_PROBES = 4
CHRONO_K = (4, 8, 16)
OT_EPS, OT_ITERS = 0.05, 200
GROUPS = {
    "knockout (idea 1)": k1.CANDS,
    "chrono (idea 3)": tuple(f"chrono_K{k}" for k in CHRONO_K),
    "jacobian (idea 4)": tuple(f"jac_s{s}" for s in JAC_STEPS),
    "optimal transport (idea 6)": ("ot", "ot_v3"),
    "denoising contrast (idea 7)": ("cfg_k0", "cfg_k10"),
}
CANDS = tuple(c for g in GROUPS.values() for c in g)
TAU = 0.55
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}


# ------------------------------------------------------------------ processor: knockout + 16/32 token recording
class Control(k1.Control):
    def __init__(self):
        super().__init__(); self.tok = {256: None, 1024: None}; self.n = {256: 0, 1024: 0}


class Proc(k1.KOCrossProc):
    """k1.KOCrossProc, additionally recording 32x32 token attention. Same arithmetic."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        ctl = self.ctl
        if ctl.record and encoder_hidden_states is not None:
            orig = attn.get_attention_scores

            def scores(q, k, mask=None):
                p = orig(q, k, mask); H = attn.heads
                if p.shape[0] == 2 * H and p.shape[1] in ctl.tok:
                    m = p[H:].mean(0).float(); N = p.shape[1]
                    ctl.tok[N] = m if ctl.tok[N] is None else ctl.tok[N] + m; ctl.n[N] += 1
                return p
            attn.get_attention_scores = scores
            try:
                rec, ctl.record = ctl.record, False       # k1's own 16x16 recorder is not used here
                return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)
            finally:
                ctl.record = rec; del attn.get_attention_scores
        return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)


def install(pipe, ctl):
    saved = dict(pipe.unet.attn_processors); new = {}
    for name, p in saved.items():
        orig = getattr(p, "original", p)
        new[name] = Proc(ctl) if name.endswith("attn2.processor") else orig
    pipe.unet.set_attn_processor(new)
    return saved


# ------------------------------------------------------------------ base trajectory with records
@torch.no_grad()
def base_run(pipe, ctl, row):
    steps = int(row["num_inferece_steps"]); pe, ne, noise = k1.embeds(pipe, row)
    sched = k1.new_sched(pipe, steps); ts = sched.timesteps; ac = sched.alphas_cumprod.to(pipe.device)
    z = noise * sched.init_noise_sigma; enc = torch.cat([ne, pe])
    zs, snaps, gdiff, x0 = [], {}, [], []
    ctl.record = True
    for i in range(len(ts)):
        t = ts[i]; zs.append(z.clone())
        if i in KS:
            snaps[i] = copy.deepcopy(sched)
        eps = pipe.unet(sched.scale_model_input(torch.cat([z] * 2), t), t, encoder_hidden_states=enc, return_dict=False)[0]
        eu, ec = eps.chunk(2); eg = eu + k1.CFG * (ec - eu)
        gdiff.append((ec - eu)[0].float().norm(dim=0))
        a = ac[int(t)]; x0.append(((z - (1 - a).sqrt() * eg) / a.sqrt())[0].float())
        z = sched.step(eg, t, z, return_dict=False)[0]
    ctl.record = False
    tok = {N: ctl.tok[N] / max(ctl.n[N], 1) for N in ctl.tok}
    return dict(pe=pe, ne=ne, sched=sched, zs=zs, snaps=snaps, zb=z, gdiff=torch.stack(gdiff), x0=torch.stack(x0), tok=tok)


# ------------------------------------------------------------------ signals
def kmeans(X, K, restarts=4, iters=50):
    g = torch.Generator(device="cpu").manual_seed(0); best, binert = None, float("inf")
    for _ in range(restarts):
        C = X[torch.randint(len(X), (1,), generator=g)]
        for _ in range(K - 1):                            # k-means++
            d = torch.cdist(X, C).min(1).values ** 2
            C = torch.cat([C, X[torch.multinomial(d / d.sum(), 1, generator=g)]])
        for _ in range(iters):
            lab = torch.cdist(X, C).argmin(1)
            C = torch.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(K)])
        lab = torch.cdist(X, C).argmin(1); inert = float(((X - C[lab]) ** 2).sum())
        if inert < binert:
            best, binert = lab, inert
    return best


def chrono_maps(x0, seed64):
    X = x0.permute(2, 3, 0, 1).reshape(4096, -1).double().cpu()          # [4096, 31*4]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8); s = torch.from_numpy(st.mm(seed64)).reshape(-1).double()
    out = {}
    for K in CHRONO_K:
        lab = kmeans(X, K); m = torch.zeros(4096, dtype=torch.float64)
        for j in range(K):
            if (lab == j).any():
                m[lab == j] = s[lab == j].mean()
        out[f"chrono_K{K}"] = st.mm(m.reshape(64, 64).numpy())
    return out


def ot_map(tok32, content, tokens, seed64):
    P = tok32.double().cpu()                              # [1024, 77]
    tgt = list(tokens); others = [c for c in content if c not in tokens]
    bg = [j for j in range(77) if j not in tgt and j not in others]
    Q = torch.stack([P[:, tgt].sum(1)] + [P[:, j] for j in others] + [P[:, bg].sum(1)], 1)
    Q = Q / Q.sum(1, keepdim=True)
    a_hat = float(np.clip(st.a_hat(seed64), 0.02, 0.98))
    rest = Q[:, 1:].mean(0); b = torch.cat([torch.tensor([a_hat], dtype=torch.float64), (1 - a_hat) * rest / rest.sum()])
    la = torch.full((1024,), -np.log(1024.0), dtype=torch.float64); lb = b.log()
    M = torch.log(Q + 1e-8) / OT_EPS                      # -cost / eps
    f = torch.zeros(1024, dtype=torch.float64); h = torch.zeros(len(b), dtype=torch.float64)
    for _ in range(OT_ITERS):
        f = la - torch.logsumexp(M + h[None], 1)
        h = lb - torch.logsumexp(M + f[:, None], 0)
    Pi = torch.exp(M + f[:, None] + h[None])
    return (Pi[:, 0] * 1024).reshape(32, 32).float().numpy(), a_hat


def jac_maps(pipe, ctl, B, seed64):
    out = {}; s = torch.from_numpy(st.mm(seed64)).to(pipe.device)
    g = torch.Generator(device="cpu").manual_seed(0)
    probes = [(torch.randn(1, 4, 64, 64, generator=g).to(pipe.device) * s) for _ in range(N_PROBES)]
    ts = B["sched"].timesteps
    for i in JAC_STEPS:
        z = B["zs"][i].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            eps = pipe.unet(B["sched"].scale_model_input(z, ts[i]), ts[i], encoder_hidden_states=B["pe"], return_dict=False)[0]
            acc = torch.zeros(64, 64, device=pipe.device)
            for r, u in enumerate(probes):
                v = torch.autograd.grad(eps, z, grad_outputs=u, retain_graph=r < N_PROBES - 1)[0]
                acc += v[0].float().norm(dim=0)
        out[f"jac_s{i}"] = (acc / N_PROBES).detach().cpu().numpy()
        del eps, z
    return out


@torch.no_grad()
def knockout_maps(pipe, ctl, B, tokens, ps):
    ctl.tokens = tokens; ts = B["sched"].timesteps; out = {}; ins = {m: [] for m in k1.MODES}
    for i in range(min(KS), len(ts)):
        zi = B["sched"].scale_model_input(B["zs"][i].repeat(3, 1, 1, 1), ts[i])
        ctl.ko_rows = (1,)
        e = pipe.unet(zi, ts[i], encoder_hidden_states=torch.cat([B["pe"], B["pe"], ps]), return_dict=False)[0].float()
        ctl.ko_rows = ()
        ins["attn"].append((e[1] - e[0]).norm(dim=0)); ins["swap"].append((e[2] - e[0]).norm(dim=0))
    for m in k1.MODES:
        E = torch.stack(ins[m])
        for k in KS:
            out[f"{m}_ins_k{k}"] = E[k - min(KS):].sum(0).cpu().numpy()
    for k in KS:
        za = k1.denoise(pipe, ctl, copy.deepcopy(B["snaps"][k]), B["zs"][k], k, B["ne"], B["pe"], ko_rows=(1,))
        zw = k1.denoise(pipe, ctl, copy.deepcopy(B["snaps"][k]), B["zs"][k], k, B["ne"], ps)
        out[f"attn_fin_k{k}"] = (za - B["zb"])[0].float().norm(dim=0).cpu().numpy()
        out[f"swap_fin_k{k}"] = (zw - B["zb"])[0].float().norm(dim=0).cpu().numpy()
    return out


# ------------------------------------------------------------------ CRF pool
def _crf_init():
    from src.segmentation.ovam_full import _import_densecrf
    _import_densecrf(st.CRF_SITE)


def _crf_job(args):
    from src.segmentation.ovam_full import apply_ovam_densecrf
    img, x = args
    return np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in cc.THR_CRF]).astype(bool)


# ------------------------------------------------------------------ selftest (GPU)
def selftest(a):
    from scripts import sg27_runtime
    runtime = sg27_runtime.make_runtime(MODEL); pipe = runtime["pipe"]; pipe.unet.requires_grad_(False)
    row = cc.gen_rows("voc")[0]; steps = int(row["num_inferece_steps"])
    ctl = Control(); saved = install(pipe, ctl)
    try:
        pe, ne, noise = k1.embeds(pipe, row)
        ref = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, latents=noise.clone(), height=512, width=512,
                   num_inference_steps=steps, guidance_scale=k1.CFG, output_type="latent").images.float()
        ctl.tok = {256: None, 1024: None}; ctl.n = {256: 0, 1024: 0}
        B = base_run(pipe, ctl, row); d1 = float((B["zb"].float() - ref).abs().max())
        torch.cuda.reset_peak_memory_stats()
        J = jac_maps(pipe, ctl, B, np.random.default_rng(0).random((64, 64)).astype(np.float32)); mem = torch.cuda.max_memory_allocated() / 2**30
    finally:
        pipe.unet.set_attn_processor(saved)
    n16, n32 = ctl.n[256], ctl.n[1024]
    print(f"base loop (with recording) vs pipe(): max |Δlatent| {d1:.3g} (expect < 1e-4)")
    print(f"token-attention calls recorded: 16x16 {n16}, 32x32 {n32} (expect > 0 and divisible by 31)")
    print(f"Jacobian maps finite: {all(np.isfinite(v).all() for v in J.values())}; peak GPU memory during VJP {mem:.1f} GiB")
    ok = d1 < 1e-4 and n16 > 0 and n32 > 0 and n16 % 31 == 0 and n32 % 31 == 0 and all(np.isfinite(v).all() for v in J.values())
    print("SELFTEST PASS" if ok else "SELFTEST FAIL"); sys.exit(0 if ok else 1)


# ------------------------------------------------------------------ run (GPU)
def run(a):
    import multiprocessing as mp
    pool = ProcessPoolExecutor(a.crf_workers, mp_context=mp.get_context("forkserver"), initializer=_crf_init)
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2, sg24_ca
    rows = cc.gen_rows(a.dataset)[a.shard::a.num_shards]
    if a.max_images:
        rows = rows[: a.max_images]
    runtime = sg27_runtime.make_runtime(MODEL); pipe = runtime["pipe"]; pipe.unet.requires_grad_(False)
    out = CACHE / f"grouped_{a.dataset}"; out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); maps64, maps_hw = {}, {}
        # 1. base trajectory
        ctl = Control(); saved = install(pipe, ctl)
        try:
            B = base_run(pipe, ctl, row); base_img = k1.to_uint8(k1.decode(pipe, B["zb"]))
        finally:
            pipe.unet.set_attn_processor(saved)
        t1 = time.perf_counter()
        # 2. frozen live capture -> BRACE; tokens; OT seed through V3 with the live SA
        with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, _b, _p):
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            seed64 = seed.float().clone()
            v3 = sg_combined_v2.aggregate(sa, seed64)[0]["V3"].float().cpu().numpy()
            tokens, how = choose_tokens(pipe, row, a.dataset, seed64, B["tok"][256])
            try:
                content = sg24_ca.content_positions(pipe.tokenizer, row["prompt"], tuple(tokens))
            except ValueError:
                content = tuple(tokens)
            ot32, a_hat = ot_map(B["tok"][1024], content, tokens, seed64.cpu().numpy())
            ot64 = torch.from_numpy(st.up(ot32, (64, 64))).to(seed64.device)
            v3_ot = sg_combined_v2.aggregate(sa, ot64)[0]["V3"].float().cpu().numpy()
        s64 = seed64.cpu().numpy()
        t2 = time.perf_counter()
        # 3. knockout, Jacobian
        ctl = Control(); saved = install(pipe, ctl)
        try:
            ps = k1.swap_embeds(pipe, row, tokens)
            maps64.update(knockout_maps(pipe, ctl, B, tokens, ps))
            maps64.update(jac_maps(pipe, ctl, B, s64))
        finally:
            pipe.unet.set_attn_processor(saved)
        # 4. CFG contrast, chrono
        for k in (0, 10):
            maps64[f"cfg_k{k}"] = B["gdiff"][k:].sum(0).cpu().numpy()
        hw = truth.shape
        maps = {c: k1.post(maps64[c], hw) for c in maps64}
        maps.update({c: st.up(v, hw) for c, v in chrono_maps(B["x0"], s64).items()})
        maps["ot"] = st.mm(st.up(ot32, hw)); maps["ot_v3"] = st.up(st.mm(v3_ot), hw)
        img_diff = float(np.abs(base_img.astype(int) - img.astype(int)).mean())
        t3 = time.perf_counter()
        # 5. counts
        valid = np.ones_like(truth); other = np.zeros_like(truth)
        res = {"cls": cls, "area": float(truth.mean())}
        xb = st.up(st.mm(v3), hw)
        jobs = [(img, xb)] + [(base_img, maps[c]) for c in CANDS]
        masks = list(pool.map(_crf_job, jobs))
        res["brace|raw"] = cc.counts(xb[None] >= cc.THR[:, None, None], truth, valid, other)
        res["brace+R2|raw"] = cc.counts(masks[0], truth, valid, other)
        for c, m in zip(CANDS, masks[1:]):
            res[f"{c}|raw"] = cc.counts(maps[c][None] >= cc.THR[:, None, None], truth, valid, other)
            res[f"{c}+R2|raw"] = cc.counts(m, truth, valid, other)
        res.update(tokens=np.array(tokens), token_rule=how, img_meandiff=img_diff, a_hat=a_hat,
                   sec_base=t1 - t0, sec_live=t2 - t1, sec_signals=t3 - t2, sec_total=time.perf_counter() - t0)
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)
        print(f"[grouped:{a.dataset}] {iid} {cls} tokens {tokens} ({how}) base-vs-live |Δpix| {img_diff:.2f} "
              f"base {t1 - t0:.1f}s live {t2 - t1:.1f}s signals {t3 - t2:.1f}s total {time.perf_counter() - t0:.1f}s", flush=True)
        del B, maps64, maps, masks; torch.cuda.empty_cache()
    pool.shutdown()


# ------------------------------------------------------------------ report (CPU)
def boot2(S, Pa, Pb, n=1000):
    rng = np.random.default_rng(0); g = np.unique(S["grp"]); idx = {x: np.where(S["grp"] == x)[0] for x in g}; d = []
    for _ in range(n):
        s = np.concatenate([idx[x] for x in rng.choice(g, len(g))]); sel = np.ones(len(s), bool)
        d.append(cc.miou(Pa[s][:, None], S["cls"][s], sel)[0] - cc.miou(Pb[s][:, None], S["cls"][s], sel)[0])
    return np.percentile(d, [2.5, 97.5]), np.percentile(d, [0.5, 99.5])


def report(a):
    j55 = int(np.flatnonzero(np.isclose(cc.THR_CRF, TAU))[0])
    S = {}
    for ds in ("voc", "coco"):
        d = CACHE / f"grouped_{ds}"
        if d.exists() and any(d.glob("*.npz")):
            S[ds] = cc.load_gen(d)
            Z = [np.load(f) for f in sorted(f for f in d.glob("*.npz") if not f.stem.endswith(".tmp"))]
            for key, name in (("rule", "token_rule"),):
                S[ds][key] = np.array([str(z[name]) for z in Z])
            S[ds]["sec"] = np.array([float(z["sec_total"]) for z in Z])
            S[ds]["imgdiff"] = np.array([float(z["img_meandiff"]) for z in Z])
    one = lambda ds: np.ones(len(S[ds]["grp"]), bool)
    memo = {}

    def oof(ds, k):
        if (ds, k) not in memo:
            memo[ds, k] = cc.oof(S[ds], S[ds][f"{k}+R2|raw"])
        return memo[ds, k]
    L = ["# Grouped run: ideas 1, 3, 4, 6, 7 (pre-registered in `PREREG.md`)\n",
         "One shared generation per image. BRACE = frozen V3 + R2 from the same process. Class-macro mIoU; "
         "OOF = τ chosen on 4 folds (crc32 % 5); Δ vs BRACE OOF + R2 with image-bootstrap 95% CI.\n"]
    for ds in S:
        n = len(S[ds]["grp"]); ref = oof(ds, "brace"); dd = S[ds]["imgdiff"]
        L += [f"## {NAME[ds]} ({n} images; base-loop vs live mean |Δpixel| {dd.mean():.2f}/255, "
              f"{(dd > 2).sum()} images > 2/255, max {dd.max():.2f}; {S[ds]['sec'].mean():.1f} s/image)\n",
              "| group | map | OOF + R2 | Δ vs BRACE (95% CI) | prec / rec | OOF raw | @0.55 + R2 |",
              "|---|---|---:|---|---|---:|---:|"]
        for g, ks in [("—", ("brace",))] + list(GROUPS.items()):
            for k in ks:
                m, P, _ = oof(ds, k)
                if k == "brace":
                    d = "—"
                else:
                    (lo, hi), _ = boot2(S[ds], P, ref[1]); d = f"{m - ref[0]:+.3f} ({lo:+.3f}…{hi:+.3f})"
                pr = cc.prf(P); raw = cc.oof(S[ds], S[ds][f"{k}|raw"])[0]
                f55 = cc.miou(S[ds][f"{k}+R2|raw"][:, j55][:, None], S[ds]["cls"], one(ds))[0]
                L.append(f"| {g} | {'BRACE (frozen)' if k == 'brace' else k} | {m:.3f} | {d} | {pr[0]:.3f} / {pr[1]:.3f} | {raw:.3f} | {f55:.3f} |")
        if ds == "coco":
            L.append("")
            for r in ("class_word", "fallback"):
                sel = S[ds]["rule"] == r
                L.append(f"- token rule **{r}**: {sel.sum()} images ({sel.mean() * 100:.0f}%); OOF + R2 on this subset: " + ", ".join(
                    f"{k} {cc.miou(oof(ds, k)[1][:, None], S[ds]['cls'], sel)[0]:.3f}" for k in ("brace",) + CANDS))
        L.append("\nPer class, OOF + R2 (BRACE / best of each group on this set):\n")
        best = [max(ks, key=lambda k: oof(ds, k)[0]) for ks in GROUPS.values()]
        L += ["| class | BRACE | " + " | ".join(best) + " |", "|---|" + "---:|" * (len(best) + 1)]
        for c in np.unique(S[ds]["cls"]):
            L.append(f"| {c} | " + " | ".join(f"{cc.miou(oof(ds, k)[1][:, None], S[ds]['cls'], S[ds]['cls'] == c)[0]:.2f}"
                                             for k in ("brace",) + tuple(best)) + " |")
        L.append("")
    res = {}
    L.append("## Registered tests (one per group)\n")
    if "voc" in S and "coco" in S and len(S["voc"]["grp"]) == 490 and len(S["coco"]["grp"]) == 707:
        rb = oof("coco", "brace")
        L += ["| group | chosen on VOC-Sim (OOF + R2) | COCO-Cap OOF + R2 vs BRACE | Δ | 95% CI | 99% CI (Bonferroni, 5 groups) | verdict |",
              "|---|---|---|---:|---|---|---|"]
        for g, ks in GROUPS.items():
            ch = max(ks, key=lambda k: oof("voc", k)[0]); m, P, _ = oof("coco", ch)
            (lo, hi), (lo99, hi99) = boot2(S["coco"], P, rb[1])
            verdict = "beats BRACE" if lo99 > 0 else ("95% CI > 0 only (not claimable family-wise)" if lo > 0 else "no gain over BRACE")
            L.append(f"| {g} | {ch} ({oof('voc', ch)[0]:.3f} vs BRACE {oof('voc', 'brace')[0]:.3f}) | {m:.3f} vs {rb[0]:.3f} | "
                     f"{m - rb[0]:+.3f} | {lo:+.3f}…{hi:+.3f} | {lo99:+.3f}…{hi99:+.3f} | **{verdict}** |")
            res[g] = {"chosen": ch, "coco_delta": m - rb[0], "ci95": [lo, hi], "ci99": [lo99, hi99], "verdict": verdict}
        g1 = res["knockout (idea 1)"]
        L.append(f"\nIdea-1 rule as originally registered (`runs/01_knockout/PREREG.md`, 95% CI > 0): "
                 f"**{'knockout beats BRACE' if g1['ci95'][0] > 0 else 'no gain over BRACE'}**.")
    else:
        L.append("Not evaluated: both full sets (490 / 707) are required.")
    for ds in S:
        res[ds] = {k: oof(ds, k)[0] for k in ("brace",) + CANDS} | {"n": len(S[ds]["grp"])}
    txt = "\n".join(L) + "\n"; print(txt)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "result.md").write_text(txt); (RUN / "result.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-images", type=int); p.add_argument("--crf-workers", type=int, default=4)
    sub.add_parser("report"); sub.add_parser("selftest")
    a = ap.parse_args()
    {"run": run, "report": report, "selftest": selftest}[a.cmd](a)
