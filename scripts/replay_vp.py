#!/usr/bin/env python3
"""(V) SA value-space compatibility edge diagnostic and (P) semantic subspace purification of the class context.
Pre-registered in runs/13_value_purify/PREREG.md. Replays the stored trajectories with the EXACT A3 seed weighting.

python scripts/replay_vp.py run --dataset voc|coco [--limit k] [--tag t]   GPU -> CACHE/replay_vp[_t]/<ds>/<id>.npz
python scripts/replay_vp.py score [--tag t]                                CPU -> runs/13_value_purify/result.md
"""
from __future__ import annotations

import argparse, json, math, os, sys, time, zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brace_paths import CACHE  # noqa: E402

RUN = ROOT / "runs/13_value_purify"; DUMP = CACHE / "ana_dump"
A3_IDX = (0, 1, 4, 5, 6, 7, 8, 9, 13, 14, 15); SA_CALLS = range(10, 21)
NUIS = ("wall", "floor", "ground", "sky", "grass", "tree", "building", "road")
ARMS = ("live", "base", "pur1", "pur0.5", "rand", "meandir")
KTOP, EMAX = 16, 2000
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
BASEF = ["dlab", "contour", "dist", "aff", "featcos", "conf_min", "conf_max"]


def work(tag):
    return CACHE / ("replay_vp" + (f"_{tag}" if tag else ""))


class Cap:
    def __init__(self, name, roles, st):
        self.name, self.roles, self.st = name, roles, st

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *a, **k):
        from diffusers.models.attention_processor import AttnProcessor
        st, c = self.st, self.st["call"]
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        if "paired" in self.roles or ("brace" in self.roles and c in SA_CALLS):
            kk = attn.head_to_batch_dim(attn.to_k(hidden_states))
            if "paired" in self.roles:
                st["paired"][self.roles["paired"]] = (q.detach().half(), kk.detach().half())
            if "brace" in self.roles and c in SA_CALLS:
                side = self.roles["brace"]; st["sa"][side].append((q.detach(), kk.detach()))
                if side == 64:
                    v = attn.to_v(hidden_states)[0].float()                                  # [4096, 320]
                    st["vn"].append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        if "a3" in self.roles:
            st["q"][self.name].append(q.detach().half())
        if "ref" in self.roles:
            st["ref"] = q.detach().permute(1, 0, 2).reshape(q.shape[1], -1).float()
        return AttnProcessor()(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


def orth(X):
    import torch
    Q, _ = torch.linalg.qr(X.T); return Q                                                  # [768, r]


def run(a):
    import torch, torch.nn.functional as F
    from scripts import sg27_runtime
    from src.segmentation.sg25_sa import _apply, _attention, _resize
    from src.segmentation.sg_combined_v2 import cap
    from src.segmentation.sg24_ca import affinity_similarity, REF_SUFFIX
    from src.segmentation.attention_aggregation import find_token_subsequence
    import idea1_knockout as k1, sg_class_comp as cc, sg_steps_test as st, ana_ops_common as oc, sa_boundary_diag as sbd
    from ideas_grouped import screen_ids
    from scipy.ndimage import gaussian_filter, sobel
    oc.crf_init()
    ids = sorted(screen_ids(a.dataset))[: a.limit or None]
    rows = {r["image_id"]: r for r in cc.gen_rows(a.dataset)}
    rt = sg27_runtime.make_runtime(k1.MODEL); pipe = rt["pipe"]; dev = pipe.device; mods = rt["modules"]
    sched = pipe.scheduler; sched.set_timesteps(30, device=dev); TS = [int(t) for t in sched.timesteps]
    procs = dict(pipe.unet.attn_processors)
    attn2 = sorted(n.removesuffix(".processor") for n in procs if n.endswith("attn2.processor"))
    a3 = [attn2[i] for i in A3_IDX]
    if tuple(a3) != tuple(rt["ca_modules"]):
        raise ValueError("A3 module list differs from the runtime's frozen CA modules")
    roles = {}
    for m in a3:
        roles.setdefault(m, {})["a3"] = 1; roles.setdefault(m[:-5] + "attn1", {})["paired"] = m
    for s in (16, 32, 64):
        roles.setdefault(mods[s], {})["brace"] = s
    roles.setdefault(REF_SUFFIX, {})["ref"] = 1
    tok = pipe.tokenizer

    def ctx(text):
        ids77 = tok(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            return ids77, pipe.text_encoder(ids77)[0][0].float()
    NV = []
    for w in NUIS:
        ids77, Hw = ctx(w); pos = find_token_subsequence(ids77[0].tolist(), tok(w, add_special_tokens=False)["input_ids"])
        NV.append(Hw[list(pos)].mean(0))
    NV = torch.stack(NV); N8 = orth(NV); N1 = orth(NV.mean(0, keepdim=True))
    g = torch.Generator(device="cpu").manual_seed(0); NR = orth(torch.randn(8, 768, generator=g).to(dev))
    out = work(a.tag) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    for iid in ids:
        o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); z = np.load(DUMP / a.dataset / f"{iid}.npz", allow_pickle=True); row = rows[iid]
        x0 = torch.from_numpy(z["x0"].astype(np.float32)).to(dev); img = z["image"]; truth = z["truth"].astype(bool)
        if x0.shape[0] != len(TS):
            raise ValueError("x0 / timestep mismatch")
        pe, _ = pipe.encode_prompt(prompt=row["prompt"], device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False)
        ids77, H = ctx(row["attribution_prompt"])
        target = list(find_token_subsequence(ids77[0].tolist(), tok(row["classname"], add_special_tokens=False)["input_ids"]))
        S = {"call": 0, "paired": {}, "sa": {16: [], 32: [], 64: []}, "vn": [], "q": {m: [] for m in a3}, "ref": None}
        pipe.unet.set_attn_processor({n: (Cap(n.removesuffix(".processor"), roles[n.removesuffix(".processor")], S)
                                          if n.removesuffix(".processor") in roles else getattr(p, "original", p)) for n, p in procs.items()})
        W = []
        try:
            with torch.no_grad():
                for c, t in enumerate(TS):
                    S["call"] = c; S["paired"].clear(); S["ref"] = None
                    pipe.unet(x0[c:c + 1], torch.tensor(t, device=dev), encoder_hidden_states=pe, return_dict=False)
                    ref = S["ref"]; pseudo = torch.softmax(ref @ ref.T / math.sqrt(ref.shape[-1]), -1)
                    sims = torch.stack([affinity_similarity(*S["paired"][m], pseudo) for m in a3]).clamp_min(0)
                    w = sims / sims.sum() if float(sims.sum()) > 1e-20 else torch.full_like(sims, 1 / len(a3))
                    W.append(w); del pseudo
        finally:
            pipe.unet.set_attn_processor(dict(procs))

        def seed_from(Hc):
            acc = torch.zeros(64, 64, device=dev)
            with torch.no_grad():
                for i, m in enumerate(a3):
                    mod = pipe.unet.get_submodule(m); K = mod.head_to_batch_dim(mod.to_k(Hc[None]).float())
                    for c, q in enumerate(S["q"][m]):
                        P = torch.softmax(q.float() @ K.transpose(1, 2) * mod.scale, -1)[..., target].mean(-1).mean(0)
                        s = int(P.numel() ** 0.5)
                        acc += F.interpolate(P.reshape(1, 1, s, s), size=(64, 64), mode="bilinear", align_corners=False)[0, 0] * W[c][i] / 31
            return (acc - acc.min()) / (acc.max() - acc.min()).clamp_min(1e-12)

        def purified(Nb, alpha):
            H2 = H.clone(); e = H2[target]; e2 = e - alpha * (e @ Nb) @ Nb.T
            H2[target] = e2 * (e.norm(dim=-1, keepdim=True) / e2.norm(dim=-1, keepdim=True).clamp_min(1e-12)); return H2
        seeds = {"live": torch.from_numpy(z["seed"]).float().to(dev), "base": seed_from(H), "pur1": seed_from(purified(N8, 1.0)),
                 "pur0.5": seed_from(purified(N8, 0.5)), "rand": seed_from(purified(NR, 1.0)), "meandir": seed_from(purified(N1, 1.0))}
        acc = {k: torch.zeros(4096, device=dev) for k in seeds}; Aavg = torch.zeros(4096, 4096, device=dev)
        with torch.no_grad():
            for call in range(11):
                A = {s: _attention(S["sa"][s][call], s) for s in (16, 32, 64)}
                Aavg += A[64].mean(0) / 11
                for k, sd in seeds.items():
                    c16 = _resize(sd.reshape(-1), 64, 16).reshape(-1)
                    raw16 = _apply(A[16], c16).mean(0); raw32 = _apply(A[32], _resize(raw16, 16, 32).reshape(-1)).mean(0)
                    z32, _ = cap(raw16, raw32, 16)
                    acc[k] += _apply(A[64], _resize(z32, 32, 64).reshape(-1)).mean(0) / 11
                del A
        res = {"cls": str(z["cls"]),
               "corr_seed": float(np.corrcoef(seeds["base"].cpu().numpy().ravel(), z["seed"].ravel())[0, 1]),
               "corr_live_v3": float(np.corrcoef(acc["live"].cpu().numpy(), z["v3"].ravel())[0, 1]),
               "corr_base_v3": float(np.corrcoef(acc["base"].cpu().numpy(), z["v3"].ravel())[0, 1])}
        for k in ARMS:
            mk = oc.crf(img, st.up(st.mm(acc[k].reshape(64, 64).cpu().numpy()), truth.shape) >= 0.55)
            res[f"cnt/{k}"] = np.array([(mk & truth).sum(), (mk & ~truth).sum(), (~mk & truth).sum()], np.int64)
            res[f"map/{k}"] = acc[k].reshape(64, 64).cpu().numpy().astype(np.float16)
        # ---- (V) edges
        gt = sbd.pool(truth.astype(float), 64).ravel(); obj = gt >= 0.75; bg = gt <= 0.25
        Ad = Aavg.clone(); Ad.fill_diagonal_(-1); top = Ad.topk(KTOP, dim=1).indices.cpu().numpy()
        I = np.repeat(np.arange(4096), KTOP); J = top.ravel()
        keep = (obj[I] | obj[J]) & ((obj[I] | bg[I]) & (obj[J] | bg[J])) & ~(bg[I] & bg[J])
        I, J = I[keep], J[keep]
        rng = np.random.default_rng(zlib.crc32(iid.encode()))
        if len(I) > EMAX:
            s = rng.choice(len(I), EMAX, replace=False); I, J = I[s], J[s]
        y = (obj[I] != obj[J]).astype(np.int8)
        Vn = torch.stack(S["vn"]); ti, tj = torch.from_numpy(I).to(dev), torch.from_numpy(J).to(dev)
        valcos = (Vn[:, ti] * Vn[:, tj]).sum(-1).mean(0).cpu().numpy()
        Am = Aavg.cpu().numpy(); aff = np.maximum(Am[I, J], Am[J, I])
        L = sbd.lab(img); Lc = sbd.pool(L, 64).reshape(-1, 3)
        Lg = gaussian_filter(L, (1, 1, 0)); G = np.sqrt(sum(sobel(Lg[..., kk], 0) ** 2 + sobel(Lg[..., kk], 1) ** 2 for kk in range(3)))
        ci = np.c_[I // 64, I % 64].astype(float); cj = np.c_[J // 64, J % 64].astype(float)
        contour = sbd.contour_max(G, ci, cj, 8)
        f = z["feat/up2/s15"].astype(np.float32).reshape(640, -1).T; f /= np.linalg.norm(f, axis=1, keepdims=True).clip(1e-12)
        conf = st.mm(z["v3"]).ravel()
        res.update({"e/y": y, "e/valcos": valcos.astype(np.float32), "e/aff": aff.astype(np.float32),
                    "e/dlab": np.linalg.norm(Lc[I] - Lc[J], axis=1).astype(np.float32), "e/contour": np.asarray(contour, np.float32),
                    "e/dist": np.linalg.norm(ci - cj, axis=1).astype(np.float32), "e/featcos": (f[I] * f[J]).sum(1).astype(np.float32),
                    "e/conf_min": np.minimum(conf[I], conf[J]).astype(np.float32), "e/conf_max": np.maximum(conf[I], conf[J]).astype(np.float32)})
        res["sec"] = time.perf_counter() - t0
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)
        iou = {k: res[f"cnt/{k}"][0] / max(res[f"cnt/{k}"].sum(), 1) for k in ARMS}
        print(f"[vp:{a.dataset}] {iid} r(seed) {res['corr_seed']:.3f} r(live v3) {res['corr_live_v3']:.3f} r(base v3) {res['corr_base_v3']:.3f} "
              + " ".join(f"{k} {iou[k]:.3f}" for k in ARMS) + f" edges {len(y)} ({y.mean():.2f} cross) {res['sec']:.1f}s", flush=True)
        del S, Aavg, Vn; torch.cuda.empty_cache()


def score(a):
    import sa_boundary_diag as sbd
    sbd.NBOOT = 2000
    W = work(a.tag); rng = np.random.default_rng(0); res = {}
    for ds in ("voc", "coco"):
        fs = sorted(f for f in (W / ds).glob("*.npz") if not f.stem.endswith(".tmp"))
        if not fs:
            continue
        Z = [np.load(f) for f in fs]; cls = np.array([str(z["cls"]) for z in Z]); n = len(Z)
        C = {k: np.stack([z[f"cnt/{k}"] for z in Z]) for k in ARMS}

        def miou(cnt, idx):
            return float(np.mean([(lambda s: s[0] / max(s.sum(), 1))(cnt[idx][cls[idx] == c].sum(0)) for c in np.unique(cls[idx])]))
        B = [rng.integers(0, n, n) for _ in range(2000)]; al = np.arange(n)

        def delta(k1, k0):
            d = miou(C[k1], al) - miou(C[k0], al); bs = [miou(C[k1], b) - miou(C[k0], b) for b in B]
            return dict(delta=d, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
        P = {k: dict(miou=miou(C[k], al), **(delta(k, "base") if k != "base" else {})) for k in ARMS}
        P["pur1_vs_rand"] = delta("pur1", "rand")
        y = np.concatenate([z["e/y"] for z in Z]).astype(int); grp = np.concatenate([np.full(len(z["e/y"]), f.stem) for z, f in zip(Z, fs)])
        X = {k: np.concatenate([z[f"e/{k}"] for z in Z]).astype(np.float64) for k in BASEF + ["valcos"]}
        S0 = np.stack([X[k] for k in BASEF], 1); g = sbd.compare(S0, np.c_[S0, X["valcos"]], y, grp)
        single = {k: float(sbd.auc(y, X[k])) for k in X}
        res[ds] = dict(n=n, P=P, V=g, single=single, edges=int(len(y)), cross=float(y.mean()),
                       corr_seed=float(np.mean([z["corr_seed"] for z in Z])), corr_live_v3=float(np.mean([z["corr_live_v3"] for z in Z])),
                       corr_base_v3=float(np.mean([z["corr_base_v3"] for z in Z])), sec=float(np.mean([z["sec"] for z in Z])))
    c, v = res.get("coco"), res.get("voc")
    vpass = bool(c and c["V"]["delta"] >= 0.05 and c["V"]["lo"] > 0)
    ppass = bool(c and v and c["P"]["pur1"]["delta"] >= 0.01 and c["P"]["pur1"]["lo"] > 0 and v["P"]["pur1"]["delta"] >= -0.005
                 and c["P"]["pur1_vs_rand"]["lo"] > 0)
    res.update(V_pass=vpass, P_pass=ppass)
    f4 = lambda x: f"{x:+.4f}"
    L = ["# (V) SA value-space compatibility and (P) semantic subspace purification (diagnostics; `PREREG.md`)\n",
         "Replayed stored trajectories of the 150 + 150 screening images with the exact A3 seed weighting; all arms from one replay. "
         "Development images (disclosed). GT scores only.\n",
         f"## Verdicts (decisive COCO-Cap): V value compatibility **{'PASS' if vpass else 'CLOSE'}**, P purification **{'PASS' if ppass else 'CLOSE'}**\n",
         "### (V) Edges: crossing (object↔background) vs within-object, among each cell's 16 strongest fine-SA links\n",
         "| set | images | edges (crossing) | baseline AUROC | + value cosine Δ (95% CI) |", "|---|---:|---|---:|---|"]
    for ds in ("voc", "coco"):
        if ds in res:
            r = res[ds]; G = r["V"]
            L.append(f"| {NAME[ds]} | {r['n']} | {r['edges']:,} ({r['cross']:.1%}) | {G['rgb']:.3f} | {f4(G['delta'])} [{f4(G['lo'])}, {f4(G['hi'])}] |")
    L += ["\nSingle-feature AUROC for crossing (higher = larger on crossing edges):\n", "| feature | VOC-Sim | COCO-Cap |", "|---|---:|---:|"]
    for k in BASEF + ["valcos"]:
        L.append(f"| {k} | " + " | ".join(f"{res[d]['single'][k]:.3f}" for d in ("voc", "coco") if d in res) + " |")
    L += ["\n### (P) Purified class context (τ 0.55 + R2, class-macro mIoU; Δ vs exact-A3 replay `base`, image bootstrap)\n",
          "| arm | role | " + " | ".join(f"{NAME[d]} mIoU | Δ (95% CI)" for d in ("voc", "coco") if d in res) + " |",
          "|---|---|" + "---:|---|" * len([d for d in ("voc", "coco") if d in res])]
    role = {"live": "fidelity (stored live seed)", "base": "exact A3 replay", "pur1": "**primary** (α 1)", "pur0.5": "secondary (α 0.5)",
            "rand": "control: random 8-d subspace", "meandir": "control: mean nuisance direction only"}
    for k in ARMS:
        cells = []
        for d in ("voc", "coco"):
            if d in res:
                x = res[d]["P"][k]; cells.append(f"{x['miou']:.4f} | " + ("—" if k == "base" else f"{f4(x['delta'])} [{f4(x['lo'])}, {f4(x['hi'])}]"))
        L.append(f"| {k} | {role[k]} | " + " | ".join(cells) + " |")
    L.append("| pur1 − rand | gate condition | " + " | ".join(
        f"| {f4(res[d]['P']['pur1_vs_rand']['delta'])} [{f4(res[d]['P']['pur1_vs_rand']['lo'])}, {f4(res[d]['P']['pur1_vs_rand']['hi'])}]" for d in ("voc", "coco") if d in res) + " |")
    for d in ("voc", "coco"):
        if d in res:
            r = res[d]
            L.append(f"\n{NAME[d]} replay fidelity: r(replay A3 seed, live seed) = {r['corr_seed']:.3f}; r(V3 from live seed, stored V3) = {r['corr_live_v3']:.3f}; "
                     f"r(V3 from replay seed, stored V3) = {r['corr_base_v3']:.3f}; {r['sec']:.0f} s/image.")
    out = RUN / (f"result_{a.tag}" if a.tag else "result")
    out.with_suffix(".md").write_text("\n".join(L) + "\n"); out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--dataset", choices=("voc", "coco"), required=True); p.add_argument("--limit", type=int); p.add_argument("--tag", default="")
    p = sub.add_parser("score"); p.add_argument("--tag", default="")
    a = ap.parse_args(); {"run": run, "score": score}[a.cmd](a)
