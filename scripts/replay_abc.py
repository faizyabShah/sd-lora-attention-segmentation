#!/usr/bin/env python3
"""Evaluation of (A) green-screen tokens, (B) depth-time signature and (C) caption-competitive attribution on
replayed trajectories with the exact A3 seed weighting. Pre-registered in runs/14_abc/PREREG.md. BRACE frozen.

python scripts/replay_abc.py run --dataset voc|coco [--limit k] [--tag t] [--tokens-tag t]   GPU -> CACHE/replay_abc[_t]/<ds>/<id>.npz
python scripts/replay_abc.py score [--tag t] [--workers 16]                                  CPU -> runs/14_abc/result.md
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brace_paths import CACHE  # noqa: E402
import replay_vp as vp  # noqa: E402
import gs_tokens as gs  # noqa: E402

RUN = ROOT / "runs/14_abc"; DUMP = CACHE / "ana_dump"
ARMS = ("live", "base", "clsonly", "tmpl", "incap", "gs", "gs_lin")
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
BASE = ["dLab_F", "dLab_B", "contour", "dist_F", "conf", "depth", "area", "s_rgb", "mean_ca", "mean_v3"]


def work(tag):
    return CACHE / ("replay_abc" + (f"_{tag}" if tag else ""))


def run(a):
    import torch, torch.nn.functional as F
    from scripts import sg27_runtime
    from src.segmentation.sg25_sa import _apply, _attention, _resize
    from src.segmentation.sg_combined_v2 import cap
    from src.segmentation.sg24_ca import affinity_similarity, REF_SUFFIX
    from src.segmentation.attention_aggregation import find_token_subsequence
    import idea1_knockout as k1, sg_class_comp as cc, sg_steps_test as st, ana_ops_common as oc
    from ideas_grouped import screen_ids
    oc.crf_init()
    ids = sorted(screen_ids(a.dataset))[: a.limit or None]
    rows = {r["image_id"]: r for r in cc.gen_rows(a.dataset)}
    rt = sg27_runtime.make_runtime(k1.MODEL); pipe = rt["pipe"]; dev = pipe.device; mods = rt["modules"]
    sched = pipe.scheduler; sched.set_timesteps(30, device=dev); TS = [int(t) for t in sched.timesteps]
    procs = dict(pipe.unet.attn_processors)
    attn2 = sorted(n.removesuffix(".processor") for n in procs if n.endswith("attn2.processor"))
    a3 = [attn2[i] for i in vp.A3_IDX]
    if tuple(a3) != tuple(rt["ca_modules"]):
        raise ValueError("A3 module list differs from the runtime's frozen CA modules")
    roles = {}
    for m in a3:
        roles.setdefault(m, {})["a3"] = 1; roles.setdefault(m[:-5] + "attn1", {})["paired"] = m
    for s in (16, 32, 64):
        roles.setdefault(mods[s], {})["brace"] = s
    roles.setdefault(REF_SUFFIX, {})["ref"] = 1
    tok = pipe.tokenizer
    tf = gs.work(a.tokens_tag) / "tokens.npz"
    T = dict(np.load(tf)) if tf.exists() else {}
    print(f"tokens: {tf} ({len(T) // 2} classes)", flush=True)

    def ctx(text, name):
        """context [77, 768] and the class-piece positions, or None if the class word was truncated away."""
        ids77 = tok(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(dev)
        try:
            pos = list(find_token_subsequence(ids77[0].tolist(), tok(name, add_special_tokens=False)["input_ids"]))
        except ValueError:
            return None
        with torch.no_grad():
            return pipe.text_encoder(ids77)[0][0].float(), pos
    out = work(a.tag) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    for iid in ids:
        o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); z = np.load(DUMP / a.dataset / f"{iid}.npz", allow_pickle=True); row = rows[iid]
        name = row["classname"]; x0 = torch.from_numpy(z["x0"].astype(np.float32)).to(dev); img = z["image"]; truth = z["truth"].astype(bool)
        if x0.shape[0] != len(TS):
            raise ValueError("x0 / timestep mismatch")
        pe, _ = pipe.encode_prompt(prompt=row["prompt"], device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False)
        S = {"call": 0, "paired": {}, "sa": {16: [], 32: [], 64: []}, "vn": [], "q": {m: [] for m in a3}, "ref": None}
        pipe.unet.set_attn_processor({n: (vp.Cap(n.removesuffix(".processor"), roles[n.removesuffix(".processor")], S)
                                          if n.removesuffix(".processor") in roles else getattr(p, "original", p)) for n, p in procs.items()})
        W = []
        try:
            with torch.no_grad():
                for c, t in enumerate(TS):
                    S["call"] = c; S["paired"].clear(); S["ref"] = None
                    pipe.unet(x0[c:c + 1], torch.tensor(t, device=dev), encoder_hidden_states=pe, return_dict=False)
                    ref = S["ref"]; pseudo = torch.softmax(ref @ ref.T / math.sqrt(ref.shape[-1]), -1)
                    sims = torch.stack([affinity_similarity(*S["paired"][m], pseudo) for m in a3]).clamp_min(0)
                    W.append(sims / sims.sum() if float(sims.sum()) > 1e-20 else torch.full_like(sims, 1 / len(a3))); del pseudo
        finally:
            pipe.unet.set_attn_processor(dict(procs))
        S["vn"].clear()

        def seed_from(Hc, target, record=False):
            acc = torch.zeros(64, 64, device=dev); logp = torch.zeros(len(a3), len(TS), 64, 64, device=dev) if record else None; res = []
            with torch.no_grad():
                for i, m in enumerate(a3):
                    mod = pipe.unet.get_submodule(m); K = mod.head_to_batch_dim(mod.to_k(Hc[None]).float())
                    for c, q in enumerate(S["q"][m]):
                        P = torch.softmax(q.float() @ K.transpose(1, 2) * mod.scale, -1)[..., target].mean(-1).mean(0)
                        s = int(P.numel() ** 0.5)
                        up = F.interpolate(P.reshape(1, 1, s, s), size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
                        acc += up * W[c][i] / 31
                        if record:
                            logp[i, c] = (up + 1e-12).log()
                    res.append(s)
            seed = (acc - acc.min()) / (acc.max() - acc.min()).clamp_min(1e-12)
            return (seed, logp, res) if record else seed
        C = {"base": ctx(row["attribution_prompt"], name), "clsonly": ctx(name, name), "tmpl": ctx(f"A photograph of a {name}", name)}
        mode = 0; C["incap"] = ctx(row["prompt"], name)
        if C["incap"] is None:
            mode = 1; C["incap"] = ctx(f"{row['prompt']}, {name}", name)
        if C["incap"] is None:
            mode = 2; C["incap"] = C["clsonly"]
        has_tok = 0
        for k in ("gs", "gs_lin"):
            H, pos = C["clsonly"]; d = T.get(f"{k}/{gs.canon(name)}")
            if d is not None:
                H = H.clone(); H[pos] = H[pos] + torch.from_numpy(d).to(dev); has_tok = 1
            C[k] = (H, pos)
        base_seed, logp, mres = seed_from(*C["base"], record=True)
        seeds = {"live": torch.from_numpy(z["seed"]).float().to(dev), "base": base_seed}
        seeds.update({k: seed_from(*C[k]) for k in ARMS if k not in seeds})
        acc = {k: torch.zeros(4096, device=dev) for k in seeds}
        with torch.no_grad():
            for call in range(11):
                A = {s: _attention(S["sa"][s][call], s) for s in (16, 32, 64)}
                for k, sd in seeds.items():
                    c16 = _resize(sd.reshape(-1), 64, 16).reshape(-1)
                    raw16 = _apply(A[16], c16).mean(0); raw32 = _apply(A[32], _resize(raw16, 16, 32).reshape(-1)).mean(0)
                    z32, _ = cap(raw16, raw32, 16)
                    acc[k] += _apply(A[64], _resize(z32, 32, 64).reshape(-1)).mean(0) / 11
                del A
        res = {"cls": str(z["cls"]), "incap_mode": mode, "has_tok": has_tok, "mres": np.array(mres),
               "logp": logp.reshape(len(a3), len(TS), 4096).cpu().numpy().astype(np.float16),
               "corr_seed": float(np.corrcoef(base_seed.cpu().numpy().ravel(), z["seed"].ravel())[0, 1]),
               "corr_base_v3": float(np.corrcoef(acc["base"].cpu().numpy(), z["v3"].ravel())[0, 1])}
        for k in ARMS:
            mk = oc.crf(img, st.up(st.mm(acc[k].reshape(64, 64).cpu().numpy()), truth.shape) >= 0.55)
            res[f"cnt/{k}"] = np.array([(mk & truth).sum(), (mk & ~truth).sum(), (~mk & truth).sum()], np.int64)
            res[f"map/{k}"] = acc[k].reshape(64, 64).cpu().numpy().astype(np.float16)
        res["sec"] = time.perf_counter() - t0
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)
        iou = {k: res[f"cnt/{k}"][0] / max(res[f"cnt/{k}"].sum(), 1) for k in ARMS}
        print(f"[abc:{a.dataset}] {iid} r(seed) {res['corr_seed']:.3f} incap {mode} tok {has_tok} "
              + " ".join(f"{k} {iou[k]:.3f}" for k in ARMS) + f" {res['sec']:.1f}s", flush=True)
        del S, logp; torch.cuda.empty_cache()


def one_b(job):
    """(B) per-image features on the cells Q."""
    ds, iid, W = job
    import equiv_noise as en
    r = en.one((ds, iid, str(en.work(""))))
    if not r["ok"] or not len(r["y"]):
        return {"id": iid, "ok": False}
    c = np.load(Path(W) / ds / f"{iid}.npz"); L = c["logp"].astype(np.float32); mres = c["mres"]      # [11, 31, 4096]
    D = L - L.mean(2, keepdims=True); D = D - D.mean((0, 1), keepdims=True)
    X = D.reshape(-1, 4096).T                                                                       # [4096, 341]
    Fm, Bm = r["F"].ravel(), r["B"].ravel(); Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    uf = X[Fm].mean(0); ub = X[Bm].mean(0)
    sig = Xn @ (uf / max(np.linalg.norm(uf), 1e-12)) - Xn @ (ub / max(np.linalg.norm(ub), 1e-12))
    fine = mres == 64; coarse = mres <= 16
    fl = D[fine][:, 16:].mean((0, 1)) - D[coarse][:, :16].mean((0, 1))
    Q = r["Q"]
    return {"id": iid, "ok": True, "y": r["y"], "X": {**{k: r["X"][k] for k in BASE}, "sig": sig[Q].astype(np.float64), "fl": fl[Q].astype(np.float64)}}


def score(a):
    import sa_boundary_diag as sbd
    sbd.NBOOT = 2000
    W = work(a.tag); rng = np.random.default_rng(0); res = {}
    for ds in ("voc", "coco"):
        fs = sorted(f for f in (W / ds).glob("*.npz") if not f.stem.endswith(".tmp"))
        if not fs:
            continue
        Z = [np.load(f) for f in fs]; cls = np.array([str(z["cls"]) for z in Z]); n = len(Z)
        C = {k: np.stack([z[f"cnt/{k}"] for z in Z]) for k in ARMS}; mode = np.array([int(z["incap_mode"]) for z in Z])

        def miou(cnt, idx):
            return float(np.mean([(lambda s: s[0] / max(s.sum(), 1))(cnt[idx][cls[idx] == c].sum(0)) for c in np.unique(cls[idx])]))
        Bt = [rng.integers(0, n, n) for _ in range(2000)]; al = np.arange(n)

        def delta(k1, k0, idx=None):
            idx = al if idx is None else idx
            d = miou(C[k1], idx) - miou(C[k0], idx); bs = [miou(C[k1], idx[b % len(idx)][: len(idx)]) - miou(C[k0], idx[b % len(idx)][: len(idx)]) for b in Bt]
            return dict(delta=d, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
        P = {k: dict(miou=miou(C[k], al), **(delta(k, "base") if k != "base" else {})) for k in ARMS}
        P["gs-clsonly"] = delta("gs", "clsonly"); P["incap-tmpl"] = delta("incap", "tmpl")
        split = {}
        for mname, mv in (("word in caption", 0), ("word appended", 1)):
            idx = al[mode == mv]
            if len(idx) >= 10:
                split[mname] = dict(n=int(len(idx)), base=miou(C["base"], idx), incap=miou(C["incap"], idx), **delta("incap", "base", idx))
        with Pool(a.workers) as p:
            R = p.map(one_b, [(ds, f.stem, str(W)) for f in fs], chunksize=1)
        ok = [r for r in R if r["ok"]]
        y = np.concatenate([r["y"] for r in ok]); grp = np.concatenate([np.full(len(r["y"]), r["id"]) for r in ok])
        X = {k: np.concatenate([r["X"][k] for r in ok]) for k in BASE + ["sig", "fl"]}
        fin = np.isfinite(np.stack(list(X.values()), 1)).all(1); y, grp = y[fin], grp[fin]; X = {k: v[fin] for k, v in X.items()}
        S0 = np.stack([X[k] for k in BASE], 1); g = sbd.compare(S0, np.c_[S0, X["sig"], X["fl"]], y, grp)
        res[ds] = dict(n=n, P=P, split=split, B=g, single={k: float(sbd.auc(y, X[k])) for k in ("conf", "sig", "fl")},
                       cells=int(len(y)), bleed=float(y.mean()), has_tok=int(sum(int(z["has_tok"]) for z in Z)),
                       fallback=int((mode == 2).sum()), corr_seed=float(np.mean([z["corr_seed"] for z in Z])),
                       corr_base_v3=float(np.mean([z["corr_base_v3"] for z in Z])), sec=float(np.mean([z["sec"] for z in Z])))
    c, v = res.get("coco"), res.get("voc")
    ga = bool(c and v and c["P"]["gs"]["delta"] >= 0.01 and c["P"]["gs"]["lo"] > 0 and v["P"]["gs"]["delta"] >= -0.005)
    gb = bool(c and c["B"]["delta"] >= 0.05 and c["B"]["lo"] > 0)
    gc = bool(c and c["P"]["incap"]["delta"] >= 0.01 and c["P"]["incap"]["lo"] > 0)
    res.update(A_pass=ga, B_pass=gb, C_pass=gc)
    tstats = json.loads((gs.work(a.tokens_tag) / "stats.json").read_text()) if (gs.work(a.tokens_tag) / "stats.json").exists() else {}
    f4 = lambda x: f"{x:+.4f}"; ci = lambda x: f"{f4(x['delta'])} [{f4(x['lo'])}, {f4(x['hi'])}]"; DS = [d for d in ("voc", "coco") if d in res]
    L = ["# (A) green-screen tokens, (B) depth–time signature, (C) caption-competitive attribution (`PREREG.md`)\n",
         "Replayed stored trajectories of the 150 + 150 screening images, exact A3 seed weighting, all arms from one replay. "
         "Development images (disclosed). τ 0.55 + DenseCRF R2, class-macro mIoU, image-bootstrap 95% CI.\n",
         f"## Verdicts: A tokens **{'PROMISING' if ga else 'STOP'}**, B signature **{'PASS' if gb else 'CLOSE'}**, C caption attribution **{'PROMISING' if gc else 'STOP'}**\n",
         "| arm | role | " + " | ".join(f"{NAME[d]} mIoU | Δ vs base (95% CI)" for d in DS) + " |", "|---|---|" + "---:|---|" * len(DS)]
    role = {"live": "fidelity (stored live seed)", "base": "frozen attribution, exact A3 replay", "clsonly": "class name alone",
            "tmpl": "C control: \"A photograph of a {class}\"", "incap": "**C primary**: class word inside the caption",
            "gs": "**A primary**: green-screen token", "gs_lin": "A secondary: closed-form step"}
    for k in ARMS:
        L.append(f"| {k} | {role[k]} | " + " | ".join(f"{res[d]['P'][k]['miou']:.4f} | " + ("—" if k == "base" else ci(res[d]["P"][k])) for d in DS) + " |")
    L.append("| gs − clsonly | token effect alone | " + " | ".join(f"| {ci(res[d]['P']['gs-clsonly'])}" for d in DS) + " |")
    L.append("| incap − tmpl | competition beyond a sentence | " + " | ".join(f"| {ci(res[d]['P']['incap-tmpl'])}" for d in DS) + " |")
    if c and c["split"]:
        L += ["\n### (C) COCO-Cap split\n", "| subset | images | base | incap | Δ (95% CI) |", "|---|---:|---:|---:|---|"]
        L += [f"| {k} | {s['n']} | {s['base']:.4f} | {s['incap']:.4f} | {ci(s)} |" for k, s in c["split"].items()]
        L.append(f"\nFallbacks to the class name alone (class word truncated): {c['fallback']}.")
    L += ["\n### (B) Bleed vs correct on Q cells: baseline of run 09 vs + `sig` + `fl`\n",
          "| set | cells (bleed) | baseline AUROC | Δ (95% CI) | single: conf / sig / fl |", "|---|---|---:|---|---|"]
    for d in DS:
        r = res[d]; g = r["B"]
        L.append(f"| {NAME[d]} | {r['cells']:,} ({r['bleed']:.1%}) | {g['rgb']:.3f} | {f4(g['delta'])} [{f4(g['lo'])}, {f4(g['hi'])}] | "
                 f"{r['single']['conf']:.3f} / {r['single']['sig']:.3f} / {r['single']['fl']:.3f} |")
    if tstats:
        L += ["\n### (A) Exemplars and tokens\n", "| class | generated | kept | token | loss first → last |", "|---|---:|---:|---|---|"]
        for k, s in tstats.items():
            L.append(f"| {k} | {s['generated']} | {s['kept']} | {s.get('token', '?')} | " + (f"{s['loss_first']:.3f} → {s['loss_last']:.3f}" if "loss_first" in s else "—") + " |")
    for d in DS:
        r = res[d]
        L.append(f"\n{NAME[d]}: {r['n']} images, {r['has_tok']} with a learned token; r(replay seed, live seed) = {r['corr_seed']:.3f}; "
                 f"r(V3 from replay seed, stored V3) = {r['corr_base_v3']:.3f}; {r['sec']:.0f} s/image.")
    L.append("\nScreens and diagnostics only. A promising arm licenses a pre-registered confirmation on fresh images with a live capture, never a BRACE change.")
    out = RUN / (f"result_{a.tag}" if a.tag else "result")
    out.with_suffix(".md").write_text("\n".join(L) + "\n"); out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--dataset", choices=("voc", "coco"), required=True); p.add_argument("--limit", type=int)
    p.add_argument("--tag", default=""); p.add_argument("--tokens-tag", default="")
    p = sub.add_parser("score"); p.add_argument("--tag", default=""); p.add_argument("--tokens-tag", default=""); p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(); {"run": run, "score": score}[a.cmd](a)
