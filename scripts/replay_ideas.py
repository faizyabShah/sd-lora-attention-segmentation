#!/usr/bin/env python3
"""Two training-free ideas, screened on replayed SD1.5 trajectories (runs/12_replay_ideas/PREREG.md). BRACE frozen.

SCW  seed-consistency weighting of self-attention heads: at every V3 propagation level, head h gets weight
     softmax_h(z(corr(A_h x, seed_level))) instead of the uniform 1/8 (the SA analogue of A3's CA layer weighting).
ROC  one closed-form label-free step in text-context space: the class token's context vector e moves along the
     gradient of [mean logit on BRACE's own core F - mean logit on clear background B] (summed over the A3 CA
     modules, heads and calls): e' = e + beta * ||e|| * g/||g||. No labels, no iterations, no UNet backprop.
ENT  reference only: iSeg-style entropy weighting of SA heads (prior art), softmax_h(-z(mean row entropy)).

Replay: the stored per-call UNet inputs x0 (31 calls) of the rich capture, conditional prompt only, are re-run
once; every arm (incl. the baseline) is computed from that single replay, so all comparisons are within capture.

python scripts/replay_ideas.py run --dataset voc|coco [--limit k] [--tag t]   GPU -> CACHE/replay_ideas[_t]/<ds>/<id>.npz
python scripts/replay_ideas.py score [--tag t]                                CPU -> runs/12_replay_ideas/result.md
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brace_paths import CACHE  # noqa: E402

RUN = ROOT / "runs/12_replay_ideas"; DUMP = CACHE / "ana_dump"
A3_IDX = (0, 1, 4, 5, 6, 7, 8, 9, 13, 14, 15)
SA_CALLS = range(10, 21)
BETAS = (0.05, 0.1, 0.2); BETA0 = 0.1
ARMS = ("live", "base", "ent", "scw", "roc0.05", "roc0.1", "roc0.2", "scw+roc0.1")
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}


def work(tag):
    return CACHE / ("replay_ideas" + (f"_{tag}" if tag else ""))


class Cap:
    def __init__(self, kind, store, name):
        self.kind, self.store, self.name = kind, store, name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *a, **k):
        from diffusers.models.attention_processor import AttnProcessor
        if self.kind == "sa" and self.store["call"] in SA_CALLS:
            q = attn.head_to_batch_dim(attn.to_q(hidden_states)); kk = attn.head_to_batch_dim(attn.to_k(hidden_states))
            self.store["sa"][self.name].append((q.detach(), kk.detach()))
        elif self.kind == "ca":
            self.store["q"][self.name].append(attn.head_to_batch_dim(attn.to_q(hidden_states)).detach().half())
        return AttnProcessor()(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


def zsoft(s):
    import torch
    return torch.softmax((s - s.mean()) / s.std().clamp_min(1e-8), 0)


def corr_rows(P, x):
    P = P - P.mean(1, keepdim=True); x = x - x.mean()
    return (P @ x) / (P.norm(dim=1) * x.norm()).clamp_min(1e-12)


def run(a):
    import torch, torch.nn.functional as F
    from scripts import sg27_runtime
    from src.segmentation.sg25_sa import _apply, _attention, _resize
    from src.segmentation.sg_combined_v2 import cap
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
    a3 = [attn2[i] for i in A3_IDX]; sa_names = {mods[s]: s for s in (16, 32, 64)}
    out = work(a.tag) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    tok = pipe.tokenizer
    for iid in ids:
        o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); z = np.load(DUMP / a.dataset / f"{iid}.npz", allow_pickle=True); row = dict(rows[iid])
        x0 = torch.from_numpy(z["x0"].astype(np.float32)).to(dev); img = z["image"]; truth = z["truth"].astype(bool)
        if x0.shape[0] != len(TS):
            raise ValueError(f"x0 calls {x0.shape[0]} vs timesteps {len(TS)}")
        pe, _ = pipe.encode_prompt(prompt=row["prompt"], device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False)
        ap = row["attribution_prompt"]
        ids77 = tok(ap, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(dev)
        target = find_token_subsequence(ids77[0].tolist(), tok(row["classname"], add_special_tokens=False)["input_ids"])
        with torch.no_grad():
            H = pipe.text_encoder(ids77)[0][0].float()                                     # [77, 768] attribution context
        store = {"call": 0, "sa": {n: [] for n in sa_names}, "q": {n: [] for n in a3}}
        new = {}
        for n, p in procs.items():
            m = n.removesuffix(".processor")
            new[n] = Cap("sa", store, m) if m in sa_names else Cap("ca", store, m) if m in a3 else getattr(p, "original", p)
        pipe.unet.set_attn_processor(new)
        try:
            with torch.no_grad():
                for c, t in enumerate(TS):
                    store["call"] = c
                    pipe.unet(x0[c:c + 1], torch.tensor(t, device=dev), encoder_hidden_states=pe, return_dict=False)
        finally:
            pipe.unet.set_attn_processor(dict(procs))
        Wk = {m: pipe.unet.get_submodule(m) for m in a3}

        def seed_from(Hc):
            acc = torch.zeros(64, 64, device=dev)
            with torch.no_grad():
                for m in a3:
                    mod = Wk[m]; K = mod.head_to_batch_dim(mod.to_k(Hc[None]).float())   # [8, 77, d]
                    for q in store["q"][m]:
                        P = torch.softmax(q.float() @ K.transpose(1, 2) * mod.scale, -1)[..., target].mean(-1).mean(0)
                        s = int(P.numel() ** 0.5)
                        acc += F.interpolate(P.reshape(1, 1, s, s), size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
            acc = acc / (len(a3) * len(TS)); return (acc - acc.min()) / (acc.max() - acc.min()).clamp_min(1e-12)

        sa = {s: store["sa"][n] for n, s in sa_names.items()}

        def propagate(seeds, rules):
            """seeds: {arm: 64x64 seed}; rules: {arm: 'uni'|'ent'|'scw'} -> V3-style maps, attention computed once per call."""
            acc = {k: torch.zeros(4096, device=dev) for k in seeds}
            for call in range(11):
                A = {s: _attention(sa[s][call], s) for s in (16, 32, 64)}
                ent = {s: -(A[s] * A[s].clamp_min(1e-20).log()).sum(-1).mean(-1) for s in A}
                for k, sd in seeds.items():
                    c64 = sd.reshape(-1); anchor = {s: _resize(c64, 64, s).reshape(-1) for s in (16, 32, 64)}

                    def comb(P, s):
                        if rules[k] == "uni":
                            return P.mean(0)
                        w = zsoft(-ent[s]) if rules[k] == "ent" else zsoft(corr_rows(P, anchor[s]))
                        return (w[:, None] * P).sum(0)
                    raw16 = comb(_apply(A[16], anchor[16]), 16)
                    raw32 = comb(_apply(A[32], _resize(raw16, 16, 32).reshape(-1)), 32)
                    z32, _ = cap(raw16, raw32, 16)
                    acc[k] += comb(_apply(A[64], _resize(z32, 32, 64).reshape(-1)), 64) / 11
                del A
            return {k: v.reshape(64, 64) for k, v in acc.items()}

        live = torch.from_numpy(z["seed"]).float().to(dev); base = seed_from(H)
        m1 = propagate({"live": live, "base": base, "ent": base, "scw": base}, {"live": "uni", "base": "uni", "ent": "ent", "scw": "scw"})
        # ---- ROC: label-free cores from the baseline arm, one closed-form step
        v = st.mm(m1["base"].cpu().numpy()); M = oc.crf(img, st.up(v, truth.shape) >= 0.55)
        import sa_boundary_diag as sbd
        Mc = sbd.pool(M.astype(float), 64) >= 0.5
        Fc = v >= 0.9
        if not Fc.any():
            Fc = v >= np.quantile(v, 0.95)
        Bc = (v <= 0.1) & ~Mc
        if Bc.sum() < 5:
            Bc = (v <= np.quantile(v, 0.05)) & ~Mc
        Ft = torch.from_numpy(Fc.astype(np.float32)).to(dev); Bt = torch.from_numpy(Bc.astype(np.float32)).to(dev)
        delta = torch.zeros(768, device=dev, requires_grad=True); J = 0
        for m in a3:
            mod = Wk[m]; Hd = H.clone(); Hd[list(target)] = Hd[list(target)] + delta
            K = mod.head_to_batch_dim(mod.to_k(Hd[None]).float())[:, list(target)]          # [8, n_t, d]
            for q in store["q"][m]:
                s = int(q.shape[1] ** 0.5)
                wf = F.adaptive_avg_pool2d(Ft[None, None], s).reshape(-1); wb = F.adaptive_avg_pool2d(Bt[None, None], s).reshape(-1)
                L = (q.float() @ K.transpose(1, 2) * mod.scale).mean(-1)                       # [8, HW] mean over pieces
                J = J + ((L @ wf) / wf.sum().clamp_min(1e-8) - (L @ wb) / wb.sum().clamp_min(1e-8)).mean()
        J.backward(); g = delta.grad.detach(); g = g / g.norm().clamp_min(1e-12)
        en = H[list(target)].norm(dim=-1).mean()
        seeds2, rules2 = {}, {}
        for b in BETAS:
            H2 = H.clone(); H2[list(target)] = H2[list(target)] + b * en * g
            seeds2[f"roc{b}"] = seed_from(H2); rules2[f"roc{b}"] = "uni"
        seeds2["scw+roc0.1"] = seeds2["roc0.1"]; rules2["scw+roc0.1"] = "scw"
        maps = {**m1, **propagate(seeds2, rules2)}
        res = {"cls": str(z["cls"]), "sec": 0.0, "corr_live_v3": float(np.corrcoef(maps["live"].cpu().numpy().ravel(), z["v3"].ravel())[0, 1]),
               "corr_seed": float(np.corrcoef(base.cpu().numpy().ravel(), z["seed"].ravel())[0, 1]),
               "F": int(Fc.sum()), "B": int(Bc.sum()), "J": float(J.detach())}
        tb = truth
        for k in ARMS:
            mk = oc.crf(img, st.up(st.mm(maps[k].cpu().numpy()), truth.shape) >= 0.55)
            res[f"cnt/{k}"] = np.array([(mk & tb).sum(), (mk & ~tb).sum(), (~mk & tb).sum()], np.int64)
            res[f"map/{k}"] = maps[k].cpu().numpy().astype(np.float16)
        res["sec"] = time.perf_counter() - t0
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)
        iou = {k: res[f"cnt/{k}"][0] / max(res[f"cnt/{k}"].sum(), 1) for k in ARMS}
        print(f"[ri:{a.dataset}] {iid} r(live,v3) {res['corr_live_v3']:.3f} r(seed) {res['corr_seed']:.3f} "
              + " ".join(f"{k} {iou[k]:.3f}" for k in ARMS) + f" {res['sec']:.1f}s", flush=True)
        del store, sa; torch.cuda.empty_cache()


def score(a):
    W = work(a.tag); res = {}; rng = np.random.default_rng(0)
    L = ["# Training-free SA head weighting (SCW) and closed-form context refinement (ROC): screen (`PREREG.md`)\n",
         "Replayed stored trajectories of the 150 + 150 screening images (development data; disclosed). All arms from the same replay. "
         "Fixed τ 0.55 + DenseCRF R2. Class-macro mIoU; image-bootstrap 95% CI of Δ vs replay baseline (2000).\n"]
    for ds in ("voc", "coco"):
        fs = sorted(f for f in (W / ds).glob("*.npz") if not f.stem.endswith(".tmp"))
        if not fs:
            continue
        Z = [np.load(f) for f in fs]; cls = np.array([str(z["cls"]) for z in Z])
        C = {k: np.stack([z[f"cnt/{k}"] for z in Z]) for k in ARMS}

        def miou(cnt, idx):
            out = []
            for c in np.unique(cls[idx]):
                s = cnt[idx][cls[idx] == c].sum(0); out.append(s[0] / max(s.sum(), 1))
            return float(np.mean(out))
        allidx = np.arange(len(Z)); B = [rng.integers(0, len(Z), len(Z)) for _ in range(2000)]
        r = {"n": len(Z), "corr_live_v3": float(np.mean([z["corr_live_v3"] for z in Z])), "corr_seed": float(np.mean([z["corr_seed"] for z in Z])),
             "sec": float(np.mean([z["sec"] for z in Z])), "arms": {}}
        for k in ARMS:
            m = miou(C[k], allidx); d = m - miou(C["base"], allidx)
            bs = [miou(C[k], b) - miou(C["base"], b) for b in B] if k != "base" else [0.0]
            r["arms"][k] = dict(miou=m, delta=d, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
        res[ds] = r
    gate = {}
    for k in ("scw", "roc0.1"):
        if "coco" in res and "voc" in res:
            c, vv = res["coco"]["arms"][k], res["voc"]["arms"][k]
            gate[k] = bool(c["delta"] >= 0.01 and c["lo"] > 0 and vv["delta"] >= -0.005)
    res["gate"] = gate
    L += [f"## Screen verdicts (primary): SCW **{'PROMISING' if gate.get('scw') else 'STOP'}**, ROC β 0.1 **{'PROMISING' if gate.get('roc0.1') else 'STOP'}**\n",
          "Bar: COCO-Cap Δ ≥ +0.010 with CI lower > 0, and VOC-Sim Δ ≥ −0.005 (vs replay baseline).\n",
          "| arm | role | " + " | ".join(f"{NAME[d]} mIoU | Δ (95% CI)" for d in res if d in NAME) + " |",
          "|---|---|" + "---:|---|" * len([d for d in res if d in NAME])]
    role = {"live": "fidelity: stored live A3 seed + replayed SA", "base": "replay baseline (A3 modules, uniform layer weights)",
            "ent": "reference: iSeg-style entropy weighting", "scw": "**primary A**", "roc0.1": "**primary B**",
            "roc0.05": "secondary", "roc0.2": "secondary", "scw+roc0.1": "secondary combo"}
    for k in ARMS:
        cells = []
        for d in ("voc", "coco"):
            if d in res:
                x = res[d]["arms"][k]
                cells.append(f"{x['miou']:.4f} | " + ("—" if k == "base" else f"{x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"))
        L.append(f"| {k} | {role[k]} | " + " | ".join(cells) + " |")
    for d in ("voc", "coco"):
        if d in res:
            L.append(f"\n{NAME[d]}: {res[d]['n']} images; replay fidelity r(live-seed V3 replay, stored live V3) = {res[d]['corr_live_v3']:.3f}; "
                     f"r(replay seed, live A3 seed) = {res[d]['corr_seed']:.3f}; {res[d]['sec']:.0f} s/image.")
    L.append("\nScreen only (development images). A promising arm licenses a pre-registered confirmation on fresh images, not a BRACE change.")
    out = RUN / (f"result_{a.tag}" if a.tag else "result")
    out.with_suffix(".md").write_text("\n".join(L) + "\n"); out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--dataset", choices=("voc", "coco"), required=True); p.add_argument("--limit", type=int); p.add_argument("--tag", default="")
    p = sub.add_parser("score"); p.add_argument("--tag", default="")
    a = ap.parse_args(); {"run": run, "score": score}[a.cmd](a)
