#!/usr/bin/env python3
"""(A) Green-screen tokens, stage 1 (runs/14_abc/PREREG.md): the generator makes its own annotated exemplars
("a photo of a {class}, isolated on a plain white background"), the mask comes from colour keying alone, and one
context offset per class is optimised on the A3 cross-attention queries. No annotation, no CLIP. BRACE frozen.

python scripts/gs_tokens.py [--classes k] [--tag t]   GPU -> CACHE/gs_tokens[_t]/{tokens.npz, stats.json, exemplars/}
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from brace_paths import CACHE  # noqa: E402

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog",
           "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv"]
A3_IDX = (0, 1, 4, 5, 6, 7, 8, 9, 13, 14, 15)
CALLS = (8, 15, 22); NGEN, NKEEP, NMIN = 40, 10, 3
STEPS, LR, LR_AC, PEN, BETA = 300, 0.005, 0.05, 0.1, 0.1


def canon(name):
    return name.replace("-", " ")


def work(tag):
    return CACHE / ("gs_tokens" + (f"_{tag}" if tag else ""))


def key_mask(img):
    """object mask by keying a plain backdrop of any colour (PREREG amendment 1): the backdrop is the smooth region
    (Lab gradient < 8 after sigma 1.5) connected to the border, plus enclosed smooth regions of the same colour."""
    from scipy.ndimage import gaussian_filter, sobel, label, binary_opening
    import sa_boundary_diag as sbd
    L = gaussian_filter(sbd.lab(img), (1.5, 1.5, 0))
    G = np.sqrt(sum(sobel(L[..., k], 0) ** 2 + sobel(L[..., k], 1) ** 2 for k in range(3)))
    lab, n = label(G < 8)
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]); edge = edge[edge > 0]
    back = np.isin(lab, edge)
    if back.any() and n:
        col = np.median(L[back], 0); sz = np.bincount(lab.ravel(), minlength=n + 1)
        mean = np.stack([np.bincount(lab.ravel(), L[..., k].ravel(), minlength=n + 1) for k in range(3)], 1) / np.maximum(sz, 1)[:, None]
        same = np.nonzero((np.linalg.norm(mean - col, axis=1) <= 10) & (sz >= 64))[0]; same = same[same > 0]
        back |= np.isin(lab, same)
    obj = binary_opening(~back, iterations=2); l2, m = label(obj)
    if m:
        sz2 = np.bincount(l2.ravel()); keep = np.nonzero(sz2 >= 0.005 * obj.size)[0]; keep = keep[keep > 0]; obj = np.isin(l2, keep)
    fr = ~obj; frame = np.r_[fr[:2].ravel(), fr[-2:].ravel(), fr[:, :2].ravel(), fr[:, -2:].ravel()]
    return obj, float(frame.mean())


class QCap:
    def __init__(self, name, store):
        self.name, self.store = name, store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *a, **k):
        from diffusers.models.attention_processor import AttnProcessor
        self.store[self.name] = attn.head_to_batch_dim(attn.to_q(hidden_states)).detach()
        return AttnProcessor()(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--classes", type=int); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import torch, torch.nn.functional as F
    from PIL import Image
    from scripts import sg27_runtime
    from src.attention_capture.live import _seed_everything
    from src.segmentation.attention_aggregation import find_token_subsequence
    import idea1_knockout as k1
    W = work(a.tag); (W / "exemplars").mkdir(parents=True, exist_ok=True)
    pipe = sg27_runtime.make_runtime(k1.MODEL)["pipe"]; dev = pipe.device; tok = pipe.tokenizer
    procs = dict(pipe.unet.attn_processors)
    attn2 = sorted(n.removesuffix(".processor") for n in procs if n.endswith("attn2.processor"))
    a3 = [attn2[i] for i in A3_IDX]; mods = {m: pipe.unet.get_submodule(m) for m in a3}
    classes = CLASSES[: a.classes or None]; rng = np.random.default_rng(0)
    EX = {c: [] for c in classes}; stats = {}
    t00 = time.perf_counter()
    for ci, c in enumerate(CLASSES):
        if c not in classes:
            continue
        prompt = f"a photo of a {c}, isolated on a plain white background"; rec = []
        for j in range(NGEN):
            if len(EX[c]) >= NKEEP:
                break
            seed = 50000 + 100 * ci + j; _seed_everything(seed); lat_in = {}

            def cb(_p, i, _t, kw):
                if i + 1 in CALLS:
                    lat_in[i + 1] = kw["latents"].detach().clone()
                return kw
            with torch.inference_mode():
                pe, ne = pipe.encode_prompt(prompt=prompt, device=dev, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None)
                lat = torch.randn((1, pipe.unet.config.in_channels, 64, 64), device=dev, dtype=pe.dtype)
                im = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, latents=lat, height=512, width=512, num_inference_steps=30,
                          guidance_scale=7.5, output_type="pil", callback_on_step_end=cb,
                          callback_on_step_end_tensor_inputs=["latents"]).images[0]
            img = np.asarray(im.convert("RGB")); obj, frame = key_mask(img); area = float(obj.mean())
            keep = bool(frame >= 0.95 and 0.05 <= area <= 0.70)
            rec.append(dict(seed=seed, frame=frame, area=area, keep=keep))
            stem = W / "exemplars" / f"{c.replace(' ', '_')}_{seed}_{'keep' if keep else 'drop'}"
            im.save(stem.with_suffix(".jpg"), quality=90); Image.fromarray((obj * 255).astype(np.uint8)).save(str(stem) + "_mask.png")
            if not keep or set(lat_in) != set(CALLS):
                continue
            TS = [int(t) for t in pipe.scheduler.timesteps]
            ot = torch.from_numpy(obj.astype(np.float32))[None, None].to(dev)
            cells = {}
            for s in (64, 16, 8):
                fr = F.adaptive_avg_pool2d(ot, s).reshape(-1)
                po = torch.nonzero(fr >= 0.9).reshape(-1).cpu().numpy(); ne_ = torch.nonzero(fr <= 0.1).reshape(-1).cpu().numpy()
                if s == 64:
                    po = rng.permutation(po)[:192]; ne_ = rng.permutation(ne_)[:192]
                idx = np.r_[po, ne_]; cells[s] = (torch.from_numpy(idx).to(dev), torch.from_numpy(np.r_[np.ones(len(po)), np.zeros(len(ne_))]).float().to(dev))
            if min(int(cells[s][1].sum()) for s in (64, 16)) == 0:
                rec[-1]["keep"] = False; continue
            store = {}; q = {}; res_of = {}
            pipe.unet.set_attn_processor({n: (QCap(n.removesuffix(".processor"), store) if n.removesuffix(".processor") in mods
                                              else getattr(p, "original", p)) for n, p in procs.items()})
            try:
                with torch.no_grad():
                    for call in CALLS:
                        store.clear()
                        pipe.unet(lat_in[call], torch.tensor(TS[call], device=dev), encoder_hidden_states=pe, return_dict=False)
                        for mi, m in enumerate(a3):
                            s = int(store[m].shape[1] ** 0.5); res_of[mi] = s; q[(mi, call)] = store[m][:, cells[s][0]].half()
            finally:
                pipe.unet.set_attn_processor(dict(procs))
            EX[c].append(dict(q=q, y={mi: cells[res_of[mi]][1] for mi in range(len(a3))}))
        stats[c] = dict(generated=len(rec), kept=len(EX[c]), exemplars=rec)
        print(f"[gs] {c}: kept {len(EX[c])}/{len(rec)}  ({time.perf_counter() - t00:.0f}s)", flush=True)

    # ------------------------------------------------------------------ token optimisation
    def ctx(name):
        ids = tok(name, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            H = pipe.text_encoder(ids)[0][0].float()
        return H, list(find_token_subsequence(ids[0].tolist(), tok(name, add_special_tokens=False)["input_ids"]))

    def scores(ex, H, target, logit=False):
        """{(mi, call): per-cell log class probability (or raw class logit)}"""
        out = {}
        for mi, m in enumerate(a3):
            mod = mods[m]; K = mod.head_to_batch_dim(mod.to_k(H[None]).float())
            for call in CALLS:
                L = ex["q"][(mi, call)].float() @ K.transpose(1, 2) * mod.scale
                out[(mi, call)] = L[..., target].mean(-1).mean(0) if logit else (torch.softmax(L, -1)[..., target].mean(-1).mean(0) + 1e-12).log()
        return out
    torch.manual_seed(0); gen = np.random.default_rng(0); tokens = {}
    for c in classes:
        own = EX[c]; others = [e for d in classes if d != c for e in EX[d]]
        if len(own) < NMIN or not others:
            stats[c].update(token="original (too few exemplars)"); continue
        H0, target = ctx(c); en = H0[target].norm(dim=-1).mean()
        theta = torch.zeros(768, device=dev, requires_grad=True)
        av = torch.ones(len(a3), device=dev, requires_grad=True); cv = torch.zeros(len(a3), device=dev, requires_grad=True)
        opt = torch.optim.Adam([{"params": [theta], "lr": LR}, {"params": [av, cv], "lr": LR_AC}]); l0 = l1 = None
        for step in range(STEPS):
            batch = [(own[i], True) for i in gen.integers(0, len(own), 4)] + [(others[i], False) for i in gen.integers(0, len(others), 4)]
            H = H0.clone(); H[target] = H[target] + en * theta
            Z = {k: [] for k in ((mi, call) for mi in range(len(a3)) for call in CALLS)}; Y = {k: [] for k in Z}
            for ex, pos in batch:
                sc = scores(ex, H, target)
                for k, s in sc.items():
                    Z[k].append(s); Y[k].append(ex["y"][k[0]] if pos else torch.zeros_like(s))
            if step == 0:
                with torch.no_grad():
                    for mi in range(len(a3)):
                        cv[mi] = torch.cat([torch.cat(Z[(mi, call)]) for call in CALLS]).mean()
            loss = 0
            for k in Z:
                z = av[k[0]] * (torch.cat(Z[k]) - cv[k[0]]); y = torch.cat(Y[k])
                pw = (1 - y).sum() / y.sum().clamp_min(1)
                loss = loss + F.binary_cross_entropy_with_logits(z, y, pos_weight=pw) / len(Z)
            l1 = float(loss.detach()); l0 = l1 if l0 is None else l0
            opt.zero_grad(); (loss + PEN * (theta ** 2).sum()).backward(); opt.step()
        # closed-form linear step from the same exemplars
        delta = torch.zeros(768, device=dev, requires_grad=True); H = H0.clone(); H[target] = H[target] + delta; J = 0
        neg8 = [others[i] for i in np.random.default_rng(1).permutation(len(others))[:8]]
        P, N = {}, {}
        for ex in own:
            for k, s in scores(ex, H, target, logit=True).items():
                y = ex["y"][k[0]]; P.setdefault(k, []).append(s[y > 0.5]); N.setdefault(k, []).append(s[y < 0.5])
        for ex in neg8:
            for k, s in scores(ex, H, target, logit=True).items():
                N[k].append(s)
        for k in P:
            J = J + torch.cat(P[k]).mean() - torch.cat(N[k]).mean()
        J.backward(); g = delta.grad / delta.grad.norm().clamp_min(1e-12)
        tokens[c] = dict(gs=(en * theta.detach()).cpu().numpy(), gs_lin=(BETA * en * g).cpu().numpy())
        stats[c].update(token="learned", loss_first=l0, loss_last=l1, theta_norm=float(theta.detach().norm()), piece_norm=float(en))
        print(f"[gs] token {c}: loss {l0:.3f} -> {l1:.3f}, |theta| {float(theta.detach().norm()):.3f}", flush=True)
    np.savez(W / "tokens.npz", **{f"{k}/{c}": v[k] for c, v in tokens.items() for k in ("gs", "gs_lin")})
    (W / "stats.json").write_text(json.dumps(stats, indent=1))
    print("kept per class:", {c: stats[c]["kept"] for c in classes})


if __name__ == "__main__":
    main()
