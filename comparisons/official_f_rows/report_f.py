#!/usr/bin/env python3
"""Threshold windows for DenseCRF, and the result tables of the JC #3 remaining rows, paired with BRACE.

  taus-voc   -> taus_voc.json   VOC-Sim DenseCRF indices: raw full-VOC-Sim optimum +-3 grid steps, plus 0.5
  taus-coco  -> taus_coco.json  COCO-Cap DenseCRF indices: DenseCRF full-VOC-Sim-best index of the window, plus 0.5
  report     -> result.md, summary.json (in this directory)

Metric, OOF and bootstrap: scripts/sg_class_comp.py (miou, oof, boot), unchanged; BRACE = comparisons/official
reproduction (V3 hierarchy, + DenseCRF R2), its rule as in comparisons/official/report.py.
"""
from __future__ import annotations

import json, sys, zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "comparisons/official"))
import sg_class_comp as cc  # noqa: E402
from common import THR_EXT  # noqa: E402

HERE = Path(__file__).resolve().parent
CAP = Path.home() / "TRDP/external/n_side/official_f_rows"
BR = Path.home() / "TRDP/external/n_side/brace_repro"
METHODS = ("ovam_w", "ovam_tf", "daam", "seediff", "sclip")
LAB = {"ovam_w": "OVAM, class word", "ovam_t": "OVAM, optimized embedding [SoT, token] (few-shot, not training-free)",
       "ovam_tf": "OVAM, released optimized token (few-shot, not training-free)",
       "daam": "DAAM", "seediff": "SeeDiff", "sclip": "SCLIP ViT-B/16 (image only)"}
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
NB = 2000
I05 = int(np.flatnonzero(np.isclose(THR_EXT, 0.5))[0])


def S_of(ids, cls):
    return {"grp": np.array(ids), "cls": np.array(cls), "fold": np.array([zlib.crc32(i.encode()) % 5 for i in ids])}


def load():
    D = {}
    for ds in ("voc", "coco"):
        ids = sorted(p.stem for p in (CAP / ds).glob("*.npz") if not p.stem.endswith(".tmp"))
        zb = {i: np.load(BR / f"{ds}_scores" / f"{i}.npz") for i in ids}
        cls = [str(zb[i]["cls"]) for i in ids]
        d = {"ids": ids, "S": S_of(ids, cls),
             "brace_crf": np.stack([zb[i]["V3+R2|raw"] for i in ids]), "brace_raw": np.stack([zb[i]["V3|raw"] for i in ids]),
             "m": {}}
        for m in METHODS:
            f = {i: CAP / "scores" / m / ds / f"{i}.npz" for i in ids}
            have = np.array([f[i].exists() for i in ids])
            if not have.any():
                continue
            z = {i: dict(np.load(f[i])) for i in ids if f[i].exists()}
            sub = [i for i in ids if i in z]
            rec = {"sub": have, "raw": np.stack([z[i]["soft|raw"] for i in sub]),
                   "own_raw": np.stack([z[i]["own|raw"][0] for i in sub]), "own_crf": np.stack([z[i]["own|crf"][0] for i in sub]),
                   "same": np.array([bool(z[i]["same_as_brace"]) for i in sub])}
            ks = sorted({int(k.split("|")[1]) for i in sub for k in z[i] if k.startswith("crf|")})
            ks = [k for k in ks if all(f"crf|{k}" in z[i] for i in sub)]
            rec["crf"] = {k: np.stack([z[i][f"crf|{k}"][0] for i in sub]) for k in ks}
            d["m"][m] = rec
        # PREREG: the class-word-in-prompt subset is DAAM's tokenizer search; SeeDiff's own first-sub-token match
        # finds one more COCO-Cap item (the only "potted plant" one, a 1-item class), so SeeDiff uses DAAM's subset.
        if "daam" in d["m"] and "seediff" in d["m"]:
            r = d["m"]["seediff"]; keep = d["m"]["daam"]["sub"][r["sub"]]
            if not keep.all():
                for k in ("raw", "own_raw", "own_crf", "same"):
                    r[k] = r[k][keep]
                r["crf"] = {t: v[keep] for t, v in r["crf"].items()}
                r["sub"] = r["sub"] & d["m"]["daam"]["sub"]
        D[ds] = d
    return D


def miou(S, P, sub=None):
    sub = np.ones(len(P), bool) if sub is None else sub
    return float(cc.miou(P[:, None], S["cls"], sub)[0])


def subS(S, sub):
    return {k: v[sub] for k, v in S.items()}


def taus_voc():
    D = load(); out = {"voc": {}}
    for m, r in D["voc"]["m"].items():
        S = subS(D["voc"]["S"], r["sub"])
        tb = int(np.argmax(cc.miou(r["raw"], S["cls"], np.ones(len(r["raw"]), bool))))
        w = sorted(set(range(max(0, tb - 3), min(len(THR_EXT), tb + 4))) | {I05})
        out["voc"][m] = w
    (CAP / "taus_voc.json").write_text(json.dumps(out)); print(out)


def taus_coco():
    D = load(); out = {"coco": {}}
    for m, r in D["voc"]["m"].items():
        S = subS(D["voc"]["S"], r["sub"]); ks = sorted(r["crf"])
        X = np.stack([r["crf"][k] for k in ks], 1)
        tb = ks[int(np.argmax(cc.miou(X, S["cls"], np.ones(len(X), bool))))]
        out["coco"][m] = sorted({tb, I05})
    (CAP / "taus_coco.json").write_text(json.dumps(out)); print(out)


def rules(D, m):
    """Per-item counts per rule for VOC (OOF where selected) and COCO; None where not available."""
    rv, rc = D["voc"]["m"][m], D["coco"]["m"].get(m)
    Sv = subS(D["voc"]["S"], rv["sub"])
    R = {}
    _, Pv, tv = cc.oof(Sv, rv["raw"]); tb = int(np.argmax(cc.miou(rv["raw"], Sv["cls"], np.ones(len(Pv), bool))))
    R["sel_raw"] = (Pv, rc["raw"][:, tb] if rc else None, float(THR_EXT[tb]), [float(THR_EXT[t]) for t in tv])
    ks = sorted(rv["crf"]); Xv = np.stack([rv["crf"][k] for k in ks], 1)
    _, Pv, tv = cc.oof(Sv, Xv); tbk = ks[int(np.argmax(cc.miou(Xv, Sv["cls"], np.ones(len(Pv), bool))))]
    R["sel_crf"] = (Pv, rc["crf"][tbk] if rc and tbk in rc["crf"] else None, float(THR_EXT[tbk]), [float(THR_EXT[ks[t]]) for t in tv])
    R["fix_raw"] = (rv["raw"][:, I05], rc["raw"][:, I05] if rc else None, 0.5, None)
    R["fix_crf"] = (rv["crf"][I05], rc["crf"][I05] if rc and I05 in rc["crf"] else None, 0.5, None)
    R["own_raw"] = (rv["own_raw"], rc["own_raw"] if rc else None, None, None)
    R["own_crf"] = (rv["own_crf"], rc["own_crf"] if rc else None, None, None)
    R["best_raw"] = tuple(float(cc.miou(r["raw"], subS(D[ds]["S"], r["sub"])["cls"], np.ones(len(r["raw"]), bool)).max())
                          if r else None for ds, r in (("voc", rv), ("coco", rc)))
    return R


def brace_rules(D):
    out = {}
    for crf in (False, True):
        Xv = D["voc"]["brace_crf" if crf else "brace_raw"]; Xc = D["coco"]["brace_crf" if crf else "brace_raw"]
        _, Pv, _ = cc.oof(D["voc"]["S"], Xv); tb = int(np.argmax(cc.miou(Xv, D["voc"]["S"]["cls"], np.ones(len(Xv), bool))))
        out[crf] = {"voc": Pv, "coco": Xc[:, tb], "tau": float(THR_EXT[tb])}
    return out


def boot(S, Pa, Pb):
    pt = miou(S, Pa) - miou(S, Pb); lo, hi = cc.boot(S, Pa, Pb, n=NB)
    return [pt, float(lo), float(hi)]


def fmt(d):
    return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"


def report():
    D = load(); B = brace_rules(D); J = {"rows": {}, "paired": {}, "checks": {}}
    L = ["# JC #3 remaining comparison rows, paired with BRACE (`PREREG.md`)\n",
         "Side N, exploratory. One SD 1.5 generation per item (wald V100, runs/20 corrected prompts, manifest seeds), "
         "all generation-time methods read from that trajectory; SCLIP on the generated image. BRACE = the N-side "
         "reproduction of runs/20 (`comparisons/official/`). Class-macro mIoU; paired Δ = method − BRACE, "
         f"{NB}-sample image bootstrap, 95% interval. *VOC-sel.* = 5-fold OOF threshold on VOC-Sim, full-VOC-Sim-best "
         "threshold applied unchanged to COCO-Cap. DenseCRF rows use a 7-step threshold window around the raw VOC-Sim "
         "optimum (see `score_f.py`). DAAM and SeeDiff need the class word in the prompt: their COCO-Cap rows and Δ use "
         "that subset, with BRACE on the same subset.\n"]
    L.append("## Data checks\n")
    for ds in ("voc", "coco"):
        same = [D[ds]["m"][m]["same"] for m in D[ds]["m"]][0] if D[ds]["m"] else np.array([])
        n_sub = {m: int(D[ds]["m"][m]["sub"].sum()) for m in D[ds]["m"]}
        J["checks"][ds] = {"items": len(D[ds]["ids"]), "image_identical_to_brace": int(same.sum()), "subset_sizes": n_sub}
        L.append(f"- {NAME[ds]}: {len(D[ds]['ids'])} items; generated image bit-identical to the BRACE reproduction's: "
                 f"{int(same.sum())}/{len(same)}; items per method: " + ", ".join(f"{m} {n}" for m, n in n_sub.items()) + "\n")
    L.append("\n## Rows (VOC-Sim / COCO-Cap)\n")
    L.append("| Method | VOC-sel., raw | VOC-sel., DenseCRF | fixed 0.5, raw | fixed 0.5, DenseCRF | own rule, raw | own rule, DenseCRF | τ (raw / CRF) |")
    L.append("|---|---|---|---|---|---|---|---|")
    bv = lambda crf: f"{miou(D['voc']['S'], B[crf]['voc']):.3f} / {miou(D['coco']['S'], B[crf]['coco']):.3f}"
    L.append(f"| BRACE (reproduction) | {bv(False)} | {bv(True)} | | | | | {B[False]['tau']:.3f} / {B[True]['tau']:.3f} |")
    for m in METHODS:
        if m not in D["voc"]["m"]:
            continue
        R = rules(D, m); row = {}
        cells = []
        for k in ("sel_raw", "sel_crf", "fix_raw", "fix_crf", "own_raw", "own_crf"):
            pv, pc = R[k][0], R[k][1]
            Sv = subS(D["voc"]["S"], D["voc"]["m"][m]["sub"])
            Sc = subS(D["coco"]["S"], D["coco"]["m"][m]["sub"]) if m in D["coco"]["m"] else None
            a = miou(Sv, pv); c = miou(Sc, pc) if pc is not None else float("nan")
            row[k] = [a, c]; cells.append(f"{a:.3f} / {c:.3f}")
        row["tau"] = [R["sel_raw"][2], R["sel_crf"][2]]; row["oof_taus"] = [R["sel_raw"][3], R["sel_crf"][3]]
        row["best_on_set_raw_audit"] = list(R["best_raw"])
        J["rows"][m] = row
        L.append(f"| {LAB[m]} | " + " | ".join(cells) + f" | {R['sel_raw'][2]:.3f} / {R['sel_crf'][2]:.3f} |")
        # paired
        P = {}
        for crf, key in ((False, "sel_raw"), (True, "sel_crf")):
            for ds, j in (("voc", 0), ("coco", 1)):
                if m not in D[ds]["m"] or R[key][j] is None:
                    continue
                sub = D[ds]["m"][m]["sub"]; S = subS(D[ds]["S"], sub)
                P[f"{ds}|{'crf' if crf else 'raw'}"] = boot(S, R[key][j], B[crf][ds][sub])
                same = D[ds]["m"][m]["same"]
                if not same.all() and same.any():
                    P[f"{ds}|{'crf' if crf else 'raw'}|identical_only"] = boot(subS(S, same), R[key][j][same], B[crf][ds][sub][same])
        for ds in ("voc", "coco"):
            for crf in (False, True):
                if m in D[ds]["m"]:
                    sub = D[ds]["m"][m]["sub"]
                    if not sub.all():
                        P[f"{ds}|brace_on_subset|{'crf' if crf else 'raw'}"] = miou(subS(D[ds]["S"], sub), B[crf][ds][sub])
        J["paired"][m] = P
    L.append("\n## Paired Δ, method − BRACE, both at VOC-sel. (primary)\n")
    L.append("| Method | VOC-Sim raw | COCO-Cap raw | VOC-Sim DenseCRF | COCO-Cap DenseCRF |")
    L.append("|---|---|---|---|---|")
    for m, P in J["paired"].items():
        L.append(f"| {LAB[m]} | " + " | ".join(fmt(P[k]) if k in P else "—" for k in ("voc|raw", "coco|raw", "voc|crf", "coco|crf")) + " |")
    subs = [(m, P) for m, P in J["paired"].items() if any("brace_on_subset" in k for k in P)]
    if subs:
        L.append("\nBRACE on the class-word-in-prompt subsets (VOC-sel.; raw / DenseCRF): " + "; ".join(
            f"{m} {NAME[ds]} {P[f'{ds}|brace_on_subset|raw']:.3f} / {P[f'{ds}|brace_on_subset|crf']:.3f}"
            for m, P in subs for ds in ("voc", "coco") if f"{ds}|brace_on_subset|raw" in P) + ".\n")
    idn = [(m, k, v) for m, P in J["paired"].items() for k, v in P.items() if k.endswith("identical_only")]
    if idn:
        L.append("\nSensitivity, items whose image is bit-identical to BRACE's: " + "; ".join(f"{m} {k.split('|')[0]} {k.split('|')[1]} {fmt(v)}" for m, k, v in idn) + ".\n")
    (HERE / "result.md").write_text("\n".join(L) + "\n")
    (HERE / "summary.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    {"taus-voc": taus_voc, "taus-coco": taus_coco, "report": report}[sys.argv[1]]()
