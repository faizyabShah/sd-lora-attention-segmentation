#!/usr/bin/env python3
"""Official N-side comparison rows, paired with BRACE (workplan Section 3 contract).

Inputs (local, git-ignored): BRACE counts from `brace_repro.py score`, N counts from `score_n.py`.
Metric code reused unchanged from `scripts/sg_class_comp.py`: `miou` (class-macro: per-class IoU pooled over
images, then mean over classes), `oof` (5-fold out-of-fold threshold, folds crc32(image_id) % 5, as runs/20),
`boot` (image bootstrap of the paired difference, here with 2000 resamples).

Threshold rules
  VOC-sel.  VOC-Sim: 5-fold OOF threshold; COCO-Cap: the full-VOC-Sim-best threshold, unchanged (as runs/20).
  fixed     0.5 for every method (N's sanity convention); 0.55 for BRACE (its frozen threshold) and also 0.5.
  own       the method's own binarisation (GoCA offset rule at offset 0; NERVE cam_to_label mask). Rules that
            are selected by a sweep on the evaluated set (GoCA offset, DiffSegmenter background score) are
            reported as in-sample audit rows only; their VOC-sel. version is reported as well.
  best      best threshold on the evaluated set (audit only, uses that set's labels).
Raw and DenseCRF R2 are never mixed inside one comparison.

-> comparisons/official/{result.md, summary.json, per_item_counts.csv}
"""
from __future__ import annotations

import csv, json, sys, zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sg_class_comp as cc  # noqa: E402
from common import EXT, OFFICIAL_OUT, THR_EXT, THR_R20  # noqa: E402
import score_n as sn  # noqa: E402

BR = EXT / "brace_repro"
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
MLAB = {"brace": "BRACE (SD 1.5, generation-time; CA seed + capped hierarchy)",
        "goca_sd15": "GoCA, SD 1.5", "goca_sdxl": "GoCA, SDXL", "goca_sd21": "GoCA, SD 2.1-base (community mirror)", "diffseg": "DiffSegmenter, SD 1.5",
        "nerve_A": "NERVE A, SD 1.5 + CLIP L/14", "nerve_V": "NERVE V, SD 1.5 + CLIP L/14",
        "nerve_W": "NERVE W, SD 1.5 + CLIP L/14"}
SHORT = {"brace": "BRACE", "goca_sd15": "GoCA SD1.5", "goca_sdxl": "GoCA SDXL", "goca_sd21": "GoCA SD2.1", "diffseg": "DiffSegmenter",
         "nerve_A": "NERVE A", "nerve_V": "NERVE V", "nerve_W": "NERVE W"}
NB = 2000
ti = lambda t: int(np.flatnonzero(np.isclose(THR_EXT, t))[0])


def load():
    D = {}
    for ds in ("voc", "coco"):
        man = {r["key"]: r for r in sn.items(ds)}
        ids = sorted(man)
        S = {"grp": np.array(ids), "cls": np.array([man[i]["classname"] for i in ids]),
             "fold": np.array([zlib.crc32(i.encode()) % 5 for i in ids])}
        C = {}
        zb = {i: np.load(BR / f"{ds}_scores" / f"{i}.npz") for i in ids}
        for c in ("seed", "V0", "V3"):
            C[("brace", c, False)] = np.stack([zb[i][f"{c}|raw"] for i in ids])
        C[("brace", "V3", True)] = np.stack([zb[i]["V3+R2|raw"] for i in ids])
        info = {k: np.array([zb[i][k] for i in ids]) for k in ("off_diff", "off_maxdiff", "changed", "seconds", "peak_mib")}
        bcls = np.array([str(zb[i]["cls"]) for i in ids])
        tcrc = {}
        for m in sn.METHODS:
            z = {i: np.load(OFFICIAL_OUT / m / ds / f"{i}.npz") for i in ids}
            C[(m, "soft", False)] = np.stack([z[i]["soft|raw"] for i in ids])
            C[(m, "soft", True)] = np.stack([z[i]["soft+R2|raw"] for i in ids])
            if "own|raw" in z[ids[0]].files:
                C[(m, "own", False)] = np.stack([z[i]["own|raw"] for i in ids])
                C[(m, "own", True)] = np.stack([z[i]["own+R2|raw"] for i in ids])
            tcrc[m] = np.array([int(z[i]["truth_crc"]) for i in ids])
        # ground truth identity: BRACE's live-sample target mask vs the annotation PNG used by every N method
        bt, gpus = [], set()
        for i in ids:
            zc = np.load(BR / ds / f"{i}.npz")
            bt.append(zlib.crc32(np.packbits(zc["truth"].astype(bool)).tobytes())); gpus.add(str(zc["gpu"]))
        bt = np.array(bt); info["gpu"] = sorted(gpus)
        D[ds] = {"S": S, "C": C, "info": info, "bcls_match": bool((bcls == S["cls"]).all()),
                 "truth_same": int((bt == tcrc[sn.METHODS[0]]).sum()),
                 "truth_same_all_n": bool(all((tcrc[m] == tcrc[sn.METHODS[0]]).all() for m in sn.METHODS))}
    return D


def miou(S, P, sub=None):
    sub = np.ones(len(P), bool) if sub is None else sub
    return float(cc.miou(P[:, None], S["cls"], sub)[0])


def sub_S(S, sub):
    return {k: v[sub] for k, v in S.items()}


def rule_preds(D, key, grid=None):
    """Per-item counts [n, 5] for each threshold rule of one cell; grid restricts the threshold indices."""
    Sv, Sc = D["voc"]["S"], D["coco"]["S"]
    Xv, Xc = D["voc"]["C"][key], D["coco"]["C"][key]
    if grid is not None:
        Xv, Xc = Xv[:, grid], Xc[:, grid]
    out = {}
    _, Pv, taus = cc.oof(Sv, Xv)
    tb = int(np.argmax(cc.miou(Xv, Sv["cls"], np.ones(len(Xv), bool))))
    out["sel"] = {"voc": Pv, "coco": Xc[:, tb], "tau_full_voc": tb, "taus_oof": taus}
    out["best"] = {ds: float(cc.miou(X, D[ds]["S"]["cls"], np.ones(len(X), bool)).max()) for ds, X in (("voc", Xv), ("coco", Xc))}
    out["best_idx"] = {ds: int(np.argmax(cc.miou(X, D[ds]["S"]["cls"], np.ones(len(X), bool)))) for ds, X in (("voc", Xv), ("coco", Xc))}
    return out, Xv, Xc


def boot(S, Pa, Pb, sub=None):
    if sub is not None:
        S, Pa, Pb = sub_S(S, sub), Pa[sub], Pb[sub]
    pt = miou(S, Pa) - miou(S, Pb); lo, hi = cc.boot(S, Pa, Pb, n=NB)
    return pt, float(lo), float(hi)


def fmt(d):
    return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"


def main():
    D = load()
    J = {"label": "official N-side comparison rows (workplan Section 3); see result.md for rule labels",
         "threshold_grid": THR_EXT.tolist(), "bootstrap": NB, "rows": [], "paired": [], "brace_repro": {}, "checks": {}}
    L = []; w = L.append
    PRED = {}          # (method, crf, rule) -> {ds: counts [n, 5]}, for the per-item CSV and the paired deltas
    TAU = {}

    # ---------------- BRACE reproduction (runs/20 grid and convention)
    rep = {}
    for c, crf in (("V3", True), ("V3", False), ("V0", False), ("seed", False)):
        r20, Xv, Xc = rule_preds(D, ("brace", c, crf), grid=THR_R20)
        i55 = int(np.flatnonzero(np.isclose(THR_EXT[THR_R20], 0.55))[0])
        rep[(c, crf)] = {"voc_sel": miou(D["voc"]["S"], r20["sel"]["voc"]), "coco_sel": miou(D["coco"]["S"], r20["sel"]["coco"]),
                         "voc_055": miou(D["voc"]["S"], Xv[:, i55]), "coco_055": miou(D["coco"]["S"], Xc[:, i55]),
                         "tau_full_voc": float(THR_EXT[THR_R20][r20["sel"]["tau_full_voc"]]),
                         "coco_best": r20["best"]["coco"]}
    REF20 = {("V3", True): (0.832, 0.709, 0.833, 0.709, 0.550, 0.719), ("V3", False): (0.818, 0.693, 0.819, 0.693, None, None),
             ("V0", False): (0.811, 0.670, 0.792, 0.690, None, None), ("seed", False): (0.706, 0.592, 0.613, 0.510, None, None)}
    J["brace_repro"] = {f"{c}{'+R2' if crf else ''}": {**v, "runs20": REF20[(c, crf)]} for (c, crf), v in rep.items()}

    # ---------------- rows
    methods = ("brace",) + sn.METHODS
    rows = []
    for m in methods:
        for crf in (False, True):
            if m == "brace":
                if ("brace", "V3", crf) not in D["voc"]["C"]:
                    continue
                key = ("brace", "V3", crf)
            else:
                key = (m, "soft", crf)
            r, Xv, Xc = rule_preds(D, key)
            X = {"voc": Xv, "coco": Xc}
            PRED[(m, crf, "VOC-sel.")] = {ds: r["sel"][ds] for ds in ("voc", "coco")}
            TAU[(m, crf, "VOC-sel.")] = {"full_voc": float(THR_EXT[r["sel"]["tau_full_voc"]]),
                                         "oof_folds": [float(THR_EXT[t]) for t in r["sel"]["taus_oof"]]}
            PRED[(m, crf, "fixed 0.5")] = {ds: X[ds][:, ti(0.5)] for ds in X}
            if m == "brace":
                PRED[(m, crf, "fixed 0.55 (frozen)")] = {ds: X[ds][:, ti(0.55)] for ds in X}
            rows.append((m, crf, "VOC-sel.", {ds: miou(D[ds]["S"], r["sel"][ds]) for ds in X},
                         f"τ full-VOC {THR_EXT[r['sel']['tau_full_voc']]:.3f}; OOF folds " +
                         "/".join(f"{THR_EXT[t]:.3f}" for t in r["sel"]["taus_oof"])))
            rows.append((m, crf, "fixed 0.5", {ds: miou(D[ds]["S"], X[ds][:, ti(0.5)]) for ds in X}, "fixed-threshold"))
            if m == "brace":
                rows.append((m, crf, "fixed 0.55 (frozen)", {ds: miou(D[ds]["S"], X[ds][:, ti(0.55)]) for ds in X},
                             "fixed-threshold; BRACE's frozen rule"))
            rows.append((m, crf, "best on set (audit)", r["best"],
                         "in-sample audit: τ " + ", ".join(f"{NAME[ds]} {THR_EXT[r['best_idx'][ds]]:.3f}" for ds in X)))
            if m == "diffseg":
                lim = np.flatnonzero((THR_EXT >= 0.30 - 1e-9) & (THR_EXT <= 0.74))
                bs = {ds: float(cc.miou(X[ds][:, lim], D[ds]["S"]["cls"], np.ones(len(X[ds]), bool)).max()) for ds in X}
                rows.append((m, crf, "own rule (in-sample bg sweep)", bs,
                             "DiffSegmenter's evaluation script: background score swept 0.30–0.74 on the evaluated set "
                             "(here step 0.025); in-sample, uses labels"))
            if (m, "own", crf) in D["voc"]["C"]:
                Ov, Oc = D["voc"]["C"][(m, "own", crf)], D["coco"]["C"][(m, "own", crf)]
                if m.startswith("goca"):
                    z0 = int(np.flatnonzero(np.isclose(sn.GOCA_OFFSETS, 0))[0])
                    PRED[(m, crf, "own rule (offset 0)")] = {"voc": Ov[:, z0], "coco": Oc[:, z0]}
                    rows.append((m, crf, "own rule (offset 0)", {"voc": miou(D["voc"]["S"], Ov[:, z0]), "coco": miou(D["coco"]["S"], Oc[:, z0])},
                                 "GoCA argmax(class, background + offset), offset 0, no tuning"))
                    _, Pv, taus = cc.oof(D["voc"]["S"], Ov); tb = int(np.argmax(cc.miou(Ov, D["voc"]["S"]["cls"], np.ones(len(Ov), bool))))
                    PRED[(m, crf, "own rule, VOC-sel. offset")] = {"voc": Pv, "coco": Oc[:, tb]}
                    rows.append((m, crf, "own rule, VOC-sel. offset", {"voc": miou(D["voc"]["S"], Pv), "coco": miou(D["coco"]["S"], Oc[:, tb])},
                                 f"offset full-VOC {sn.GOCA_OFFSETS[tb]:+.1f}; OOF folds " + "/".join(f"{sn.GOCA_OFFSETS[t]:+.1f}" for t in taus)))
                    bo = {ds: cc.miou(O, D[ds]["S"]["cls"], np.ones(len(O), bool)) for ds, O in (("voc", Ov), ("coco", Oc))}
                    rows.append((m, crf, "own rule, best offset on set (audit)", {ds: float(v.max()) for ds, v in bo.items()},
                                 "in-sample audit (GoCA's eval sweeps offsets −0.4…+0.4 on the evaluated set): offset " +
                                 ", ".join(f"{NAME[ds]} {sn.GOCA_OFFSETS[int(np.argmax(v))]:+.1f}" for ds, v in bo.items())))
                else:
                    PRED[(m, crf, "own rule")] = {"voc": Ov[:, 0], "coco": Oc[:, 0]}
                    rows.append((m, crf, "own rule", {"voc": miou(D["voc"]["S"], Ov[:, 0]), "coco": miou(D["coco"]["S"], Oc[:, 0])},
                                 "NERVE cam_to_label (bkg_thre 0, cls_thre 0.5) with its image-score gate; fixed"))
    for m, crf, rule, v, note in rows:
        J["rows"].append({"method": m, "densecrf": crf, "rule": rule, "voc_sim": v["voc"], "coco_cap": v["coco"], "note": note})

    # ---------------- paired deltas vs BRACE (same images, same CRF status)
    resolved = {ds: D[ds]["info"]["off_diff"] < 2 for ds in D}
    close1 = {ds: D[ds]["info"]["off_diff"] < 1 for ds in D}
    PAIRS = []
    for m in sn.METHODS:
        for crf in (False, True):
            PAIRS.append((m, crf, "VOC-sel.", "VOC-sel."))
            PAIRS.append((m, crf, "fixed 0.5", "fixed 0.55 (frozen)"))
            PAIRS.append((m, crf, "fixed 0.5", "fixed 0.5"))
            for own in ("own rule (offset 0)", "own rule, VOC-sel. offset", "own rule"):
                if (m, crf, own) in PRED:
                    PAIRS.append((m, crf, own, "fixed 0.55 (frozen)"))
    for m, crf, ra, rb in PAIRS:
        rec = {"method": m, "densecrf": crf, "method_rule": ra, "brace_rule": rb}
        for ds in ("voc", "coco"):
            S = D[ds]["S"]; Pa = PRED[(m, crf, ra)][ds]; Pb = PRED[("brace", crf, rb)][ds]
            rec[ds] = boot(S, Pa, Pb)
            if ra == "VOC-sel.":
                rec[ds + "_resolved"] = boot(S, Pa, Pb, resolved[ds])
                rec[ds + "_close1"] = boot(S, Pa, Pb, close1[ds])
        J["paired"].append(rec)

    # ---------------- checks
    for ds in D:
        inf = D[ds]["info"]
        J["checks"][ds] = {"n": int(len(D[ds]["S"]["grp"])), "changed_prompt": int(inf["changed"].sum()),
                           "bit_identical_to_official": int((inf["off_maxdiff"] == 0).sum()),
                           "mean_abs_diff_lt1": int((inf["off_diff"] < 1).sum()), "mean_abs_diff_ge2": int((inf["off_diff"] >= 2).sum()),
                           "median_mean_abs_diff": float(np.median(inf["off_diff"])),
                           "class_labels_match": D[ds]["bcls_match"], "truth_identical_brace_vs_annotation": D[ds]["truth_same"],
                           "truth_identical_across_n_methods": D[ds]["truth_same_all_n"],
                           "brace_seconds_median": float(np.median(inf["seconds"])), "brace_seconds_total": float(inf["seconds"].sum()),
                           "brace_peak_mib_max": float(inf["peak_mib"].max()), "brace_gpu": inf["gpu"]}
    J["tau"] = {f"{m}|{'R2' if crf else 'raw'}|{r}": v for (m, crf, r), v in TAU.items()}

    # ---------------- per-item counts CSV
    with (HERE / "per_item_counts.csv").open("w", newline="") as h:
        wr = csv.writer(h); wr.writerow(["method", "dataset", "key", "classname", "densecrf", "rule", "tp", "fp", "fn"])
        for (m, crf, rule), P in sorted(PRED.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            for ds in ("voc", "coco"):
                S = D[ds]["S"]
                for i in range(len(S["grp"])):
                    wr.writerow([m, NAME[ds], S["grp"][i], S["cls"][i], int(crf), rule, *map(int, P[ds][i, :3])])
    (HERE / "summary.json").write_text(json.dumps(J, indent=1))

    # ---------------- result.md
    ck = J["checks"]
    w("# Official comparison rows, side N (GoCA, NERVE, DiffSegmenter), paired with BRACE\n")
    w("Workplan Section 3 contract. Class-macro mIoU (per-class IoU pooled over images, then mean over classes). "
      "Scripts: `comparisons/official/` (`brace_repro.py`, `score_n.py`, `report.py`); metric, OOF and bootstrap code "
      "reused unchanged from `scripts/sg_class_comp.py` (`miou`, `oof`, `boot`); DenseCRF R2 = "
      "`src/segmentation/ovam_full.apply_ovam_densecrf` (OVAM parameters (10, 80, 13, 3, 3, 5), SimpleCRF 0.2.1.1). "
      f"Threshold grid ({len(THR_EXT)} values): {', '.join(f'{t:g}' for t in THR_EXT)}. Paired Δ: {NB}-sample image bootstrap, 95% interval.\n")
    w("**Labels.** *VOC-sel.* = threshold selected on VOC-Sim: 5-fold out-of-fold on VOC-Sim (folds crc32(id) mod 5), "
      "and the full-VOC-Sim-best threshold applied unchanged to COCO-Cap (untuned transfer). *fixed* = fixed-threshold. "
      "*audit* = uses the evaluated set's labels, never a result. BRACE's VOC-Sim numbers are in-sample for the method "
      "design (VOC-Sim was used to develop BRACE, not the comparison methods). Everything here is exploratory in the "
      "sense of AGENTS.md: no row was pre-registered beyond the Section 3 contract and Entry 3.\n")
    w("**Inputs.** N methods read the stored OVAM benchmark images (`comparisons/manifests/`, 490 VOC-Sim, 707 COCO-Cap). "
      "BRACE reads attention during generation, so it is regenerated with the corrected prompts of runs/20 "
      "(`configs/new_reruns_prompts.json`, safety checker off): *same prompts, regenerated*. Pairing is on identical "
      "image ids and identical ground-truth masks.\n")
    w("## Data checks\n")
    for ds in ("voc", "coco"):
        c = ck[ds]
        w(f"- {NAME[ds]}: {c['n']} images ({c['changed_prompt']} with corrected prompt). BRACE regeneration vs stored "
          f"official PNG: bit-identical {c['bit_identical_to_official']}/{c['n']}; mean |Δ| < 1 grey level on "
          f"{c['mean_abs_diff_lt1']}, ≥ 2 (unresolved) on {c['mean_abs_diff_ge2']}; median mean |Δ| {c['median_mean_abs_diff']:.2f}. "
          f"Ground-truth masks identical (BRACE live sample vs annotation PNG used by N): {c['truth_identical_brace_vs_annotation']}/{c['n']}; "
          f"identical across N methods: {c['truth_identical_across_n_methods']}; class labels match: {c['class_labels_match']}.")
    w("")
    w("## BRACE reproduction of runs/20 (runs/20 grid 0.30–0.70, runs/20 convention)\n")
    w("| Cell | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | VOC-Sim 0.55 | COCO-Cap 0.55 | runs/20 (same four) |")
    w("|---|---:|---:|---:|---:|---|")
    for (c, crf), v in rep.items():
        ref = REF20[(c, crf)]
        lab = {"V3": "BRACE hierarchy", "V0": "fine-only", "seed": "seed alone"}[c] + (" + DenseCRF" if crf else ", raw")
        w(f"| {lab} | {v['voc_sel']:.3f} | {v['coco_sel']:.3f} | {v['voc_055']:.3f} | {v['coco_055']:.3f} | "
          f"{ref[0]:.3f} / {ref[1]:.3f} / {ref[2]:.3f} / {ref[3]:.3f} |")
    v = rep[("V3", True)]
    w(f"\nFull-VOC-Sim-best threshold (BRACE + DenseCRF): {v['tau_full_voc']:.3f} (runs/20: 0.550). COCO-Cap best-threshold audit {v['coco_best']:.3f} (runs/20: 0.719).\n")

    for crf in (False, True):
        w(f"## Official rows, {'with DenseCRF R2' if crf else 'raw (no DenseCRF)'}\n")
        w("| Method | Rule | VOC-Sim | COCO-Cap | Note |"); w("|---|---|---:|---:|---|")
        for m, c2, rule, v, note in rows:
            if c2 == crf:
                w(f"| {SHORT[m]} | {rule} | {v['voc']:.3f} | {v['coco']:.3f} | {note} |")
        w("")
    w("## Paired Δ (method − BRACE), same images, 95% bootstrap interval\n")
    w("Primary: both at VOC-sel. Secondary: each at its fixed default (method 0.5 vs BRACE's frozen 0.55), both at 0.5, "
      "and the method's own rule vs BRACE's frozen rule. Raw and DenseCRF rows are separate comparisons.\n")
    for crf in (False, True):
        w(f"**{'With DenseCRF R2 (both sides)' if crf else 'Raw (both sides)'}**\n")
        w("| Method | Method rule | BRACE rule | VOC-Sim Δ | COCO-Cap Δ |"); w("|---|---|---|---|---|")
        for rec in J["paired"]:
            if rec["densecrf"] == crf:
                w(f"| {SHORT[rec['method']]} | {rec['method_rule']} | {rec['brace_rule']} | {fmt(rec['voc'])} | {fmt(rec['coco'])} |")
        w("")
    w("**Sensitivity (VOC-sel. both sides): images whose BRACE regeneration is within 2 / within 1 grey level (mean |Δ|) of the stored PNG**\n")
    w(f"Subset sizes: VOC-Sim {int(resolved['voc'].sum())} / {int(close1['voc'].sum())}, COCO-Cap {int(resolved['coco'].sum())} / {int(close1['coco'].sum())}.\n")
    w("| Method | CRF | VOC-Sim < 2 | COCO-Cap < 2 | VOC-Sim < 1 | COCO-Cap < 1 |"); w("|---|---|---|---|---|---|")
    for rec in J["paired"]:
        if rec["method_rule"] == "VOC-sel.":
            w(f"| {SHORT[rec['method']]} | {'R2' if rec['densecrf'] else 'raw'} | {fmt(rec['voc_resolved'])} | {fmt(rec['coco_resolved'])} | "
              f"{fmt(rec['voc_close1'])} | {fmt(rec['coco_close1'])} |")
    w("")
    w("## Runtime\n")
    w(f"BRACE regeneration + capture + hierarchy (this run, {', '.join(sorted(set(ck['voc']['brace_gpu']) | set(ck['coco']['brace_gpu'])))}): median "
      f"{ck['voc']['brace_seconds_median']:.1f} s / image VOC-Sim, {ck['coco']['brace_seconds_median']:.1f} s / image COCO-Cap; "
      f"total {(ck['voc']['brace_seconds_total'] + ck['coco']['brace_seconds_total']) / 3600:.2f} GPU-h; peak allocated "
      f"{max(ck['voc']['brace_peak_mib_max'], ck['coco']['brace_peak_mib_max']) / 1024:.1f} GiB. N methods: see Entry 4 (A100, "
      "per-item seconds in `outputs/segmentation/comparisons/n_side/<run>/per_item_counts.csv`). Different GPUs; not a cost comparison.\n")
    (HERE / "result.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
