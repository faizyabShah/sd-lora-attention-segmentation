#!/usr/bin/env python3
"""SG10 seed-connected bottleneck propagation (protocol Section 77; ``configs/sg10_seed_connected_bottleneck.json``).

One live generation per image; class word ``W`` and OVAM's released token ``T``
(keys ``[<SoT>, token]``, conditional query) on the frozen CA seed ``c`` and SA
module 15 (all 31 calls, 8 heads). Per attribution:

  seed           c
  REF            mu = mean_i(A_i c), observation-wise one pass (SG10-B0)
  P1             mean_i(A_i^2 c), reported for comparison only
  {op}_{G}_q{90,95}  op in B1, B2a25, B2a50, B3; G1 = mean graph, G2 = sqrt(A A^T)
  L_{op}_q{90,95}    diagnostic only: G2 restricted to 4-neighbour edges

The mean affinity is used only for connectivity; ``mu`` is the propagated
evidence. ``--normalization-smoke`` writes bottleneck statistics only (no
segmentation score) for the registered smoke images.

    python scripts/narmyn/attention/run_sg10_bottleneck.py --dataset voc --class-shard 0
    python scripts/narmyn/attention/run_sg10_bottleneck.py --normalization-smoke --output /path/stats.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_coco_cap_transfer as transfer  # noqa: E402  (rows; its recorder is replaced below)
import scripts.run_sa_package_b as sa_common  # noqa: E402
from scripts.narmyn.attention import sg10_bottleneck as sg  # noqa: E402
from scripts.run_fusion_calibration import upsampled  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.fusion_variants import minmax  # noqa: E402
from src.segmentation.sg_visual_audit import (  # noqa: E402
    best_thresholds, released_image, render_variant_pages,
)

CONFIG = ROOT / "configs" / "sg10_seed_connected_bottleneck.json"
OUT = ROOT / "outputs" / "segmentation" / "controlled_attention_cv" / "sg10_seed_connected_bottleneck"
THRESHOLDS = transfer.THRESHOLDS
TRACKS = ("W", "T")
QUANTILES = (0.90, 0.95)
GRAPHS = ("G1", "G2", "L")
OPS = ("B1", "B2a25", "B2a50", "B3")
BINS = np.linspace(0, 1, 21)


def cell_names(track: str) -> list[str]:
    names = [f"{track}/seed", f"{track}/REF", f"{track}/P1"]
    for g in ("G1", "G2"):
        names += [f"{track}/{op}_{g}_q{int(q * 100)}" for q in QUANTILES for op in OPS]
    names += [f"{track}/L_{op}_q{int(q * 100)}" for q in QUANTILES for op in OPS]
    return names


CONFIGURATIONS = tuple(n for t in TRACKS for n in cell_names(t))
_EXTRA: dict = {"pair": None, "class_hidden": None}


class SG10Recorder(sa_common.SAPackageRecorder):
    """Frozen class-word CA (``ca_calls``) plus the released token's CA (``token_calls``)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parts = [_EXTRA["pair"]] if _EXTRA["class_hidden"] is None else [_EXTRA["class_hidden"], _EXTRA["pair"]]
        self.attribution_hidden_states = torch.cat(parts, dim=1)
        self.token_calls: dict[str, list[torch.Tensor]] = {}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        if kind != "cross":
            sa_common.SAPackageRecorder.record(self, module_name, kind, query, key, value, attention_mask, scale)
            return
        offset = 0 if _EXTRA["class_hidden"] is None else 77
        if offset:  # COCO-Cap: class keys from the separately encoded class name (Sections 48, 55)
            key, attention_mask = torch.stack((key[0], attribution_key[0][:, :77])), None
        sa_common.SAPackageRecorder.record(self, module_name, kind, query, key, value, attention_mask, scale)
        with torch.inference_mode():
            pair = attribution_key[0][:, offset:offset + 2].float()
            probs = (torch.matmul(query[1].float(), pair.transpose(-1, -2)) * scale).softmax(dim=-1)[..., 1]
            self.token_calls.setdefault(module_name, []).append(probs.half())


sa_common.live.CompactAttentionRecorder = SG10Recorder


@torch.inference_mode()
def propagate(recorder, module_name: str, seeds: torch.Tensor, device: str):
    """One and two observation-wise passes for [S, P] seeds, and the mean affinity [P, P]."""
    flat = seeds.T  # [P, S]
    one, two = torch.zeros_like(flat), torch.zeros_like(flat)
    mean = torch.zeros((flat.shape[0], flat.shape[0]), device=device)
    calls = recorder.sa_qk[module_name]
    for query, key in calls:
        q, k = query.to(device).float(), key.to(device).float()
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)  # [heads, P, P]
        first = torch.matmul(a, flat)
        one += first.mean(dim=0)
        two += torch.matmul(a, first).mean(dim=0)
        mean += a.mean(dim=0)
        del q, k, a, first
    n = len(calls)
    if n != 31 or recorder.sa_qk[module_name][0][0].shape[0] != 8:
        raise RuntimeError(f"Expected 31 calls x 8 heads, got {n} x {recorder.sa_qk[module_name][0][0].shape[0]}")
    return (one / n).T, (two / n).T, mean / n


def connectivity(mean_affinity: torch.Tensor, seeds: torch.Tensor):
    """Raw widest-path values for every graph and source set: {G: [len(TRACKS) * len(QUANTILES), P]}, sources, tops."""
    sources = torch.cat([sg.seed_sets(seeds[i], QUANTILES) for i in range(len(TRACKS))])  # [4, P]
    g2 = sg.symmetric_graph(mean_affinity)
    graphs = {"G1": mean_affinity, "G2": g2, "L": sg.local_graph(g2)}
    raw = {g: sg.widest_path(w, sources) for g, w in graphs.items()}
    tops = {g: w.max() for g, w in graphs.items()}
    return raw, sources, tops


def setup(args):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    processors = [n.removesuffix(".processor") for n in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(n for n in processors if n.endswith("attn2")))
    sa_modules = tuple(sorted(n for n in processors if n.endswith("attn1")))
    seed_spec = sa_common.resolve_seed_spec(json.loads(Path(ROOT / "configs/controlled_sa_seed_frozen.json").read_text()), ca_modules, 31)
    return pipe, ca_modules, sa_modules[transfer.SELECTED_SA_MODULE], seed_spec


def class_keys(pipe, classname: str, prompt: str, voc: bool, device: str) -> None:
    released = np.load(ROOT / "configs/ovam/tokens_VOC.npy", allow_pickle=True).item()  # committed copy (MIT)
    tok = pipe.tokenizer
    with torch.inference_mode():
        sot = pipe.text_encoder(tok("", return_tensors="pt").input_ids.to(device))[0][:, :1].float()
        token = torch.from_numpy(np.asarray(released[classname.replace(" ", "-")], dtype=np.float32)).to(device)[None, None]
        _EXTRA["pair"] = torch.cat((sot, token), dim=1)
        if voc:
            _EXTRA["class_hidden"] = None
        else:
            ids = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
            _EXTRA["class_hidden"] = pipe.text_encoder(ids.to(device))[0]


def live_maps(pipe, row, manifest_dir, ca_modules, selected, seed_spec, device):
    """Generate one image; return seeds [2, P], REF [2, P], P1 [2, P], raw bottlenecks, sources, tops, sample."""
    sa_common._RECORDERS.clear()
    sample = sa_common.live.generate_attention_sample(pipe, row, manifest_dir, "float16", cross_map_kind="conditional")
    recorder = sa_common._RECORDERS[-1]
    try:
        ca_res = {n: r for r, names in sample.modules_by_resolution.items() for n in names}
        word_calls = recorder.ca_calls
        seeds = []
        for calls in (word_calls, recorder.token_calls):
            recorder.ca_calls = calls
            seeds.append(sa_common.ca_seed(recorder, ca_modules, ca_res, seed_spec, device).reshape(-1))
        recorder.ca_calls = word_calls
        seeds = torch.stack(seeds)
        one, two, mean_affinity = propagate(recorder, selected, seeds, device)
        recorder.sa_qk.clear()
        raw, sources, tops = connectivity(mean_affinity, seeds)
        del mean_affinity
        return seeds, one, two, raw, sources, tops, sample
    finally:
        recorder.ca_calls.clear(); recorder.sa_qk.clear(); recorder.token_calls.clear()
        del recorder


def smoke_rows(spec: dict):
    for dataset, shards in (("voc", spec["normalization_smoke"]["voc_class_shards"]), ("coco", spec["normalization_smoke"]["coco_class_shards"])):
        for s in shards:
            classname, rows = (transfer.voc_rows if dataset == "voc" else transfer.coco_rows)(s)
            yield dataset, classname, rows[0]


def normalization_smoke(args, spec) -> None:
    pipe, ca_modules, selected, seed_spec = setup(args)
    with args.output.open("w") as log:
        for dataset, classname, row in smoke_rows(spec):
            class_keys(pipe, classname, row["attribution_prompt"], dataset == "voc", args.device)
            manifest_dir = transfer.EVAL / ("ovam/voc_sim" if dataset == "voc" else "ovam/coco_captions")
            seeds, one, two, raw, sources, tops, sample = live_maps(pipe, row, manifest_dir, ca_modules, selected, seed_spec, args.device)
            rec = {"dataset": dataset, "image_id": row["image_id"], "class": classname, "stats": {}}
            for g, values in raw.items():
                ratio = (values / tops[g]).cpu().numpy()
                src = sources.cpu().numpy()
                for i, (t, q) in enumerate((t, q) for t in TRACKS for q in QUANTILES):
                    v = ratio[i][~src[i]]
                    p05, p25, p50, p75, p95 = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
                    rec["stats"][f"{t}/{g}/q{int(q * 100)}"] = {
                        "top_weight": float(tops[g]), "median": float(p50), "iqr": float(p75 - p25), "p05": float(p05), "p95": float(p95),
                        "share_below_0.01": float((v < 0.01).mean()), "share_above_0.99": float((v > 0.99).mean()),
                        "distinct_values": int(np.unique(v).size)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"{row['image_id']}: " + "; ".join(f"{k} med {s['median']:.3f} iqr {s['iqr']:.3f}" for k, s in rec["stats"].items() if "/q90" in k), flush=True)
            sample.clear(); del sample; gc.collect(); torch.cuda.empty_cache()


def cells_for(t_index: int, seed, mu, p1, conn: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Maps in ``cell_names`` order for one track; ``conn[(G, q)]`` is the normalized connectivity [P]."""
    maps = [seed, mu, p1]
    for g in ("G1", "G2", "L"):
        for q in QUANTILES:
            ops = sg.operators(mu, seed, conn[(g, q)])
            maps += [ops[o] for o in OPS]
    return maps


def histogram(values: np.ndarray) -> list[int]:
    return np.histogram(np.clip(values, 0, 1), bins=BINS)[0].astype(int).tolist()


def diagnostics(seed, mu, cand_maps: dict, conn_full: dict, truth: np.ndarray) -> dict:
    """AN1-style seed -> REF additions with connectivity histograms; per-candidate change against REF; components."""
    shape = truth.shape
    seed_m = (upsampled(minmax(seed.reshape(64, 64)), shape) >= 0.45).cpu().numpy()
    ref_m = (upsampled(minmax(mu.reshape(64, 64)), shape) >= 0.45).cpu().numpy()
    new_tp, new_fp = ref_m & ~seed_m & truth, ref_m & ~seed_m & ~truth
    dist = ndimage.distance_transform_edt(~truth) if truth.any() else np.full(shape, 1e6)
    labels, n = ndimage.label(ref_m)
    touching = np.unique(labels[ref_m & truth]); touching = touching[touching > 0]
    separate = ~np.isin(labels, touching) & ref_m
    groups = {"new_tp": new_tp, "new_fp": new_fp, "new_fp_distant": new_fp & (dist > 16), "new_fp_separate": new_fp & separate}
    out = {"counts": {k: int(v.sum()) for k, v in groups.items()}, "hist": {}, "change_vs_ref": {}, "components": []}
    for key, b in conn_full.items():
        out["hist"][key] = {k: histogram(b[m]) for k, m in groups.items()}
    out["change_vs_seed"] = {"REF": {"new_tp": int(new_tp.sum()), "new_fp": int(new_fp.sum())}}
    for name, m in cand_maps.items():
        cm = (upsampled(minmax(m.reshape(64, 64)), shape) >= 0.45).cpu().numpy()
        out["change_vs_ref"][name] = {"new_tp": int((cm & ~ref_m & truth).sum()), "new_fp": int((cm & ~ref_m & ~truth).sum()),
                                      "lost_tp": int((ref_m & ~cm & truth).sum()), "removed_fp": int((ref_m & ~cm & ~truth).sum())}
        out["change_vs_seed"][name] = {"new_tp": int((cm & ~seed_m & truth).sum()), "new_fp": int((cm & ~seed_m & ~truth).sum())}
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1)) if n else []
    for lab in (np.argsort(sizes)[::-1][:50] + 1 if n else []):
        comp = labels == lab
        rec = {"size": int(comp.sum()), "tp_fraction": round(float((comp & truth).sum() / comp.sum()), 4)}
        for key, b in conn_full.items():
            rec[key] = [round(float(b[comp].max()), 4), round(float(b[comp].mean()), 4)]
        out["components"].append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("voc", "coco"))
    parser.add_argument("--class-shard", type=int, choices=range(20))
    parser.add_argument("--normalization-smoke", action="store_true")
    parser.add_argument("--output", type=Path, help="statistics JSONL for --normalization-smoke")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--visual-dir", type=Path,
                        help="Render the first image and every registered cell; do not write metric shards")
    parser.add_argument("--threshold-results", type=Path, default=OUT,
                        help="SG10 result directory used to recover full-VOC display thresholds")
    args = parser.parse_args()
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.normalization_smoke:
        normalization_smoke(args, spec)
        return
    convention = spec.get("bottleneck_normalization")
    if convention not in ("ratio", "rank"):
        raise SystemExit("bottleneck_normalization is not frozen in the config; run the normalization smoke first")
    voc = args.dataset == "voc"
    classname, rows = (transfer.voc_rows if voc else transfer.coco_rows)(args.class_shard)
    rows = rows[:1] if args.visual_dir else (rows[: args.max_samples] if args.max_samples else rows)
    manifest_dir = transfer.EVAL / ("ovam/voc_sim" if voc else "ovam/coco_captions")
    pipe, ca_modules, selected, seed_spec = setup(args)
    prompts = {r["attribution_prompt"] for r in rows}
    if len(prompts) != 1:
        raise RuntimeError(f"{classname}: expected one attribution prompt, found {sorted(prompts)}")
    class_keys(pipe, classname, prompts.pop(), voc, args.device)
    settings = {"package": "sg10_seed_connected_bottleneck_v1", "protocol": "protocol Section 77", "dataset": args.dataset,
                "bottleneck_normalization": convention, "quantiles": list(QUANTILES), "graphs": list(GRAPHS), "operators": list(OPS),
                "ca_seed": seed_spec, "selected_sa_module": selected, "thresholds": list(THRESHOLDS),
                "configurations": list(CONFIGURATIONS), "model": args.model, "model_dtype": "float32", "qk_storage_dtype": "float16"}
    folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv") if voc else {r["image_id"]: 0 for r in rows}
    stem = f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    writer = None if args.visual_dir else cv.ShardWriter(args.output_dir / args.dataset / "shards" / f"{stem}.npz", CONFIGURATIONS, folds, settings, thresholds=THRESHOLDS)
    diag_path = args.output_dir / args.dataset / "diagnostics" / f"{stem}.jsonl"
    if writer is not None:
        diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_done = {json.loads(l)["image_id"] for l in diag_path.open()} if writer is not None and diag_path.exists() else set()
    visual_thresholds = best_thresholds(args.threshold_results) if args.visual_dir else {}
    panel_ids = set(spec["panel_image_ids"])
    for row in rows:
        if writer is not None and row["image_id"] in writer.completed:
            continue
        seeds, one, two, raw, sources, tops, sample = live_maps(pipe, row, manifest_dir, ca_modules, selected, seed_spec, args.device)
        try:
            truth = np.asarray(sample.target_mask, bool)
            maps, diag = [], {"image_id": row["image_id"], "class": classname, "tracks": {}}
            for ti, t in enumerate(TRACKS):
                conn, conn_full = {}, {}
                for g in GRAPHS:
                    norm = sg.normalize_bottleneck(raw[g], sources, tops[g], convention)
                    for qi, q in enumerate(QUANTILES):
                        conn[(g, q)] = norm[ti * len(QUANTILES) + qi]
                        conn_full[f"{g}_q{int(q * 100)}"] = F.interpolate(conn[(g, q)].reshape(1, 1, 64, 64), size=truth.shape,
                                                                         mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                track_maps = cells_for(ti, seeds[ti], one[ti], two[ti], conn)
                maps += track_maps
                names = cell_names(t)
                cand = {n.split("/", 1)[1]: m for n, m in zip(names, track_maps) if n.split("/", 1)[1] not in ("seed", "REF", "P1")}
                diag["tracks"][t] = diagnostics(seeds[ti], one[ti], cand, conn_full, truth)
                if not args.visual_dir and row["image_id"] in panel_ids:
                    save_panel(args.output_dir / args.dataset / "panels" / f"{row['image_id']}_{t}.png", sample, seeds[ti], one[ti],
                               conn, cand, truth, spec["panel_config"])
            if args.visual_dir:
                ca_maps = [seeds[0] if name.startswith("W/") else seeds[1] for name in CONFIGURATIONS]
                render_variant_pages(
                    args.visual_dir, "sg10", classname, row["image_id"],
                    released_image(row, manifest_dir), truth, CONFIGURATIONS,
                    ca_maps, maps, visual_thresholds,
                )
                print(f"[visual:{classname}] {row['image_id']} cells={len(CONFIGURATIONS)}", flush=True)
                continue
            counts, ap = cv.score_maps(torch.stack(maps).reshape(-1, 64, 64), sample.target_mask, "minmax", thresholds=THRESHOLDS)
            if row["image_id"] not in diag_done:
                with diag_path.open("a") as log:
                    log.write(json.dumps(diag) + "\n")
            writer.add(row["image_id"], classname, counts, ap)
            print(f"[{classname}] {row['image_id']} ({len(writer.completed)}/{len(rows)})", flush=True)
        finally:
            sample.clear(); del sample; gc.collect(); torch.cuda.empty_cache()
    if writer is not None and args.max_samples is None:
        writer.finish()


def save_panel(path: Path, sample, seed, mu, conn, cand, truth, config: dict) -> None:
    """Fixed display configuration only (registered); never used for selection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q, g, op = config["quantile"], config["graph"], config["operator"]
    shape = truth.shape
    up = lambda m: upsampled(minmax(m.reshape(64, 64)), shape).cpu().numpy()  # noqa: E731
    ref_m, out_m = up(mu) >= 0.45, up(cand[f"{op}_{g}_q{int(q * 100)}"]) >= 0.45
    panels = [("image", sample.generated_image), ("CA seed", up(seed)), ("REF (one pass)", up(mu)),
              (f"connectivity {g} q{int(q * 100)}", F.interpolate(conn[(g, q)].reshape(1, 1, 64, 64), size=shape, mode="bilinear")[0, 0].cpu().numpy()),
              (f"SG10 {op}", up(cand[f"{op}_{g}_q{int(q * 100)}"])), ("removed FP", ref_m & ~out_m & ~truth), ("lost TP", ref_m & ~out_m & truth)]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.4))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap=None if img.ndim == 3 else "magma", vmin=0, vmax=1)
        ax.contour(truth, levels=[0.5], colors="cyan", linewidths=0.6) if img.ndim == 2 else None
        ax.set_title(title, fontsize=7); ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
