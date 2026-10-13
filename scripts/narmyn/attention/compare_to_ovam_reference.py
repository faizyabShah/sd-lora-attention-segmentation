"""Compare captured generations against the original OVAM VOC-Sim images.

capture_ovam_attention.py computes this inline, but only for samples captured
while the reference images were on disk. This recomputes it for every sample,
using the same definitions, so a run that finished before the references were
fetched can still be checked.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument("--captures", type=Path, default=Path("outputs/attention/voc_sim"))
ap.add_argument("--manifest", type=Path,
                default=Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"))
a = ap.parse_args()

rows = {r["image_id"]: r for r in csv.DictReader(a.manifest.open())}
base = a.manifest.parent
out, exact, missing = [], 0, 0

for d in sorted(a.captures.glob("ovam-voc-sim-*")):
    if not (d / "COMPLETE").exists():
        continue
    row = rows.get(d.name)
    gen = d / "generated.png"
    ref = base / row["image"] if row else None
    if row is None or not gen.exists() or ref is None or not ref.exists():
        missing += 1
        continue
    g = np.asarray(Image.open(gen).convert("RGB"), dtype=np.int16)
    r = np.asarray(Image.open(ref).convert("RGB"), dtype=np.int16)
    if g.shape != r.shape:
        out.append({"image_id": d.name, "error": f"shape {g.shape} vs {r.shape}"})
        continue
    diff = np.abs(g - r)
    is_exact = bool(np.array_equal(g, r))
    exact += is_exact
    out.append({
        "image_id": d.name,
        "classname": row["classname"],
        "seed": int(row["seed"]),
        "exact_pixels": is_exact,
        "pixel_mae": float(diff.mean()),
        "pixel_rmse": float(np.sqrt((diff.astype(np.float64) ** 2).mean())),
        "pixel_max_error": int(diff.max()),
        "identical_file": sha256(gen) == sha256(ref),
    })

print(f"compared {len(out)} samples  ({missing} skipped: no reference or incomplete)")
if out:
    ok = [o for o in out if "pixel_mae" in o]
    print(f"  bit-exact matches: {exact}/{len(ok)}")
    if ok:
        maes = sorted(o["pixel_mae"] for o in ok)
        print(f"  pixel MAE  min {maes[0]:.4f}  median {maes[len(maes)//2]:.4f}  max {maes[-1]:.4f}")
        print(f"  worst max-error: {max(o['pixel_max_error'] for o in ok)}")
    print()
    for o in out[:10]:
        if "error" in o:
            print(f"  {o['image_id']:<20} {o['error']}")
        else:
            flag = "EXACT" if o["exact_pixels"] else f"MAE {o['pixel_mae']:.4f} max {o['pixel_max_error']}"
            print(f"  {o['image_id']:<20} {o['classname']:<12} {flag}")
dst = a.captures / "ovam_reference_comparison.json"
dst.write_text(json.dumps(out, indent=2))
print(f"\nwrote {dst}")
