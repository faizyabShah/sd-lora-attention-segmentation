#!/usr/bin/env python3
"""Dev check on D (2026-11-09): does the C16 propagation depend on which prompt's SA is used?
For every D photo with >=2 annotated classes and every class k, propagate the SCLIP VOC21 seed of k
(mm, as the frozen clip C16) through the SA of the run for class k (own) and of every other class j (cross),
threshold at the frozen tau 0.575 and score IoU vs the class mask at full size. Label-free maps; masks only score."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.segmentation.sg25_sa import _attention, _apply, _resize
VOC = ROOT / "datasets/voc2012/VOCdevkit/VOC2012"
dev = "cuda"

def c16_multi(sa, P):          # P [C,64,64] -> [C,64,64]; identical to sg27_fusion.base_paths C16 per channel
    C = P.shape[0]; c16 = _resize(P.reshape(C, -1), 64, 16)          # [C,256]
    out = torch.zeros(C, 4096, device=dev)
    for call in range(11):
        a16 = _attention(sa[16][call], 16); a64 = _attention(sa[64][call], 64)
        co = torch.einsum("hij,cj->hci", a16, c16).mean(0)           # [C,256]
        up = _resize(co, 16, 64)                                      # [C,4096]
        out += torch.einsum("hij,cj->ci", a64, up)
        del a16, a64
    return (out / 88).reshape(C, 64, 64)

def load_sa(iid, k):
    q = np.load(f"/home/shafa/sg_real_atlas/D/{iid}/{k}/qk_ddim.npz")
    return {s: [(torch.from_numpy(q[f"s{s}"][c, 0]).to(dev), torch.from_numpy(q[f"s{s}"][c, 1]).to(dev)) for c in range(11)]
            for s in (16, 64)}

mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
man = json.loads((ROOT / "outputs/segmentation/sg_real_atlas/manifest.json").read_text())["D"]
rows = []; check = []
for e in man:
    ks = e["classes"]
    if len(ks) < 2: continue
    iid = e["image"]; lab = np.asarray(Image.open(VOC / f"SegmentationClass/{iid}.png")); H, W = lab.shape
    P = torch.from_numpy(np.load(f"/home/shafa/sg_comp/clip_D/{iid}__B16.npz")["prob21_64"]).float().to(dev)
    seeds = torch.stack([mm(P[k]) for k in ks])
    for j in ks:
        sa = load_sa(iid, j); M = c16_multi(sa, seeds)
        if j == ks[0]:   # replay vs the frozen per-seed implementation
            from src.segmentation import sg27_fusion
            sa_full = dict(sa); q = np.load(f"/home/shafa/sg_real_atlas/D/{iid}/{j}/qk_ddim.npz")
            sa_full[32] = [(torch.from_numpy(q["s32"][c, 0]).to(dev), torch.from_numpy(q["s32"][c, 1]).to(dev)) for c in range(11)]
            ref = sg27_fusion.base_paths(sa_full, seeds[0])["C16"].reshape(64, 64)
            check.append(float((ref - M[0]).abs().max() / ref.abs().max()))
        for i, k in enumerate(ks):
            m = F.interpolate(mm(M[i])[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy() >= 0.575
            valid = lab != 255; t = lab == k
            iou = (m & t & valid).sum() / max(((m | t) & valid).sum(), 1)
            rows.append((iid, k, j, int(j == k), float(iou)))
    print(iid, flush=True)
np.save(ROOT / "outputs/segmentation/sg_real_atlas/analysis/sa_prompt_dep_D.npy", np.array(rows, dtype=object))
own = {(r[0], r[1]): r[4] for r in rows if r[3]}
cross = {}
for r in rows:
    if not r[3]: cross.setdefault((r[0], r[1]), []).append(r[4])
keys = sorted(own); cls = sorted({k for _, k in keys})
mac = lambda d: np.mean([np.mean([d[x] for x in keys if x[1] == c]) for c in cls])
cm = {x: np.mean(cross[x]) for x in keys}
d = np.array([cm[x] - own[x] for x in keys])
print(f"replay max rel err {max(check):.2e}")
print(f"pairs {len(keys)}  class-macro own {mac(own):.4f}  cross {mac(cm):.4f}  mean pair delta {d.mean():+.4f}  |delta| p50 {np.median(abs(d)):.4f} p90 {np.quantile(abs(d),.9):.4f}")
