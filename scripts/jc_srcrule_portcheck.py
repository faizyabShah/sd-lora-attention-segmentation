"""Run in the NERVE environment: NERVE's own refinement vs scripts/nerve_walk_port.py on the same capture (runs/19)."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import jc_swap_nerve as jn
import nerve_walk_port as port
from PIL import Image

class A: pass
pipe, cfg, _ = jn.build(refine=True)
for ds in ("voc", "coco"):
    for r in jn.rows_for(ds, 0, 1, 0)[:3]:
        zs = np.load(jn.SWAP / "nerve_seed" / ds / (r["key"] + ".npz"))
        pil = Image.open(r["image_local"]).convert("RGB")
        ori = (torch.tensor(np.array(pil)).permute(2, 0, 1) / 255.0)[None]
        seed = torch.from_numpy(zs["seg"]).to(pipe.device)
        with torch.no_grad():
            torch.manual_seed(0); ref = pipe.refinement(ori, seed)[0].float()
            qk = [pipe.attn_refine.qk_store[i] for i in cfg.attention_layers_to_use]
            cross = port.seed_column(seed)
            fused = port.chained_attention_product(qk, cross, d=cfg.sfsa_head_dim)
            out = port.apply_walk(port.walk_operator(fused), seed)[0].float()
        print(ds, r["key"], "max |NERVE - port|", float((ref - out).abs().max()), "map max", float(ref.abs().max()), flush=True)
