#!/usr/bin/env python3
"""Contact sheets of the generated benchmark images (for visual selection). python scripts/cs_bench_sheet.py [start] [n]"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]; G = ROOT / "outputs/cs_bench/gen"; S = ROOT / "outputs/cs_bench/sheets"
rows = json.loads((ROOT / "runs/11_cs_bench/prompts.json").read_text())
a, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 100)
sz, cols = 384, 4
for b in range(a, a + n, 8):
    sel = [r for r in rows[b:min(b + 8, a + n)] if (G / f"{r['image_id']}.png").exists()]
    if not sel:
        continue
    sh = Image.new("RGB", (cols * sz, ((len(sel) + cols - 1) // cols) * (sz + 22)), "white"); d = ImageDraw.Draw(sh)
    for k, r in enumerate(sel):
        x, y = (k % cols) * sz, (k // cols) * (sz + 22)
        sh.paste(Image.open(G / f"{r['image_id']}.png").resize((sz, sz)), (x, y + 22))
        d.text((x + 4, y + 4), f"{r['image_id']}: " + ", ".join(r["classes"]), fill="black")
    sh.save(S / f"sheet_{b:03d}.jpg", quality=88)
print(sorted(p.name for p in S.glob("*.jpg")))
