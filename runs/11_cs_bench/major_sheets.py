"""Sheets of the 56 images with >= 2 visible prompted major classes, for marking unprompted major classes."""
import json
from pathlib import Path
from PIL import Image, ImageDraw
R = Path(__file__).parent; ROOT = R.parents[1]; G = ROOT / "outputs/cs_bench/gen"; S = ROOT / "outputs/cs_bench/major_sheets"; S.mkdir(exist_ok=True)
MAJOR = ["road", "sidewalk", "building", "vegetation", "sky", "car", "person", "bus"]
s = json.loads((R / "selection.json").read_text())["images"]
ids = sorted(i for i in s if len([c for c in s[i]["kept"] if c in MAJOR]) >= 2); print(len(ids))
sz, cols = 448, 3
for b in range(0, len(ids), 6):
    sel = ids[b:b + 6]; sh = Image.new("RGB", (cols * sz, 2 * (sz + 20)), "white"); d = ImageDraw.Draw(sh)
    for k, i in enumerate(sel):
        x, y = (k % cols) * sz, (k // cols) * (sz + 20)
        sh.paste(Image.open(G / f"{i}.png").resize((sz, sz)), (x, y + 20)); d.text((x + 4, y + 4), f"{i} prompted: " + ", ".join(c for c in s[i]["kept"] if c in MAJOR), fill="black")
    sh.save(S / f"m_{b:03d}.jpg", quality=88)
print(ids)
