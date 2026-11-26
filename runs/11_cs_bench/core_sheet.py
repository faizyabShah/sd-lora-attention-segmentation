"""One overview sheet of the core50 images with their query classes."""
import json
from pathlib import Path
from PIL import Image, ImageDraw
R = Path(__file__).parent; G = R.parents[1] / "outputs/cs_bench/gen"
s = json.loads((R / "selection.json").read_text())["images"]; ids = sorted(i for i in s if s[i]["core50"])
sz, cols = 300, 5; sh = Image.new("RGB", (cols * sz, ((len(ids) + cols - 1) // cols) * (sz + 30)), "white"); d = ImageDraw.Draw(sh)
for k, i in enumerate(ids):
    x, y = (k % cols) * sz, (k // cols) * (sz + 30)
    sh.paste(Image.open(G / f"{i}.png").resize((sz, sz)), (x, y + 30)); d.text((x + 3, y + 2), i, fill="black"); d.text((x + 3, y + 15), ", ".join(s[i]["kept"]), fill="black")
sh.save(R.parents[1] / "outputs/cs_bench/core50_overview.jpg", quality=85); print(len(ids))
