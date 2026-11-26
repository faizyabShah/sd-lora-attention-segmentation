"""Per-image overlay strips of BRACE masks (P = prompted, U = unprompted) + one colour-coded combined view."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
R = Path(__file__).parent; ROOT = R.parents[1]; sys.path.insert(0, str(R)); from queries import queries, MAJOR
G = ROOT / "outputs/cs_bench/gen"; B = ROOT / "outputs/cs_bench/brace"; O = ROOT / "outputs/cs_bench/overlays"; O.mkdir(exist_ok=True)
COL = {"road": (128, 64, 128), "sidewalk": (244, 35, 232), "building": (70, 70, 70), "vegetation": (107, 142, 35),
       "sky": (70, 130, 180), "car": (0, 0, 142), "person": (220, 20, 60), "bus": (0, 60, 100)}   # Cityscapes palette
by = {}
for q in queries():
    by.setdefault(q["image_id"], []).append(q)
sz = 256; n = 0
for i, qs in by.items():
    img = np.asarray(Image.open(G / f"{i}.png").convert("RGB"), np.float32)
    tiles = [("image", img)]; comb = img.copy() * 0.45
    for q in sorted(qs, key=lambda q: (q["kind"], q["cls"])):
        f = B / f"{i}__{q['cls']}.npz"
        if not f.exists():
            continue
        m = np.load(f)["mask"]; c = np.array(COL[q["cls"]], np.float32)
        ov = img.copy(); ov[m] = 0.45 * ov[m] + 0.55 * c
        tiles.append((f"{q['cls']} ({'P' if q['kind'] == 'prompted' else 'U'}) {m.mean():.0%}", ov)); comb[m] += 0.55 * c / 1.0
    tiles.insert(1, ("all overlaid", np.clip(comb, 0, 255)))
    sh = Image.new("RGB", (len(tiles) * sz, sz + 18), "white"); d = ImageDraw.Draw(sh)
    for k, (t, a) in enumerate(tiles):
        sh.paste(Image.fromarray(a.astype(np.uint8)).resize((sz, sz)), (k * sz, 18)); d.text((k * sz + 3, 3), (i + " " if k == 0 else "") + t, fill="black")
    sh.save(O / f"{i}.jpg", quality=88); n += 1
print(n, "overlay strips")
