"""Records Claude's visual review (2026-11-16) of all 100 generated images; no BRACE map exists. Writes selection.json.
keep[id] = named classes clearly visible and outlinable; images with < 2 are rejected. Notes give drops/reasons.
core50: deterministic class-balancing subset (greedy: repeatedly take the image whose kept classes are currently
least covered, ties by id) - label-free and BRACE-free."""
import json
from collections import Counter
from pathlib import Path
R = Path(__file__).parent
rows = {r["image_id"]: r for r in json.loads((R / "prompts.json").read_text())}
K = {  # id: (kept classes, note)
 "cs000": ("traffic light,sky,sidewalk", "motorcycle absent"),
 "cs001": ("pole,train,building", "terrain absent"),
 "cs002": ("rider,road,vegetation", "person ambiguous with riders"),
 "cs003": ("", "bus/bicycle absent; wall/fence unclear"),
 "cs004": ("car,sidewalk,road", "truck absent"),
 "cs005": ("pole,sky", "traffic sign too small; terrain absent"),
 "cs006": ("person,bicycle,vegetation,building", ""),
 "cs007": ("traffic light,train", "fence/wall absent"),
 "cs008": ("", "only truck clear; motorcycle absent; terrain/sky tiny"),
 "cs009": ("car,vegetation", "rider tiny/ambiguous; wall absent"),
 "cs010": ("bus,road,sidewalk", "traffic sign too small"),
 "cs011": ("person,traffic light,building", "fence absent"),
 "cs012": ("traffic sign,pole,vegetation", "fence unclear"),
 "cs013": ("rider,motorcycle,sidewalk,building", ""),
 "cs014": ("truck,wall,terrain", "car absent"),
 "cs015": ("road,sky", "vehicle ambiguous bus vs tram: both dropped"),
 "cs016": ("bicycle,road", "car absent; wall absent"),
 "cs017": ("person,sidewalk", "motorcycle/fence absent"),
 "cs018": ("traffic light,sky,building", "traffic sign too small"),
 "cs019": ("rider,bicycle,vegetation", "terrain unclear"),
 "cs020": ("truck,bus,sidewalk,vegetation", ""),
 "cs021": ("train,pole,road", "terrain absent"),
 "cs022": ("person,building,sky", "pole too small"),
 "cs023": ("", "only car clear; rider tiny; wall/fence absent"),
 "cs024": ("bicycle,building,sky", "motorcycle ambiguous"),
 "cs025": ("", "vehicle ambiguous truck vs tram; fence/wall absent"),
 "cs026": ("bus,traffic sign,road,vegetation", ""),
 "cs027": ("car,sidewalk", "traffic lights too small; terrain absent"),
 "cs028": ("person,car,vegetation,building", ""),
 "cs029": ("train,sky", "bicycle/wall absent"),
 "cs030": ("rider,traffic sign,sidewalk,road", ""),
 "cs031": ("pole,truck,fence,terrain", ""),
 "cs032": ("bus,sky,sidewalk", "motorcycle absent"),
 "cs033": ("train,vegetation", "motorcycle/terrain absent"),
 "cs034": ("person,building", "traffic sign too small; wall absent"),
 "cs035": ("rider,pole,road", "fence absent"),
 "cs036": ("bus,sidewalk", "truck ambiguous (second bus); wall absent"),
 "cs037": ("car,vegetation,building", "bicycle too small"),
 "cs038": ("traffic light,train,road", "fence absent"),
 "cs039": ("rider,sky", "person ambiguous with riders; terrain absent"),
 "cs040": ("bicycle,building", "motorcycle/terrain absent"),
 "cs041": ("bus,sidewalk", "truck/wall absent"),
 "cs042": ("car,road", "traffic sign too small; fence absent"),
 "cs043": ("traffic light,pole,vegetation,sky", ""),
 "cs044": ("traffic light,train,vegetation", "fence unclear"),
 "cs045": ("bicycle,rider,building,road", ""),
 "cs046": ("", "motorcycle/terrain/wall absent; traffic sign too small"),
 "cs047": ("car,sidewalk,sky", "bus absent"),
 "cs048": ("person,pole,vegetation,building", ""),
 "cs049": ("", "only truck clear; motorcycle absent; terrain/sky tiny"),
 "cs050": ("bus,sidewalk", "truck/wall absent"),
 "cs051": ("train,road", "bicycle/fence absent"),
 "cs052": ("traffic light,sidewalk,road", "rider absent (parked bicycles only)"),
 "cs053": ("person,pole,vegetation,building", ""),
 "cs054": ("car,sky", "traffic sign too small; terrain absent"),
 "cs055": ("", "only train clear; truck/fence/wall absent"),
 "cs056": ("", "only road clear; traffic light/sign too small; wall absent"),
 "cs057": ("car,sidewalk,building", "pole too small"),
 "cs058": ("motorcycle,sky", "bicycle too small; fence absent"),
 "cs059": ("rider,vegetation", "person ambiguous with riders; terrain absent"),
 "cs060": ("bus,vegetation", "rider/fence absent"),
 "cs061": ("", "truck/wall absent; traffic sign too small"),
 "cs062": ("bicycle,traffic light,sky,building", ""),
 "cs063": ("person,car,road,terrain", ""),
 "cs064": ("", "motorcycle tiny; pole small; wall/fence absent"),
 "cs065": ("", "vehicles ambiguous bus vs tram; terrain absent; only sky clear"),
 "cs066": ("rider,building,road", "person ambiguous with riders"),
 "cs067": ("bicycle,traffic sign,vegetation", "sidewalk unclear"),
 "cs068": ("car,fence,sky", "traffic light too small"),
 "cs069": ("bus,road,vegetation", "motorcycle absent"),
 "cs070": ("train,building", "pole too small; sidewalk unclear"),
 "cs071": ("", "only truck clear; traffic light small; wall/terrain absent"),
 "cs072": ("car,sky,vegetation", "pole too small"),
 "cs073": ("train,road", "rider absent; fence small"),
 "cs074": ("truck,person,sidewalk,terrain", ""),
 "cs075": ("bus,motorcycle,building", "wall absent"),
 "cs076": ("bicycle,building,vegetation", "traffic sign absent"),
 "cs077": ("person,sky", "traffic light/wall absent"),
 "cs078": ("truck,sidewalk", "rider/terrain absent"),
 "cs079": ("car,traffic sign,road", "fence too small"),
 "cs080": ("bicycle,sidewalk,building", "pole absent"),
 "cs081": ("vegetation,sky", "train too small; motorcycle absent"),
 "cs082": ("bus,person,road", "fence absent"),
 "cs083": ("traffic sign,bicycle", "wall/terrain absent"),
 "cs084": ("", "vehicle ambiguous van vs truck; motorcycle distorted; wall/terrain absent"),
 "cs085": ("rider,traffic light,building,sky", ""),
 "cs086": ("pole,bus,road", "fence small"),
 "cs087": ("train,vegetation,sidewalk", "car absent"),
 "cs088": ("", "only vegetation clear; pole/motorcycle/terrain absent"),
 "cs089": ("", "black image (pipeline safety filter)"),
 "cs090": ("bus,road,building", "traffic light too small"),
 "cs091": ("car,truck", "wall/sidewalk absent"),
 "cs092": ("rider,person,building", "fence absent"),
 "cs093": ("train,pole,sky", "wall absent"),
 "cs094": ("bus,road,sidewalk", "rider distorted (artefact)"),
 "cs095": ("truck,vegetation", "traffic sign too small; terrain absent"),
 "cs096": ("traffic light,car,road", "fence absent"),
 "cs097": ("bicycle,sky,vegetation", "person ambiguous with riders"),
 "cs098": ("", "train too small; motorcycle/terrain absent; only building clear"),
 "cs099": ("motorcycle,sidewalk", "pole/wall absent"),
}
assert set(K) == set(rows)
sel = {}
for i, (k, note) in K.items():
    kept = [c for c in k.split(",") if c]
    assert all(c in rows[i]["classes"] for c in kept), i
    sel[i] = {"kept": kept, "dropped": [c for c in rows[i]["classes"] if c not in kept], "note": note, "eligible": len(kept) >= 2}
el = sorted(i for i in sel if sel[i]["eligible"])
cov, core = Counter(), []
while len(core) < 50:
    best = min((i for i in el if i not in core), key=lambda i: (min(cov[c] for c in sel[i]["kept"]), i))
    core.append(best); cov.update(sel[best]["kept"])
for i in sel:
    sel[i]["core50"] = i in core
allc = Counter(c for i in el for c in sel[i]["kept"])
out = {"eligible": len(el), "rejected": 100 - len(el), "queries_eligible": sum(allc.values()), "class_counts_eligible": dict(allc.most_common()),
       "core50_queries": sum(cov.values()), "class_counts_core50": dict(cov.most_common()), "images": sel}
(R / "selection.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "images"})
