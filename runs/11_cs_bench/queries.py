"""Query list for BRACE pre-annotation of the 56 major-class images (8 major Cityscapes classes).
prompted  = major classes named in the prompt and judged clearly visible (selection.json);
unprompted = major classes NOT named in the prompt but judged clearly visible (Claude, visual review 2026-11-16,
before any BRACE output). Ambiguous person-vs-rider cases are not queried as person."""
import json
from pathlib import Path
R = Path(__file__).parent
MAJOR = ["road", "sidewalk", "building", "vegetation", "sky", "car", "person", "bus"]
WORD = {"road": "road", "sidewalk": "sidewalk", "building": "buildings", "vegetation": "trees", "sky": "sky",
        "car": "cars", "person": "people", "bus": "bus"}
UNPROMPTED = {
 "cs000": "road,building,vegetation,car", "cs002": "sky,building", "cs004": "building,sky", "cs006": "road,sky",
 "cs009": "road,sky,building", "cs010": "building,sky", "cs011": "road,sidewalk,vegetation,sky", "cs013": "road,car",
 "cs015": "building,vegetation", "cs017": "road,building,car,vegetation", "cs018": "road,car", "cs020": "road,building",
 "cs022": "road", "cs024": "road,car", "cs026": "car,sidewalk", "cs027": "road,building,vegetation,sky",
 "cs028": "road,sky", "cs030": "building,sky", "cs032": "road,car,building,vegetation", "cs034": "road,car,sky",
 "cs036": "road,vegetation,building", "cs037": "road,sky", "cs041": "road,building,vegetation,sky",
 "cs042": "building,sidewalk,sky", "cs043": "road,car", "cs045": "vegetation,sky,car", "cs047": "road,building,vegetation",
 "cs048": "road,sky", "cs050": "road,building,sky", "cs052": "building", "cs053": "sky", "cs054": "road,building,sidewalk",
 "cs057": "road,vegetation,sky", "cs060": "road,building", "cs062": "road,car", "cs063": "building,sky", "cs066": "sky",
 "cs068": "road,building,vegetation", "cs069": "car,sky", "cs072": "road,building", "cs074": "building,vegetation",
 "cs075": "road,vegetation,car,sky", "cs076": "road,sky", "cs077": "road,building,vegetation,car,sidewalk",
 "cs079": "vegetation", "cs080": "road,sky", "cs081": "road,building", "cs082": "building,sidewalk", "cs085": "road,car",
 "cs086": "building,car,sky,vegetation", "cs087": "road,building,sky", "cs090": "vegetation,sidewalk", "cs092": "vegetation",
 "cs094": "building,sky", "cs096": "building,sky,sidewalk", "cs097": "road,sidewalk",
}


def queries():
    sel = json.loads((R / "selection.json").read_text())["images"]
    prompts = {r["image_id"]: r for r in json.loads((R / "prompts.json").read_text())}
    ids = sorted(i for i in sel if len([c for c in sel[i]["kept"] if c in MAJOR]) >= 2)
    assert set(ids) == set(UNPROMPTED), set(ids) ^ set(UNPROMPTED)
    out = []
    for i in ids:
        p = prompts[i]; pm = [c for c in sel[i]["kept"] if c in MAJOR]; up = UNPROMPTED[i].split(",")
        assert not set(pm) & set(up) and all(c in MAJOR for c in up), i
        assert all(c not in p["classes"] for c in up), i          # truly unprompted
        for c in pm:
            assert WORD[c] in p["prompt"], (i, c)
            out.append(dict(image_id=i, cls=c, word=WORD[c], kind="prompted", prompt=p["prompt"], seed=p["seed"]))
        for c in up:
            out.append(dict(image_id=i, cls=c, word=WORD[c], kind="unprompted", prompt=p["prompt"], seed=p["seed"]))
    return out


if __name__ == "__main__":
    from collections import Counter
    q = queries(); print(len(q), Counter(x["kind"] for x in q), Counter((x["cls"], x["kind"]) for x in q).most_common())
