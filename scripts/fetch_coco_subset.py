#!/usr/bin/env python
"""Download only the COCO 2017 photos that our item lists use (instead of the 18 GB train2017 zip).

    python scripts/fetch_coco_subset.py                 # profile photos (train2017, 80 COCO + 33 Context words) + COCO-Object val photos
    python scripts/fetch_coco_subset.py --only prof     # profile photos only
    python scripts/fetch_coco_subset.py --list my.json  # any json list of {"path": "train2017/....jpg"} items

Photos land in datasets/coco2017/<split>/<id>.jpg, the layout scripts/jc_coco40.py expects. Existing files are kept.
Source: http://images.cocodataset.org/<split>/<file> (the official COCO image server).
"""
from __future__ import annotations

import argparse, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from brace_paths import CACHE  # noqa: E402

COCO = ROOT / "datasets/coco2017"


def paths(only, extra):
    out = set()
    if extra:
        out |= {x["path"] for x in json.loads(Path(extra).read_text())}
        return sorted(out)
    if only in ("all", "prof"):
        out |= {x["path"] for x in json.loads((CACHE / "jc_ov42/items_prof.json").read_text())["items"]}
        out |= {x["path"] for x in json.loads((CACHE / "jc_real44/ours/items_prof44.json").read_text())["jobs"]}
    if only in ("all", "val"):
        out |= {x["path"] for x in json.loads((CACHE / "jc_ov42/items_val.json").read_text())}
    return sorted(out)


def get(rel):
    dst = COCO / rel
    if dst.exists() and dst.stat().st_size > 0:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(f"http://images.cocodataset.org/{rel}", tmp)
            tmp.replace(dst)
            return 1
        except Exception as e:  # noqa: BLE001  (retry transient network errors)
            if attempt == 4:
                print(f"FAILED {rel}: {e}", flush=True)
                return -1
            time.sleep(2 * (attempt + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=("all", "prof", "val"), default="all")
    p.add_argument("--list", default="")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    todo = paths(a.only, a.list)
    print(f"{len(todo)} photos requested", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        r = list(ex.map(get, todo))
    print(f"downloaded {r.count(1)}, already present {r.count(0)}, failed {r.count(-1)}")
    sys.exit(1 if -1 in r else 0)


if __name__ == "__main__":
    main()
