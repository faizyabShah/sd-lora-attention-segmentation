#!/usr/bin/env python3
"""Extract benchmark inputs only, retaining archives and all existing files."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import tarfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"


def write_missing(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Each destination appears once in an extraction job.
        with path.open("xb") as handle:
            handle.write(content())


def ovam():
    with ZipFile(DATA / "downloads/ovam_experiment_with_dataset.zip") as archive:
        base = "ovam_experiment_with_dataset/evaluation/"
        prefixes = tuple(base + f"ovam/{ds}/{kind}/" for ds in ("voc_sim", "coco_captions")
                         for kind in ("images", "annotations"))
        manifests = {base + "coco_captions_sampled.csv", base + "ovam/voc_sim/ovam_voc_sim.csv"}
        members = [n for n in archive.namelist() if not n.endswith("/") and
                   (n.startswith(prefixes) or n in manifests)]
        def extract(name):
            write_missing(DATA / name, lambda: archive.read(name))
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(extract, members))
        print(f"OVAM: {len(members)} benchmark files ready", flush=True)


def voc():
    archive_path = DATA / "downloads/VOCtrainval_11-May-2012.tar"
    base = "VOCdevkit/VOC2012/"
    with tarfile.open(archive_path) as archive:
        split = archive.extractfile(base + "ImageSets/Segmentation/trainval.txt")
        ids = set(split.read().decode().split())
        members = [m for m in archive.getmembers() if m.isfile() and (
            m.name.startswith((base + "SegmentationClass/", base + "SegmentationObject/",
                               base + "ImageSets/Segmentation/")) or
            (m.name.startswith(base + "JPEGImages/") and Path(m.name).stem in ids))]
    descriptor = os.open(archive_path, os.O_RDONLY)
    try:
        def extract(member):
            def read():
                content = os.pread(descriptor, member.size, member.offset_data)
                if len(content) != member.size:
                    raise IOError(f"Short read of {member.name}")
                return content
            write_missing(DATA / "voc2012" / member.name, read)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(extract, members))
    finally:
        os.close(descriptor)
    print(f"VOC: {len(ids)} segmentation images and {len(members)} total inputs ready", flush=True)


if __name__ == "__main__":
    ovam()
    voc()
