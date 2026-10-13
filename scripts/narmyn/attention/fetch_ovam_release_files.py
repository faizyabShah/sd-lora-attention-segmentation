#!/usr/bin/env python3
"""Fetch selected files from OVAM's public dataset zip without downloading all 14 GB.

The server supports HTTP range requests, so the zip's central directory (at the
end of the file) is read first, and then only the requested members are fetched
and inflated. By default this gets what run_token_crf.py needs: the VOC-Sim and
COCO-Cap annotations, the VOC-Sim manifest, and the sampled COCO captions
(about 3 MB). Files are written below --output-dir with the path they have
under ``ovam_experiment_with_dataset/evaluation/`` in the zip.
"""

from __future__ import annotations

import argparse
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "http://www-vpu.eps.uam.es/publications/ovam/ovam_experiment_with_dataset.zip"
DEFAULT_PATTERNS = (
    "/ovam/voc_sim/annotations/",
    "/ovam/voc_sim/ovam_voc_sim.csv",
    "/ovam/coco_captions/annotations/",
    "/coco_captions_sampled.csv",
)


def http_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def central_directory(url: str) -> list[dict]:
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        total = int(response.headers["Content-Length"])
    tail = http_range(url, total - 65536 - 22, total - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    entries, cd_size, cd_offset = struct.unpack("<HII", tail[eocd + 10 : eocd + 20])
    if 0xFFFFFFFF in (cd_size, cd_offset) or entries == 0xFFFF:
        z64 = tail.rfind(b"PK\x06\x06")
        entries, cd_size, cd_offset = struct.unpack("<QQQ", tail[z64 + 32 : z64 + 56])
    data = http_range(url, cd_offset, cd_offset + cd_size - 1)
    out, p = [], 0
    for _ in range(entries):
        (_, _, _, _, method, _, _, _, csize, usize, nlen, elen, clen, _, _, _, offset) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", data[p : p + 46]
        )
        name = data[p + 46 : p + 46 + nlen].decode("utf-8", "replace")
        extra = data[p + 46 + nlen : p + 46 + nlen + elen]
        q = 0
        while q < len(extra):  # ZIP64 extended sizes and offset
            header_id, size = struct.unpack("<HH", extra[q : q + 4])
            if header_id == 1:
                values = list(struct.unpack("<" + "Q" * (size // 8), extra[q + 4 : q + 4 + 8 * (size // 8)]))
                if usize == 0xFFFFFFFF:
                    usize = values.pop(0)
                if csize == 0xFFFFFFFF:
                    csize = values.pop(0)
                if offset == 0xFFFFFFFF:
                    offset = values.pop(0)
            q += 4 + size
        out.append({"name": name, "method": method, "csize": csize, "usize": usize, "offset": offset})
        p += 46 + nlen + elen + clen
    return out


def fetch_member(url: str, entry: dict, output_dir: Path) -> str:
    header_end = entry["offset"] + 30 + len(entry["name"].encode()) + 4096
    data = http_range(url, entry["offset"], header_end + entry["csize"])
    signature, *_, method, _, _, _, _, _, nlen, elen = struct.unpack("<IHHHHHIIIHH", data[:30])
    if signature != 0x04034B50:
        raise ValueError(f"Bad local header for {entry['name']}")
    body = data[30 + nlen + elen : 30 + nlen + elen + entry["csize"]]
    raw = body if method == 0 else zlib.decompress(body, -15)
    if len(raw) != entry["usize"]:
        raise ValueError(f"Size mismatch for {entry['name']}")
    relative = entry["name"].split("evaluation/", 1)[1]
    path = output_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/ovam_experiment_with_dataset/evaluation"))
    parser.add_argument("--pattern", action="append", help="substring of member paths to fetch (repeatable)")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    patterns = tuple(args.pattern or DEFAULT_PATTERNS)
    wanted = [
        e for e in central_directory(args.url)
        if "__MACOSX" not in e["name"] and not e["name"].endswith("/") and "evaluation/" in e["name"]
        and any(p in e["name"] for p in patterns)
    ]
    with ThreadPoolExecutor(16) as pool:
        written = list(pool.map(lambda e: fetch_member(args.url, e, args.output_dir), wanted))
    print(f"Wrote {len(written)} files below {args.output_dir}")


if __name__ == "__main__":
    main()
