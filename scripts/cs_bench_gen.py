#!/usr/bin/env python3
"""Small generated Cityscapes-style benchmark (runs/11_cs_bench/README.md).
SD1.5 + Cityscapes LoRA (rank 128, training seed 2, final step 5000), BRACE's live generation settings exactly
(float32 pipe from sg27_runtime, global seeding, CUDA initial noise, 512 px, 30 PNDM steps, CFG 7.5), so a later
BRACE capture with the same LoRA on the same GPU type regenerates the same image.

python scripts/cs_bench_gen.py prompts          -> runs/11_cs_bench/prompts.json (deterministic)
python scripts/cs_bench_gen.py run [--limit k]  GPU -> outputs/cs_bench/gen/<id>.png + manifest.json
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
RUN = ROOT / "runs/11_cs_bench"; OUT = ROOT / "outputs/cs_bench/gen"
LORA = ROOT / "outputs/lora/rank_128/seed_2"          # pytorch_lora_weights.safetensors = final step 5000
N, SEED0 = 100, 30000

# Cityscapes class -> the word used in the prompt (and as the query word)
STUFF = {"road": "the road", "sidewalk": "a sidewalk", "building": "buildings", "vegetation": "trees",
         "sky": "the sky", "terrain": "grass", "wall": "a wall", "fence": "a fence"}
THINGS = {"car": "cars", "person": "people", "bus": "a bus", "truck": "a truck", "train": "a tram",
          "motorcycle": "a motorcycle", "bicycle": "bicycles", "rider": "a cyclist", "pole": "poles",
          "traffic light": "traffic lights", "traffic sign": "traffic signs"}
SCENES = ["city street", "busy intersection", "residential street", "wide boulevard", "street in the city center",
          "quiet street in a small town"]
QWORD = {"the road": "road", "a sidewalk": "sidewalk", "buildings": "buildings", "trees": "trees", "the sky": "sky",
         "grass": "grass", "a wall": "wall", "a fence": "fence", "cars": "cars", "people": "people", "a bus": "bus",
         "a truck": "truck", "a tram": "tram", "a motorcycle": "motorcycle", "bicycles": "bicycles",
         "a cyclist": "cyclist", "poles": "poles", "traffic lights": "traffic lights", "traffic signs": "traffic signs"}


def cycle(keys, n, rng):
    out = []
    while len(out) < n:
        k = list(keys); rng.shuffle(k); out += k
    return out[:n]


def prompts(_a):
    rng = random.Random(0)
    st, th = cycle(STUFF, 2 * N, rng), cycle(THINGS, 2 * N, rng)
    rows = []
    for i in range(N):
        s = [st[2 * i], st[2 * i + 1]]; t = [th[2 * i], th[2 * i + 1]]
        if s[0] == s[1]:
            s[1] = next(k for k in STUFF if k not in s)
        if t[0] == t[1]:
            t[1] = next(k for k in THINGS if k not in t)
        sky = "sky" in s; ss = [STUFF[k] for k in s if k != "sky"]
        p = f"The image shows a {SCENES[i % len(SCENES)]} with {THINGS[t[0]]} and {THINGS[t[1]]}"
        p += (f", next to {ss[0]} and {ss[1]}." if len(ss) == 2 else f", next to {ss[0]}.")
        if sky:
            p += " The sky is visible above."
        classes = {c: QWORD[(STUFF | THINGS)[c]] for c in t + s}
        assert all(w in p for w in classes.values()), p
        rows.append({"image_id": f"cs{i:03d}", "prompt": p, "seed": SEED0 + i, "classes": classes})
    RUN.mkdir(parents=True, exist_ok=True); (RUN / "prompts.json").write_text(json.dumps(rows, indent=1))
    from collections import Counter
    print(Counter(c for r in rows for c in r["classes"]).most_common()); print(rows[0]["prompt"]); print(rows[1]["prompt"])


def run(a):
    import numpy as np, torch
    from PIL import Image
    from scripts import sg27_runtime
    from src.attention_capture.live import _seed_everything
    import idea1_knockout as k1
    rows = json.loads((RUN / "prompts.json").read_text())[: a.limit or None]
    pipe = sg27_runtime.make_runtime(k1.MODEL)["pipe"]
    pipe.load_lora_weights(str(LORA), weight_name="pytorch_lora_weights.safetensors"); OUT.mkdir(parents=True, exist_ok=True)
    meta = {"lora": str(LORA.relative_to(ROOT)), "gpu": torch.cuda.get_device_name(0), "dtype": str(pipe.unet.dtype),
            "steps": 30, "guidance": 7.5, "scheduler": type(pipe.scheduler).__name__, "size": 512}
    for r in rows:
        o = OUT / f"{r['image_id']}.png"
        if o.exists():
            continue
        t0 = time.perf_counter(); _seed_everything(r["seed"])
        with torch.inference_mode():
            pe, ne = pipe.encode_prompt(prompt=r["prompt"], device=pipe.device, num_images_per_prompt=1,
                                        do_classifier_free_guidance=True, negative_prompt=None)
            lat = torch.randn((1, pipe.unet.config.in_channels, 64, 64), device=pipe.device, dtype=pe.dtype)
            im = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, latents=lat, height=512, width=512,
                      num_inference_steps=30, guidance_scale=7.5, output_type="pil").images[0]
        im.convert("RGB").save(o)
        print(f"[cs] {r['image_id']} {time.perf_counter() - t0:.1f}s", flush=True)
    (OUT / "manifest.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts"); p = sub.add_parser("run"); p.add_argument("--limit", type=int)
    a = ap.parse_args(); {"prompts": prompts, "run": run}[a.cmd](a)
