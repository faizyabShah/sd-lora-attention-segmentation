"""Regenerate a few already-made indices on this node, for cross-GPU comparison.

Writes to output/hwcheck/<node>_<idx>.png so the originals are untouched.
"""
import json, socket, sys
from pathlib import Path
import os
import torch
from diffusers import StableDiffusionPipeline

# Data and outputs live under $TRDP_ROOT when set (the pipeline working
# tree on the cluster); otherwise they resolve relative to this repo.
ROOT = Path(os.environ.get("TRDP_ROOT",
                                   Path(__file__).resolve().parents[2]))
rows = [json.loads(l) for l in open(ROOT/"data/florence2_detailedcaptions/metadata.jsonl") if l.strip()]
out = ROOT/"output/hwcheck"; out.mkdir(parents=True, exist_ok=True)
node = socket.gethostname().split(".")[0]
idxs = [int(x) for x in sys.argv[1:]] or [100, 101, 102, 1000, 3000]

print(f"node={node} torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}", flush=True)
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16,
    safety_checker=None, feature_extractor=None, requires_safety_checker=False).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.load_lora_weights(str(ROOT/"output/lora_weights/rank8"),
                       weight_name="pytorch_lora_weights.safetensors")
for i in idxs:
    g = torch.Generator("cuda").manual_seed(i)
    img = pipe(rows[i]["text"], generator=g, num_inference_steps=30, height=768, width=768).images[0]
    img.save(out/f"{node}_{i:05d}.png")
    print(f"  wrote {node}_{i:05d}.png", flush=True)
