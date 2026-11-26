# Run 48: set up the VPU-lab machine for small-set COCO diagnostics (setup only, no experiment)

For the Codex on the VPU-lab machine. **Goal:** this machine can run our method on SD 1.5 and SD 2.1-base on small
COCO subsets. Experiments come later as separate `runs/<nn>_*/RUN.md` files, so **score nothing yet**.

Read `AGENTS.md` first (frozen method, label-free inference, pre-registration, compare only within one capture,
delete nothing without the owner's yes).
- **If this is the A40 already set up with `docs/NEW_MACHINE_SETUP.md`:** skip step 1 and keep its `.venv`.
- **Faiz authorises pushing** the setup report in step 6, and nothing else.

## 1. Code and Python environment

```bash
git clone git@github.com:faizyabShah/sd-lora-attention-segmentation.git && cd sd-lora-attention-segmentation
# or, in an existing clone:  git pull --rebase
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118   # pick the CUDA build matching the driver
pip install -r requirements.txt
source scripts/brace_env.sh          # sets HF_HOME and BRACE_CACHE_DIR (default outputs/sg_comp)
```

The cluster's pinned versions are in `requirements.txt`; diffusers 0.36.0 and transformers 4.44.2 matter most. If a
CUDA build doesn't fit the driver, choose another torch build, then **record it in the setup report**.

## 2. Model weights (Hugging Face; about 10 GB)

| Model | Hub id | Revision used on the cluster | Notes |
|---|---|---|---|
| SD 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` | our method (paper 2, frozen) |
| SD 2.1-base | `sd2-community/stable-diffusion-2-1-base` | `4e63672c03103b6c636b8fb4119ba982469b2955` | the SD 2.1-base mirror that iSeg and NERVE use (the `stabilityai` repo was taken down); 512 px, epsilon prediction |
| BLIP-large (optional) | `Salesforce/blip-image-captioning-large` | latest | only for caption-prompt experiments (run 45) |

```bash
source scripts/brace_env.sh
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 --revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 \
  --include "model_index.json" "scheduler/*" "tokenizer/*" "feature_extractor/*" \
            "unet/config.json" "unet/diffusion_pytorch_model.safetensors" \
            "vae/config.json" "vae/diffusion_pytorch_model.safetensors" \
            "text_encoder/config.json" "text_encoder/model.safetensors"
huggingface-cli download sd2-community/stable-diffusion-2-1-base --revision 4e63672c03103b6c636b8fb4119ba982469b2955 \
  --include "model_index.json" "scheduler/*" "tokenizer/*" "feature_extractor/*" \
            "unet/config.json" "unet/diffusion_pytorch_model.safetensors" \
            "vae/config.json" "vae/diffusion_pytorch_model.safetensors" \
            "text_encoder/config.json" "text_encoder/model.safetensors"
huggingface-cli download Salesforce/blip-image-captioning-large      # optional
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1                         # after the downloads
```

Only these fp32 safetensors files are needed (about 4.9 GB per SD model). Skip the `.ckpt`, `.bin`, fp16 and EMA files.
- **SD 2.1 code path:** `scripts/jc_sd21_47.py` loads SD 2.1 from `$SD21_MODEL` if it is set, otherwise from the hub id
  above via the `HF_HOME` cache.

## 3. COCO 2017 (official COCO server; about 2.5 GB in total)

Put everything under `datasets/coco2017/` (the layout the drivers expect). The full 18 GB train2017 zip is **not**
needed.

```bash
mkdir -p datasets/coco2017 datasets/downloads
curl -fL -C - -o datasets/downloads/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
curl -fL -C - -o datasets/downloads/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip -n datasets/downloads/annotations_trainval2017.zip -d datasets/coco2017     # -> annotations/instances_*.json, captions_*.json
unzip -n datasets/downloads/val2017.zip -d datasets/coco2017                      # -> val2017/ (5,000 photos)
```

- **Labels:** COCO-Object masks come from `annotations/instances_val2017.json` (and `instances_train2017.json` for
  profile photos), so nothing else needs downloading.
- **Captions:** `annotations/captions_train2017.json` is what the profile photos were selected with.

## 4. Shipped item lists and profiles (in git under `shipped/cache/`; 38 MB)

These are the exact cluster files: the item lists for runs 42 and 44, plus the SD 1.5 profiles (`fit.pt` = 80 COCO
words, `fit44.pt` = 33 Context words). Copy them into the cache, keeping anything that is already there:

```bash
source scripts/brace_env.sh
mkdir -p "$BRACE_CACHE_DIR" && cp -rn shipped/cache/. "$BRACE_CACHE_DIR"/
python scripts/fetch_coco_subset.py --only prof      # the 8,646 train2017 profile photos (~1.4 GB), from images.cocodataset.org
```

- **Photos:** `fetch_coco_subset.py` downloads single photos into `datasets/coco2017/train2017/` and skips files that
  already exist. `--only val` checks the COCO-Object val photos (already in val2017.zip).
- **What the item lists are:**
  - `jc_ov42/items_val.json`: 14,012 COCO-Object items on 4,752 val2017 photos (run 42 / 44 COCO column);
  - `jc_ov42/items_prof.json`: profile photos and negatives for the 80 COCO words;
  - `jc_real44/items.json`: all three run 44 columns;
  - `jc_realvoc37b/thr.json`: read when the drivers are imported.
- **Cross-GPU caution:** the SD 1.5 profiles were captured on the cluster's GPUs. Regeneration across GPUs is not
  pixel-identical. Diagnostics here can reuse them, but any paired comparison must come from captures made on this
  machine (AGENTS.md). SD 2.1 profiles don't exist yet; a later RUN.md will build them here.

## 5. Check

```bash
source scripts/brace_env.sh && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/vpu_check_setup.py          # files, then one recorded pass per backbone on one COCO val photo
```

**Expected:**
- every file `ok`;
- `profile photos (train2017): 8646 / 8646 present`;
- `COCO-Object val photos: 4752 / 4752 present`;
- one line per backbone, then `GPU check passed` and `files complete`.

On each backbone line:
- **SD 1.5:** 64×64 SA heads 8.
- **SD 2.1-base:** 64×64 SA heads 5 (SD 2.1 uses 64-dim heads; `jc_sd21_47.setup()` relaxes only the frozen 8-head
  shape check).
- **Both:** 11 gate modules, query sizes including 64, and seed (64, 64).

GPU memory: one fp32 SD pass at 512 px fits in about 8 GB. Run long jobs in GNU Screen, as on the A40.

## 6. Report back (commit and push only this file)

Write `runs/48_vpu_setup/result.md` with:
- **Hardware:** GPU model and memory; driver and CUDA version.
- **Software:** Python, torch, diffusers and transformers versions.
- **Data and paths:** disk free; `BRACE_CACHE_DIR`; `HF_HOME`.
- **Check output:** the full output of `vpu_check_setup.py`.
- **Deviations:** anything that differs from this file (other versions, a failed download, a different revision).

Then `git add runs/48_vpu_setup/result.md && git commit -m "runs/48: VPU machine setup report" && git pull --rebase && git push`.

**Do not commit** datasets, weights, `outputs/` or any cache files.
