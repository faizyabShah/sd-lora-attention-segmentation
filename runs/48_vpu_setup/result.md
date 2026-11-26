# Run 48 setup report — VPU lab A40

The Run 48 setup check passed on both pinned Stable Diffusion backbones. This
run verified input files and attention capture. No benchmark was scored.

- GPU: NVIDIA A40, 46068 MiB, 520.61.05. CUDA runtime used by PyTorch: 11.8.
- Python 3.10.12, PyTorch 2.6.0+cu118, Diffusers 0.36.0, Transformers 4.44.2.
- `BRACE_CACHE_DIR=/mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp`
- `HF_HOME=/mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/setup/huggingface`
- SD 1.5 revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` reused from the local cache.
- SD 2.1-base revision `4e63672c03103b6c636b8fb4119ba982469b2955` staged on local disk before copying to the repository cache.
- COCO 2017 annotations, 5,000 validation photos and the 8,646 listed training photos from the official server. Shipped SD 1.5 profiles copied into the cache without overwriting existing files.

The existing A40 environment was reused. Optional BLIP and the full COCO
training archive were outside this setup. The Run 48 checker tests the
ungated seed and fine-only propagation on both models. SD 2.1 profile fitting
is a separate experiment.

Disk space:

```text
Filesystem                    Size  Used Avail Use% Mounted on
rhome.vpu.eps.uam.es:/rhome    50T   26T   25T  52% /mnt/rhome
/dev/mapper/ubuntu--vg-lv--0  1.5T  1.4T   25G  99% /
```

Full offline check output:

```text
ok  /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_ov42/items_prof.json
ok  /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_ov42/items_val.json
ok  /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_ov42/fit.pt
ok  /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_ov42/thr.json
ok  /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_real44/items.json
ok  /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_real44/ours/items_prof44.json
ok  /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_real44/ours/fit44.pt
ok  /mnt/rhome/fsh/sd-lora-attention-segmentation/outputs/sg_comp/jc_realvoc37b/thr.json
ok  /mnt/rhome/fsh/sd-lora-attention-segmentation/datasets/coco2017/annotations/instances_val2017.json
ok  /mnt/rhome/fsh/sd-lora-attention-segmentation/datasets/coco2017/annotations/instances_train2017.json
ok  /mnt/rhome/fsh/sd-lora-attention-segmentation/datasets/coco2017/annotations/captions_train2017.json
profile photos (train2017): 8646 / 8646 present
COCO-Object val photos: 4752 / 4752 present
GPU: NVIDIA A40; photo val2017/000000007386.jpg word 'dog'
Couldn't connect to the Hub: Cannot reach https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-v1-5: offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable..
Will try to load from local cache.

Loading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s]
Loading pipeline components...:  33%|███▎      | 2/6 [00:00<00:00, 10.41it/s]/home/fsh/.venvs/brace/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(

Loading pipeline components...:  67%|██████▋   | 4/6 [00:01<00:00,  3.24it/s]
Loading pipeline components...:  83%|████████▎ | 5/6 [00:02<00:00,  2.00it/s]
Loading pipeline components...: 100%|██████████| 6/6 [00:02<00:00,  1.99it/s]
Loading pipeline components...: 100%|██████████| 6/6 [00:02<00:00,  2.33it/s]
/mnt/rhome/fsh/sd-lora-attention-segmentation/scripts/jc_realvoc.py:161: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
  x = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1)[None].float().to(DEV) / 127.5 - 1
SD 1.5: 11 gate modules, query sizes [8, 16, 64], 64x64 SA heads 8, seed (64, 64), V0 finite True
Couldn't connect to the Hub: Cannot reach https://huggingface.co/api/models/sd2-community/stable-diffusion-2-1-base: offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable..
Will try to load from local cache.

Loading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s]
Loading pipeline components...:  33%|███▎      | 2/6 [00:00<00:00, 11.84it/s]
Loading pipeline components...:  67%|██████▋   | 4/6 [00:00<00:00, 10.57it/s]
Loading pipeline components...: 100%|██████████| 6/6 [00:01<00:00,  2.59it/s]
Loading pipeline components...: 100%|██████████| 6/6 [00:01<00:00,  3.27it/s]
SD 2.1-base: 11 gate modules, query sizes [8, 16, 64], 64x64 SA heads 5, seed (64, 64), V0 finite True
GPU check passed
files complete
```
