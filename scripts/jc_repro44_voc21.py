# Reproduce each CLIP repo's published VOC21 mIoU with its own config and mmseg runner (only data_root set; optional
# model overrides). Usage: python repro_voc21.py <repo> [key=value ...]
import importlib, os, sys
repo = sys.argv[1]; R = f"/home/shafa/sd-lora-attention-segmentation/other repositories/{repo}"
os.chdir(R); sys.path.insert(0, R)
from mmengine.config import Config
from mmengine.runner import Runner
mod = {"sclip": "clip_segmentor", "clearclip": "clearclip_segmentor", "naclip": "naclip", "proxyclip": "proxyclip_segmentor"}[repo]
importlib.import_module(mod); importlib.import_module("custom_datasets")
cfg = Config.fromfile("configs/cfg_voc21.py")
cfg.test_dataloader.dataset.data_root = "/home/shafa/sd-lora-attention-segmentation/datasets/voc2012/VOCdevkit/VOC2012"
for kv in sys.argv[2:]:
    k, v = kv.split("="); cfg.model[k] = v
cfg.work_dir = f"/home/shafa/sg_comp/jc_real44/repro/{repo}_{'_'.join(sys.argv[2:]) or 'default'}"
cfg.default_hooks.pop("visualization", None); cfg.test_dataloader.num_workers = 2
cfg.launcher = "none"
print("MODEL", dict(cfg.model)); Runner.from_cfg(cfg).test()
