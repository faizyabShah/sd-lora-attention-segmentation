"""Shared live A3/SG25 capture setup for SG27 smoke and production shards."""
from __future__ import annotations

import gc
import json
from contextlib import contextmanager
from pathlib import Path

from scripts import run_coco_cap_transfer as transfer
from scripts import run_sg18_temporal_windowed as sg18
from src.attention_capture import live
from src.segmentation import sg24_ca, sg25_sa, sg27_fusion
from src.segmentation.attention_aggregation import find_token_subsequence

ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/"configs/sg27.json"


def make_runtime(model:str, recorder_class=None):
    spec=json.loads(SPEC.read_text())
    sg24=json.loads((ROOT/"configs/sg24.json").read_text())
    sg25=json.loads((ROOT/"configs/sg25.json").read_text())
    frozen=json.loads((ROOT/"configs/controlled_sa_seed_frozen.json").read_text())
    if not frozen["frozen"] or frozen["modules"]!=sg24["selected_ca_modules"]:
        raise ValueError("SG27 A3 CA observation set changed")
    if spec["thresholds"]!=list(transfer.THRESHOLDS):
        raise ValueError("SG27 threshold grid changed")
    pipe,ca_modules,selected_sa,_=sg18.make_pipe(model,"cuda",json.loads((ROOT/"configs/sg12_crf_sg14_sg15.json").read_text()),frozen)
    modules={int(side):name for side,name in sg25["sa_modules"].items()}
    if selected_sa!=modules[64]:
        raise ValueError("SG27 fine module changed")
    available={n.removesuffix(".processor") for n in pipe.unet.attn_processors if n.endswith(".attn1.processor")}
    if set(modules.values())-available:
        raise ValueError("SG27 coarse/fine module missing")
    recorder_class=recorder_class or sg25_sa.SG25Recorder
    recorder_class.pipe=pipe
    recorder_class.ca_modules=tuple(ca_modules[i] for i in frozen["modules"])
    recorder_class.selected_modules=recorder_class.ca_modules
    recorder_class.selected_sa=selected_sa
    recorder_class.extra_sa=(modules[16],modules[32])
    recorder_class.experiment="a"
    recorder_class.enabled=("A3",)
    recorders=[]
    original_init=recorder_class.__init__
    def tracked_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        recorders.append(self)
    recorder_class.__init__=tracked_init
    live.CompactAttentionRecorder=recorder_class
    return {"pipe":pipe,"modules":modules,"ca_modules":recorder_class.ca_modules,
            "recorders":recorders,"model":model,"spec":spec,"recorder_class":recorder_class}


@contextmanager
def live_maps(runtime,row,dataset):
    pipe=runtime["pipe"]
    prompt=row["attribution_prompt"]
    ids=pipe.tokenizer(prompt,padding="max_length",max_length=77,truncation=True)["input_ids"]
    class_ids=pipe.tokenizer(row["classname"],add_special_tokens=False)["input_ids"]
    target=find_token_subsequence(ids,class_ids)
    runtime["recorder_class"].content=sg24_ca.content_positions(pipe.tokenizer,prompt,tuple(target))
    runtime["recorder_class"].attribution_prompt=prompt
    recorders=runtime["recorders"]
    recorders.clear()
    manifest=transfer.EVAL/("ovam/voc_sim" if dataset=="voc" else "ovam/coco_captions")
    sample=live.generate_attention_sample(pipe,row,manifest,"float16",cross_map_kind="conditional")
    recorder=recorders[-1]
    try:
        seed=recorder.seeds()["selected/A3"]
        modules=runtime["modules"]
        sa={16:recorder.multiscale_qk[modules[16]],32:recorder.multiscale_qk[modules[32]],64:recorder.sa_qk}
        base=sg27_fusion.base_paths(sa,seed)
        yield sample,sa,seed,base,sg27_fusion.proposed_maps(base)
    finally:
        recorder.clear();sample.clear();recorders.clear()
        del recorder,sample
        gc.collect()
        import torch
        torch.cuda.empty_cache()
