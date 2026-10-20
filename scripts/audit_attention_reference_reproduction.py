#!/usr/bin/env python3
"""Small GPU audit of failed SG11/SG12 and SG2+CRF historical-mask gates.

Selects discrepant samples for diagnosis, never method/threshold selection.
Stores only 64x64 score vectors, confusion counts, hashes and numerical errors.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_sg11_sg12_observation_reliability import WordObservationRecorder,transfer,common
from scripts.narmyn.attention import sg2_consensus_edges as legacy_sg2
from src.segmentation import controlled_cv as cv
from src.segmentation.observation_reliability import propagate_qk,reliability_maps,configuration_names
from src.segmentation.consensus_graph import locked_sg2_maps

CASES=(('voc',17,'ovam-voc-sim-441',False),('voc',19,'ovam-voc-sim-482',True),
       ('coco',14,'ovam-coco-472976-person-4017470167',True),
       ('coco',14,'ovam-coco-117816-person-2575768907',False))


def original_counts(family,ds,classname,image_id):
    directory=ROOT/'outputs/segmentation/controlled_attention_cv'/family/ds/'shards'
    for path in sorted(directory.glob('*.npz')):
        if '.diagnostics.' in path.name or '.regions.' in path.name:continue
        with np.load(path,allow_pickle=False) as data:
            ids=data['image_ids'].tolist()
            if image_id in ids:
                return data['counts'][:,ids.index(image_id)],data['configurations'].tolist()
    raise ValueError(f'missing original {family}: {image_id}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--crf-site',type=Path,default=ROOT/'outputs/dependencies/simplecrf_py310')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    spec=json.loads((ROOT/'configs/sg11_sg12_observation_reliability.json').read_text())
    thresholds=tuple(spec['thresholds'])
    pipe=StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5',torch_dtype=torch.float32,local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True);transfer._PIPE=pipe
    modules=[n.removesuffix('.processor') for n in pipe.unet.attn_processors]
    ca=tuple(sorted(n for n in modules if n.endswith('attn2')))
    selected=tuple(sorted(n for n in modules if n.endswith('attn1')))[15]
    WordObservationRecorder.selected_sa=selected
    ca_spec=common.resolve_seed_spec(json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text()),ca,31)
    # Run the original selected SG2 formula without the unrelated grid, using
    # two identical WORD seeds; no optimized token is loaded or evaluated.
    legacy_sg2.RELIABILITY=('cv',);legacy_sg2.GAMMAS=(2.,);legacy_sg2.FORMS=('G',)
    legacy_names=tuple(n for t in legacy_sg2.TRACKS for n in legacy_sg2.cell_names(t))
    records=[]
    for ds,shard,image_id,check_crf in CASES:
        classname,rows=(transfer.voc_rows if ds=='voc' else transfer.coco_rows)(shard)
        row=next(r for r in rows if r['image_id']==image_id)
        transfer._ATTRIBUTION_PROMPT[0]=row['attribution_prompt'];common._RECORDERS.clear()
        manifest=transfer.EVAL/('ovam/voc_sim' if ds=='voc' else 'ovam/coco_captions')
        sample=common.live.generate_attention_sample(pipe,row,manifest,'float16',cross_map_kind='conditional')
        recorder=common._RECORDERS[-1]
        try:
            resolution={n:r for r,ns in sample.modules_by_resolution.items() for n in ns}
            seed=common.ca_seed(recorder,ca,resolution,ca_spec,'cuda')
            observations,reference=propagate_qk(recorder.sa_qk[selected],seed,spec['sa'],'cuda')
            maps,diagnostics=reliability_maps(observations,seed,spec)
            dense=locked_sg2_maps(recorder.sa_qk[selected],seed)
            small_truth=F.interpolate(torch.as_tensor(sample.target_mask,dtype=torch.float32)[None,None],(64,64),mode='nearest')[0,0].to('cuda')
            old,_=legacy_sg2.compute(recorder,selected,torch.stack((seed.reshape(-1),seed.reshape(-1))),small_truth,'cuda')
            old_selected=old[[legacy_names.index(n) for n in ('W/REF','W/P1','W/G_cv_g20_n1')]]
            errors=(old_selected-dense).abs().max(1).values.cpu().numpy()
            torch.testing.assert_close(old_selected,dense,rtol=1e-5,atol=2e-6)
            torch.testing.assert_close(maps[0],reference,rtol=1e-5,atol=2e-6)
            sg_counts,_=cv.score_maps(maps.reshape(-1,64,64),sample.target_mask,'minmax',thresholds)
            dense_counts,_=cv.score_maps(dense.reshape(-1,64,64),sample.target_mask,'minmax',thresholds)
            previous,_=original_counts('sg11_sg12_observation_reliability',ds,classname,image_id)
            previous_dense,_=original_counts('sg2_densecrf_confirmation',ds,classname,image_id)
            old_sg8,names8=original_counts('sg8_observation_consensus',ds,classname,image_id)
            stats={'dataset':ds,'class':classname,'image_id':image_id,'old_new_SG2_max_abs_by_REF_P1_SG2':errors.tolist(),
                   'count_delta_sg11_saved_max':int(np.abs(sg_counts-previous).max()),
                   'count_delta_dense_raw_saved_max':int(np.abs(dense_counts-previous_dense[:3]).max()),
                   'count_delta_sg8_REF_max':int(np.abs(sg_counts[0]-old_sg8[names8.index('W/REF')]).max()),
                   'image_sha256':hashlib.sha256(sample.generated_image.tobytes()).hexdigest(),
                   'raw_ranges':{n:float(v.max()-v.min()) for n,v in [('seed',seed),('REF',maps[0]),('P1',dense[1]),('SG2',dense[2])]},
                   'normalization_amplification_REF':float(1/(maps[0].max()-maps[0].min()).clamp_min(1e-12))}
            if check_crf:
                crf=np.concatenate([transfer.crf_counts(m.reshape(64,64),sample.target_mask,sample.generated_image,args.crf_site) for m in dense])
                stats['count_delta_dense_CRF_saved_max']=int(np.abs(crf-previous_dense[3:]).max())
            filename=f'{ds}_{image_id}.npz'
            np.savez_compressed(args.output_dir/filename,seed=seed.cpu().numpy(),maps=maps.cpu().numpy(),dense=dense.cpu().numpy(),
                                legacy_dense=old_selected.cpu().numpy(),counts_sg11=sg_counts,counts_dense=dense_counts)
            stats['small_arrays']=filename;records.append(stats)
            (args.output_dir/'audit.json').write_text(json.dumps({'environment':cv.runtime_environment(),'samples':records},indent=2)+'\n')
            print(json.dumps(stats),flush=True)
            del seed,observations,reference,maps,diagnostics,dense,old,old_selected,small_truth
        finally:
            recorder.ca_calls.clear();recorder.sa_qk.clear();sample.clear()
            del sample,recorder
            common._RECORDERS.clear();gc.collect();torch.cuda.empty_cache()
    print('All live selected-SG2 and propagation algebra audits passed',flush=True)


if __name__=='__main__':main()
