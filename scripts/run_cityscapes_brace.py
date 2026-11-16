"""Exploratory fixed-image Cityscapes BRACE/SCLIP/early-geo comparison."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sg27_runtime, sg_steps_test as st
from src.attention_capture.live import _install
from src.segmentation.sg25_sa import SG25Recorder, _attention, _resize
from src.segmentation.sg_combined_v2 import aggregate
from src.segmentation.attention_aggregation import find_token_subsequence

CLASSES = ['road','sidewalk','building','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
LINES = ['wall, fence, pole, traffic light, traffic sign, other object',
         'road, street','sidewalk, pavement','building','tree, bush, hedge, vegetation',
         'grass, dirt, soil, terrain','sky','person, pedestrian',
         'cyclist, motorcyclist, person riding a bicycle','car','truck','bus','train, tram',
         'motorcycle, motorbike, scooter','bicycle, bike']
METHODS = ['brace','sclip_brace','early_geo']
THRESHOLDS = [.40,.55,.70]
DATA = ROOT / 'datasets/cityscapes_lora_benchmark_20261116'
OUT = ROOT / 'outputs/segmentation/cityscapes_50'


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(obj, indent=2)+'\n'); tmp.replace(path)


def norm(x):
    lo=x.amin((-2,-1),keepdim=True); hi=x.amax((-2,-1),keepdim=True)
    return (x-lo)/(hi-lo).clamp_min(1e-12)


class MultiA3Recorder(SG25Recorder):
    """Collect independent padded class queries, with unchanged A3 layer weights."""
    query_states = None
    query_positions = None

    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.multi_pending={}
        self.multi_sum=torch.zeros((len(CLASSES),64,64),device=self.pipe.device)

    def record(self,name,kind,query,key,value,attention_mask,scale,attribution_key=None):
        super().record(name,kind,query,key,value,attention_mask,scale,attribution_key)
        if kind!='cross' or name not in self.ca_modules:
            return
        module=self.pipe.unet.get_submodule(name)
        states=self.query_states
        if module.norm_cross:states=module.norm_encoder_hidden_states(states)
        keys=module.to_k(states).reshape(len(CLASSES),77,module.heads,-1).transpose(1,2)
        if module.norm_k is not None:keys=module.norm_k(keys)
        probs=(query[1][None].float() @ keys.float().transpose(-1,-2)*scale).softmax(-1)
        maps=torch.stack([probs[i,:,:,p].mean(-1).mean(0) for i,p in enumerate(self.query_positions)])
        side=math.isqrt(maps.shape[-1])
        self.multi_pending[name]=maps.reshape(len(CLASSES),1,side,side)

    def flush_step(self,step,timestep,latent):
        super().flush_step(step,timestep,latent)
        _,label,weights=self.weights[-1]
        assert label=='selected'
        for i,name in enumerate(self.selected_modules):
            up=F.interpolate(self.multi_pending[name],size=(64,64),mode='bilinear',align_corners=False)[:,0]
            self.multi_sum+=up*float(weights[i])/31
        self.multi_pending.clear()

    def clear(self):
        super().clear();self.multi_pending.clear();self.multi_sum=None


@torch.inference_mode()
def batched_v3(sa,seeds):
    """Evaluate only frozen V3, sharing each call's attention across seed columns."""
    n=len(seeds); c16=_resize(seeds,64,16)
    sums=torch.zeros((n,4096),device=seeds.device)
    for call in range(11):
        a16=_attention(sa[16][call],16).mean(0)
        a32=_attention(sa[32][call],32).mean(0)
        a64=_attention(sa[64][call],64).mean(0)
        raw16=c16 @ a16.T
        raw32=_resize(raw16,16,32) @ a32.T
        parent=raw16.clamp_min(0).reshape(n,16,16)
        q=raw32.clamp_min(0).reshape(n,16,2,16,2)
        means=q.mean((2,4))
        factor=torch.where(means>parent,parent/(means+1e-12),torch.ones_like(means))
        capped=(q*factor[:,:,None,:,None]).reshape(n,1024)
        sums+=(_resize(capped,32,64) @ a64.T)/11
    return sums.reshape(n,64,64)


@torch.inference_mode()
def capture(a):
    torch.set_num_threads(1)
    from scripts import sg_clip_dense as cd
    rows=json.loads((DATA/'manifest.json').read_text())
    if a.max_images:rows=rows[:a.max_images]
    rows=rows[a.shard::a.num_shards]
    out=OUT/'maps';out.mkdir(parents=True,exist_ok=True)
    todo=[r for r in rows if not (out/(r['image_id']+'.npz')).exists()]
    if not todo:print('All requested captures already complete',flush=True);return
    runtime=sg27_runtime.make_runtime('stable-diffusion-v1-5/stable-diffusion-v1-5',MultiA3Recorder)
    pipe=runtime['pipe']
    pipe.load_lora_weights(str(ROOT),weight_name='pytorch_lora_weights.safetensors',adapter_name='cityscapes',local_files_only=True)
    pipe.set_adapters('cityscapes',adapter_weights=1.)
    clip=cd.DenseCLIP('B16',torch.device('cuda'))
    names,idx=cd.split(LINES);texts=clip.text(names)
    prompts=[f'a photograph of a {c}' for c in CLASSES]
    tokens=pipe.tokenizer(prompts,padding='max_length',max_length=77,truncation=True,return_tensors='pt')
    states=pipe.text_encoder(tokens.input_ids.to('cuda'))[0]
    positions=[list(find_token_subsequence(tokens.input_ids[i].tolist(),pipe.tokenizer(c,add_special_tokens=False)['input_ids'])) for i,c in enumerate(CLASSES)]
    MultiA3Recorder.query_states=states
    MultiA3Recorder.query_positions=positions
    MultiA3Recorder.attribution_prompt=prompts[0]
    from src.segmentation.sg24_ca import content_positions
    MultiA3Recorder.content=content_positions(pipe.tokenizer,prompts[0],tuple(positions[0]))
    checked=(OUT/'selftest.json').exists()
    for row in todo:
        t0=time.perf_counter();iid=row['image_id'];gen=row['generation']
        torch.cuda.reset_peak_memory_stats()
        img=np.asarray(Image.open(DATA/row['image_file']).convert('RGB')).copy()
        png_hash=hashlib.sha256((DATA/row['image_file']).read_bytes()).hexdigest()
        assert png_hash==gen['png_sha256']
        pipe.set_adapters('cityscapes',adapter_weights=gen['lora_scale'])
        # DenseCLIP operates on the ORIGINAL image, then its probabilities are pooled
        # to the square attention grid, exactly as the earlier real-photo workflow.
        _,prob=cd.postprocess(clip.logits(img,texts),idx,65,.1,.1)
        clip_seeds=F.adaptive_avg_pool2d(prob[1:,None].float(),(64,64))[:,0]
        sclippool=clip_seeds.clone()
        tclip=time.perf_counter()-t0
        resized=np.asarray(Image.fromarray(img).resize((512,512),Image.Resampling.BICUBIC)).copy()
        x=torch.from_numpy(resized).permute(2,0,1)[None].to('cuda',pipe.vae.dtype)/127.5-1
        z0=pipe.vae.encode(x).latent_dist.mean*pipe.vae.config.scaling_factor
        noise_seed=zlib.crc32(iid.encode())
        noise=torch.randn(z0.shape,generator=torch.Generator(device='cuda').manual_seed(noise_seed),device='cuda',dtype=torch.float32)
        pipe.scheduler.set_timesteps(30,device='cuda');ts=pipe.scheduler.timesteps.tolist()
        def latent_at(t):
            alpha=pipe.scheduler.alphas_cumprod[int(t)].to('cuda')
            return (alpha.sqrt()*z0.float()+(1-alpha).sqrt()*noise).to(z0.dtype)
        runtime['recorders'].clear()
        rec=MultiA3Recorder(target_positions=positions[0],prompt_token_count=77,ovam_target_position=positions[0][-1],attribution_hidden_states=states[:1],storage_dtype='float16')
        _install(pipe,rec)
        def cb(p,i,t,kw):
            rec.flush_step(i,int(t),kw['latents'])
            if i+1<len(ts):kw['latents']=latent_at(ts[i+1])
            return kw
        pipe(prompt=gen['prompt'],negative_prompt=gen['negative_prompt'],height=512,width=512,
             num_inference_steps=30,guidance_scale=7.5,latents=latent_at(ts[0]),output_type='latent',
             callback_on_step_end=cb,callback_on_step_end_tensor_inputs=['latents'])
        rec.finalize();assert len(rec.steps)==31
        seeds=norm(rec.multi_sum)
        seed_error=float((seeds[0]-rec.seeds()['selected/A3']).abs().max())
        assert seed_error<1e-5,seed_error
        sa={16:rec.multiscale_qk[runtime['modules'][16]],32:rec.multiscale_qk[runtime['modules'][32]],64:rec.sa_qk}
        tcapture=time.perf_counter()-t0-tclip
        clip_seeds=norm(clip_seeds)
        early=norm(torch.sqrt((seeds*clip_seeds).clamp_min(0)))
        inputs=torch.stack((seeds,clip_seeds,early)).reshape(-1,64,64)
        maps=batched_v3(sa,inputs)
        checks=[]
        if not checked:
            # Check each method, including a non-primary class, against frozen source.
            for k in [0,len(CLASSES)+13,2*len(CLASSES)+6]:
                reference=aggregate(sa,inputs[k])[0]['V3']
                error=float((maps[k]-reference).abs().max());checks.append({'seed_index':k,'max_error':error})
                assert error<=1e-5,checks
            save_json(OUT/'selftest.json',{'status':'PASS','A3_seed_error':seed_error,'V3_checks':checks})
            print('SELFTEST PASS '+json.dumps(checks),flush=True);checked=True
        maps=maps.reshape(3,len(CLASSES),64,64)
        meta={'image_id':iid,'classes':CLASSES,'methods':METHODS,'thresholds':THRESHOLDS,'source_png_sha256':png_hash,
              'lora_sha256':gen['lora_sha256'],'lora_scale':gen['lora_scale'],'source_generation_prompt':gen['prompt'],
              'source_negative_prompt':gen['negative_prompt'],'attribution_prompts':prompts,'noise_seed':noise_seed,
              'capture':'image-anchored renoise, VAE mean, 512 square, original generation conditioning',
              'timesteps':ts,'A3_seed_error':seed_error,'V3_checks':checks,'sclip_vocabulary':LINES,
              'sclip_logit_scale':65,'sclip_probability_threshold':.1,'sclip_area_threshold':.1,
              'sclip_seconds':tclip,'capture_seconds':tcapture,'seconds':time.perf_counter()-t0,
              'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
              'source_size':[img.shape[1],img.shape[0]],'empty_clip_seed_classes':[c for c,v in zip(CLASSES,clip_seeds) if float(v.max())==0]}
        target=out/(iid+'.npz');tmp=out/(iid+'.tmp.npz')
        np.savez_compressed(tmp,maps=maps.cpu().numpy(),a3_seed=seeds.cpu().numpy(),clip_seed=clip_seeds.cpu().numpy(),
                            sclip_probability64=sclippool.cpu().numpy(),metadata=json.dumps(meta))
        tmp.replace(target);save_json(out/(iid+'.json'),meta)
        print(f'[city-brace] {iid} clip {tclip:.1f}s capture {tcapture:.1f}s total {meta["seconds"]:.1f}s peak {meta["peak_gpu_allocated_gib"]:.2f}GiB empty-clip {len(meta["empty_clip_seed_classes"])}',flush=True)
        rec.clear();rec.attribution_hidden_states=None;runtime['recorders'].clear()
        del rec,sa,maps,inputs,seeds,clip_seeds,prob,z0,x,early,sclippool
        gc.collect();torch.cuda.empty_cache()


def overlay(img,mask):
    base=Image.fromarray(img).convert('RGB');tint=Image.new('RGB',base.size,(240,40,30))
    alpha=Image.fromarray((mask.astype(np.uint8)*105),'L')
    mixed=Image.composite(tint,base,alpha)
    border=Image.fromarray(mask.astype(np.uint8)*255).filter(ImageFilter.FIND_EDGES)
    mixed.paste((255,230,30),(0,0),border)
    return mixed


def render_one(iid):
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import apply_ovam_densecrf
    target=OUT/'image_reports'/f'{iid}.json'
    if target.exists():return iid
    rows={r['image_id']:r for r in json.loads((DATA/'manifest.json').read_text())}
    row=rows[iid];img=np.asarray(Image.open(DATA/row['image_file']).convert('RGB'))
    z=np.load(OUT/'maps'/f'{iid}.npz');maps=z['maps'];h,w=img.shape[:2]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    stats=[];t0=time.perf_counter()
    for k,cls in enumerate(CLASSES):
        full=np.stack([st.up(st.mm(m[k]),(h,w)) for m in maps])
        sheets=[];masks=np.empty((3,3,h,w),dtype=bool)
        for mi,method in enumerate(METHODS):
            for ti,t in enumerate(THRESHOLDS):
                raw=full[mi]>=t
                # A spatially constant hard binary unary stays constant under R2.
                # Verified against its original output in the smoke render.
                refined=apply_ovam_densecrf(img,raw,st.CRF_SITE) if raw.any() and not raw.all() else raw.copy()
                masks[mi,ti]=refined
                for mode,mask in [('raw',raw),('crf',refined)]:
                    p=OUT/'masks'/iid/cls/f'{method}_t{t:.2f}_{mode}.png';p.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(mask.astype(np.uint8)*255).save(p)
                stats.append({'image_id':iid,'class':cls,'method':method,'threshold':t,'raw_fraction':float(raw.mean()),'crf_fraction':float(refined.mean()),'empty':not bool(refined.any()),'full':bool(refined.all())})
        # Original + heatmaps in top row; source + method overlays for each threshold.
        cellw=256;cellh=220;sheet=Image.new('RGB',(4*cellw,4*cellh),(250,250,250));draw=ImageDraw.Draw(sheet)
        def tile(im,col,rr,label):
            thumb=ImageOps.contain(im,(cellw,188));x=col*cellw+(cellw-thumb.width)//2;y=rr*cellh
            sheet.paste(thumb,(x,y));draw.text((col*cellw+4,y+190),label,font=small,fill='black')
        tile(Image.fromarray(img),0,0,f'{iid}  {cls}  ORIGINAL')
        for mi,method in enumerate(METHODS):
            m=full[mi];rgb=np.stack((m,np.minimum(m*2,1),1-m),-1)
            tile(Image.fromarray((rgb*255).astype(np.uint8)),mi+1,0,method+' score (min-max)')
        for ti,t in enumerate(THRESHOLDS):
            tile(Image.fromarray(img),0,ti+1,f'tau {t:.2f} | red mask / yellow edge')
            for mi,method in enumerate(METHODS):
                frac=float(masks[mi,ti].mean())
                tile(overlay(img,masks[mi,ti]),mi+1,ti+1,f'{method} + R2 | area {frac:.1%}')
        p=OUT/'comparisons'/cls/f'{iid}.jpg';p.parent.mkdir(parents=True,exist_ok=True);sheet.save(p,quality=87)
    save_json(target,{'image_id':iid,'seconds':time.perf_counter()-t0,'stats':stats})
    return iid


def render(a):
    from concurrent.futures import ProcessPoolExecutor
    ids=[p.stem for p in sorted((OUT/'maps').glob('cs_*.npz')) if '.tmp' not in p.stem]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,iid in enumerate(pool.map(render_one,ids),1):print(f'[render] {i}/{len(ids)} {iid}',flush=True)


def report(a):
    rows=json.loads((DATA/'manifest.json').read_text());ids=[r['image_id'] for r in rows]
    reports=[json.loads((OUT/'image_reports'/f'{i}.json').read_text()) for i in ids]
    stats=[s for r in reports for s in r['stats']]
    assert len(stats)==6300
    assert len(list((OUT/'comparisons').glob('*/*.jpg')))==700
    assert len(list((OUT/'masks').glob('*/*/*_crf.png')))==6300
    assert len(list((OUT/'masks').glob('*/*/*_raw.png')))==6300
    summaries=[]
    for cls in CLASSES:
        for method in METHODS:
            for t in THRESHOLDS:
                ss=[s for s in stats if s['class']==cls and s['method']==method and s['threshold']==t]
                summaries.append({'class':cls,'method':method,'threshold':t,'mean_foreground_fraction':float(np.mean([s['crf_fraction'] for s in ss])),'empty_masks':sum(s['empty'] for s in ss),'full_masks':sum(s['full'] for s in ss)})
    save_json(OUT/'summary.json',{'images':50,'classes':CLASSES,'methods':METHODS,'thresholds':THRESHOLDS,'post_crf_masks':6300,'comparisons':700,'summaries':summaries})
    import csv
    with (OUT/'mask_statistics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    options=''.join(f'<option>{c}</option>' for c in CLASSES)
    image_options=''.join(f'<option>{r["image_id"]}</option>' for r in rows)
    witnesses=json.loads((DATA/'coverage.json').read_text())['class_witnesses']
    html='''<!doctype html><meta charset="utf-8"><title>Cityscapes 50 — BRACE comparison</title>
<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;background:#eee;color:#222}select,button{font:inherit;padding:8px}img{width:100%;height:auto}header{position:sticky;top:0;background:#eee;padding:10px}p{line-height:1.5}a{color:#1454a1}</style>
<h1>Cityscapes LoRA 50: class-wise threshold comparison</h1><p>Same fixed image and captured attention for BRACE, SCLIP-seeded BRACE and early geometric fusion. Columns compare methods; rows compare thresholds 0.40 / 0.55 / 0.70. Top row contains the original and normalized scores. Binary masks can overlap across classes. No ground-truth masks or accuracy labels are available: area is not quality. Absent classes can still produce masks after min-max normalization. This is image-anchored LoRA inference, not original-trajectory replay.</p>
<header><label>Image <select id="image">IMAGE_OPTIONS</select></label> <label>Class <select id="cls">CLASS_OPTIONS</select></label> <button onclick="example()">Class example</button> <button onclick="move(-1)">Previous image</button> <button onclick="move(1)">Next image</button> <a id="full" target="_blank">Open sheet</a></header><img id="sheet"><p><a href="README.md">Protocol and run details</a> · <a href="summary.json">Summary JSON</a> · <a href="mask_statistics.csv">Mask statistics CSV</a></p>
<script>const witnesses=WITNESS_JSON;const im=document.getElementById('image'),cl=document.getElementById('cls');function update(){let p=`comparisons/${cl.value}/${im.value}.jpg`;document.getElementById('sheet').src=p;document.getElementById('full').href=p;location.hash=im.value+'/'+cl.value}function example(){im.value=witnesses[cl.value];update()}function move(d){im.selectedIndex=(im.selectedIndex+d+im.length)%im.length;update()}im.onchange=cl.onchange=update;let parts=location.hash.slice(1).split('/');if(parts.length===2){im.value=parts[0];cl.value=parts[1]}else{cl.value='person'}update();</script>'''.replace('IMAGE_OPTIONS',image_options).replace('CLASS_OPTIONS',options).replace('WITNESS_JSON',json.dumps(witnesses))
    (OUT/'index.html').write_text(html)
    metas=[json.loads((OUT/'maps'/f'{iid}.json').read_text()) for iid in ids]
    text=f'''# Cityscapes LoRA 50 — exploratory segmentation comparison\n\nComplete: 50 images × 14 classes × 3 methods × 3 thresholds = **6300 post-CRF masks**, with 6300 pre-CRF masks, 700 comparison sheets, seeds and propagated maps.\n\nOpen `index.html` to browse every image/class. Columns: BRACE, SCLIP-seeded BRACE, early geometric fusion. Rows: thresholds **0.40 / 0.55 / 0.70**. All methods use the same captured attention for each image.\n\nNo ground-truth masks exist. This run provides visual comparison and mask area statistics, not mIoU, precision, recall, threshold selection or a scientific winner. Class masks are independent and may overlap; absent-class maps can yield false foreground because each nonconstant class map is min-max normalized.\n\n## Capture and variants\n\nThe original PNGs are fixed. Capture uses the repository's image-anchored renoise procedure: 512-square bicubic input, VAE posterior mean, fixed crc32(image_id) noise, original generation prompt and negative prompt, 30-step PNDM with 31 UNet calls, and the supplied Cityscapes LoRA at the recorded image scale. Outputs are resized back to native dimensions. This is a Cityscapes/LoRA input adaptation, not exact replay of original batched generation.\n\nThe selected A3 seed and V3 one-sided-capped hierarchy plus R2 remain unchanged. Independent 77-token `a photograph of a <class>` attribution queries allow unprompted classes. Seeds and all methods share the image's own attention. SCLIP is ViT-B/16 with the committed Cityscapes vocabulary, logit scale 65 and probability/area threshold .1. VOC21 grouped the stuff classes into background and cannot be reused unchanged for this task.\n\nThe prior VOC-selected SCLIP-seeded variant is V3. Early sqrt(a*c) is the requested exploratory fusion; the earlier registered fusion gain test failed for its selected arithmetic candidate. No variant is promoted here.\n\nSelftest: `selftest.json` checks batched A3 and V3 against unchanged source. Mean capture+propagation time {np.mean([m['seconds'] for m in metas]):.1f} s/image, peak allocated GPU memory {max(m['peak_gpu_allocated_gib'] for m in metas):.2f} GiB. Timings exclude CPU CRF/rendering.\n\n## Files\n\n- `comparisons/<class>/<image>.jpg`: original, heatmaps and 9 post-CRF overlays.\n- `masks/<image>/<class>/<method>_t<threshold>_raw.png` and `_crf.png`: native-size binary PNGs, 0/255.\n- `maps/<image>.npz`: [method,class,64,64] propagated maps, A3 and SCLIP seeds; JSON sidecars record settings and source hashes.\n- `summary.json`, `mask_statistics.csv`: foreground area and empty/full-mask counts (not quality metrics).\n- `PREREG.md`: committed before inference.\n- `VISUAL_REVIEW.md`: observations from representative completed sheets.\n'''
    (OUT/'README.md').write_text(text)
    save_json(OUT/'status.json',{'status':'complete','images':50,'post_crf_masks':6300,'comparisons':700})
    print('COMPLETE: 50 images, 6300 post-CRF masks, 700 comparison sheets',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['capture','render','report']);p.add_argument('--max-images',type=int);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
    a=p.parse_args();assert 0<=a.shard<a.num_shards;OUT.mkdir(parents=True,exist_ok=True)
    {'capture':capture,'render':render,'report':report}[a.command](a)

if __name__=='__main__':main()
