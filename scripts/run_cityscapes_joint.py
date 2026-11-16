"""Owner-selected Cityscapes joint, mutually exclusive BRACE comparison."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_cityscapes_brace as prior, sg27_runtime
from src.attention_capture.live import _install
from src.segmentation.sg25_sa import SG25Recorder
from src.segmentation.sg_combined_v2 import aggregate
from src.segmentation.attention_aggregation import find_token_subsequence
from src.segmentation.sg24_ca import content_positions
from src.segmentation.ovam_full import _import_densecrf, OVAM_CRF_PARAMETERS

DATA=ROOT/'datasets/cityscapes_lora_benchmark_20261116'
OUT=ROOT/'outputs/segmentation/cityscapes_joint_owner'
RUN=ROOT/'runs/cityscapes_joint_owner'
IDS=[0,1,2,4,5,14,25,27,28,34,43,45,47,57,61,65,89,112,115,124,143,144,146,149,499,497,494,490,489,486,471,468,467,456,447,437,430,422,412,402,399,394,390,381,372,365,338,335,331,324,318,306,302,298,292,289,282,283,274,276,266,257,215,204,192]
GROUPS=[['road'],['sidewalk'],['building','wall'],['vegetation','terrain'],['sky'],['person','rider'],['car'],['bus'],['train','tram'],['motorcycle','bicycle']]
NAMES=['/'.join(g) for g in GROUPS]
ATOMS=[c for g in GROUPS for c in g]
PROMPT=', '.join(ATOMS)
METHODS=['brace','sclip_brace','early_geo']
THRESHOLDS=[.4,.55,.7]
LINES=['background, other object, truck, fence, pole, traffic light, traffic sign',
       'road, street','sidewalk, pavement','building, wall','tree, bush, hedge, vegetation, grass, dirt, soil, terrain',
       'sky','person, pedestrian, rider, cyclist, motorcyclist, person riding a bicycle','car','bus','train, tram',
       'motorcycle, motorbike, scooter, bicycle, bike']
COLORS=np.array([[0,0,0],[128,64,128],[244,35,232],[70,70,70],[107,142,35],[70,130,180],
                 [220,20,60],[0,0,142],[0,60,100],[0,80,100],[119,11,32]],dtype=np.uint8)
prior.CLASSES=ATOMS
CONFIG={'version':1,'ids':IDS,'groups':GROUPS,'attribution_prompt':PROMPT,'sclip_vocabulary':LINES,
        'methods':METHODS,'thresholds':THRESHOLDS,'normalization':'one common peak per class stack; no per-class minmax',
        'capture':'image-anchored renoise; recorded LoRA scale; original conditioning; float32',
        'decoder':'argmax across 10 groups, threshold max to background; CRF can veto winner only'}
CONFIG_HASH=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def jsave(p,x):
    prior.save_json(p,x)

def peak(x):
    return x.clamp_min(0)/x.amax().clamp_min(1e-12)

def group_union(x):
    return torch.stack([x[[ATOMS.index(c) for c in g]].amax(0) for g in GROUPS])

def assign(scores,t):
    winner=scores.argmax(0).astype(np.uint8)+1
    return np.where(scores.max(0)>=t,winner,0).astype(np.uint8)

class JointRecorder(prior.MultiA3Recorder):
    """One shared text softmax; retain each class token's raw A3 measure."""
    def record(self,name,kind,query,key,value,attention_mask,scale,attribution_key=None):
        SG25Recorder.record(self,name,kind,query,key,value,attention_mask,scale,attribution_key)
        if kind!='cross' or name not in self.ca_modules:return
        assert attribution_key is not None
        prob=(query[1].float() @ attribution_key[0].float().transpose(-1,-2)*scale).softmax(-1)
        maps=torch.stack([prob[...,p].mean(-1).mean(0) for p in self.query_positions])
        side=int(maps.shape[-1]**.5)
        self.multi_pending[name]=maps.reshape(len(ATOMS),1,side,side)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'manifest.json').exists(),'Run already prepared; use existing manifest.'
    weight_hash=digest(ROOT/'pytorch_lora_weights.safetensors')
    rows=[]
    for n in IDS:
        iid=f'cs_{n:04d}';p=DATA/'candidates'/f'{iid}.png';rec=json.loads(p.with_suffix('.json').read_text())
        assert rec['lora_sha256']==weight_hash
        with Image.open(p) as im:im.verify()
        rows.append({'image_id':iid,'candidate_number':n,'image_file':f'candidates/{iid}.png',
                     'png_sha256':digest(p),'generation':rec})
    assert len(rows)==len(set(IDS))==65
    jsave(OUT/'manifest.json',rows);jsave(OUT/'config.json',dict(CONFIG,config_sha256=CONFIG_HASH))
    shutil.copy2(RUN/'PREREG.md',OUT/'PREREG.md')
    print('Prepared exact owner list: 65 images, 10 joint groups, 585 outputs per decoding mode.')


def selftest_cpu():
    # Joint competition must never independently promote an absent weak class.
    s=np.array([[[.8,.3]],[[.1,.2]]],np.float32)
    assert assign(s,.4).tolist()==[[1,0]]
    assert assign(s,.7).tolist()==[[1,0]]
    assert assign(s,.9).tolist()==[[0,0]]
    assert torch.equal(group_union(torch.arange(len(ATOMS)).float()[:,None,None])[2],torch.tensor([[3.]]))
    labels=np.zeros((32,32),np.uint8);labels[:,:10]=1;labels[:,10:22]=4;labels[:,22:]=10
    im=np.zeros((32,32,3),np.uint8)
    result=np.asarray(_import_densecrf(prior.st.CRF_SITE).densecrf(im,np.eye(11,dtype=np.float32)[labels],OVAM_CRF_PARAMETERS))
    assert np.array_equal(result,labels)
    jsave(OUT/'selftest_cpu.json',{'status':'PASS','joint_assignment':True,'group_union':True,'multiclass_R2_labels':True})
    print('CPU SELFTEST PASS')


@torch.inference_mode()
def capture(a):
    torch.set_num_threads(1)
    from scripts import sg_clip_dense as cd
    rows=json.loads((OUT/'manifest.json').read_text())
    if a.max_images:rows=rows[:a.max_images]
    rows=rows[a.shard::a.num_shards]
    dest=OUT/'maps';dest.mkdir(exist_ok=True)
    todo=[]
    for r in rows:
        p=dest/f'{r["image_id"]}.npz'
        if p.exists():
            with np.load(p) as z:
                meta=json.loads(str(z['metadata']));assert meta['config_sha256']==CONFIG_HASH and meta['source_sha256']==r['png_sha256']
        else:todo.append(r)
    if not todo:print('Shard complete; verified cached results.',flush=True);return
    assert digest(ROOT/'pytorch_lora_weights.safetensors')==todo[0]['generation']['lora_sha256']
    runtime=sg27_runtime.make_runtime('stable-diffusion-v1-5/stable-diffusion-v1-5',JointRecorder)
    pipe=runtime['pipe'];pipe.load_lora_weights(str(ROOT),weight_name='pytorch_lora_weights.safetensors',adapter_name='cityscapes',local_files_only=True)
    clip=cd.DenseCLIP('B16',torch.device('cuda'));names,idx=cd.split(LINES);texts=clip.text(names)
    tokenized=pipe.tokenizer(PROMPT,padding='max_length',max_length=77,truncation=True,return_tensors='pt')
    assert len(pipe.tokenizer(PROMPT,truncation=False)['input_ids'])<=77
    states=pipe.text_encoder(tokenized.input_ids.to('cuda'))[0]
    positions=[list(find_token_subsequence(tokenized.input_ids[0].tolist(),pipe.tokenizer(c,add_special_tokens=False)['input_ids'])) for c in ATOMS]
    JointRecorder.attribution_prompt=PROMPT;JointRecorder.query_positions=positions
    JointRecorder.content=content_positions(pipe.tokenizer,PROMPT,tuple(p for pp in positions for p in pp))
    checked=False
    for row in todo:
        start=time.perf_counter();iid=row['image_id'];gen=row['generation']
        assert digest(DATA/row['image_file'])==row['png_sha256']
        img=np.asarray(Image.open(DATA/row['image_file']).convert('RGB')).copy()
        pipe.set_adapters('cityscapes',adapter_weights=gen['lora_scale']);torch.cuda.reset_peak_memory_stats()
        _,prob=cd.postprocess(clip.logits(img,texts),idx,65,.1,.1)
        c=peak(F.adaptive_avg_pool2d(prob[1:,None].float(),(64,64))[:,0])
        square=np.asarray(Image.fromarray(img).resize((512,512),Image.Resampling.BICUBIC)).copy()
        x=torch.from_numpy(square).permute(2,0,1)[None].to('cuda',pipe.vae.dtype)/127.5-1
        z0=pipe.vae.encode(x).latent_dist.mean*pipe.vae.config.scaling_factor
        noise_seed=zlib.crc32(iid.encode());noise=torch.randn(z0.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(noise_seed))
        pipe.scheduler.set_timesteps(30,device='cuda');ts=pipe.scheduler.timesteps.tolist()
        def latent_at(t):
            alpha=pipe.scheduler.alphas_cumprod[int(t)].to('cuda')
            return (alpha.sqrt()*z0.float()+(1-alpha).sqrt()*noise).to(z0.dtype)
        runtime['recorders'].clear()
        rec=JointRecorder(target_positions=positions[0],prompt_token_count=77,ovam_target_position=positions[0][-1],attribution_hidden_states=states,storage_dtype='float16')
        _install(pipe,rec)
        def cb(p,i,t,kw):
            rec.flush_step(i,int(t),kw['latents'])
            if i+1<len(ts):kw['latents']=latent_at(ts[i+1])
            return kw
        pipe(prompt=gen['prompt'],negative_prompt=gen['negative_prompt'],height=512,width=512,num_inference_steps=30,
             guidance_scale=7.5,latents=latent_at(ts[0]),output_type='latent',callback_on_step_end=cb,callback_on_step_end_tensor_inputs=['latents'])
        rec.finalize();assert len(rec.steps)==31
        parity=float((prior.norm(rec.multi_sum[:1])[0]-rec.seeds()['selected/A3']).abs().max());assert parity<=1e-5,parity
        a3=peak(group_union(rec.multi_sum));fusion=peak(torch.sqrt((a3*c).clamp_min(0)))
        sa={16:rec.multiscale_qk[runtime['modules'][16]],32:rec.multiscale_qk[runtime['modules'][32]],64:rec.sa_qk}
        seeds=torch.stack((a3,c,fusion)).reshape(30,64,64);raw=prior.batched_v3(sa,seeds)
        checks=[]
        if not checked:
            # Include a nonzero SCLIP seed if available, rather than a trivial empty one.
            ci=10+int(c.amax((-2,-1)).argmax())
            for k in [0,ci,20+int(fusion.amax((-2,-1)).argmax())]:
                ref=aggregate(sa,seeds[k])[0]['V3'];error=float((ref-raw[k]).abs().max());assert error<=1e-5,error
                checks.append({'seed_index':k,'max_error':error})
            jsave(OUT/f'selftest_gpu_{a.shard}.json',{'status':'PASS','A3_error':parity,'V3_checks':checks})
            print('GPU SELFTEST PASS '+json.dumps(checks),flush=True);checked=True
        maps=torch.stack([peak(m) for m in raw.reshape(3,10,64,64)])
        meta={'image_id':iid,'config_sha256':CONFIG_HASH,'source_sha256':row['png_sha256'],'noise_seed':noise_seed,
              'lora_scale':gen['lora_scale'],'lora_sha256':gen['lora_sha256'],'A3_error':parity,'seconds':time.perf_counter()-start,
              'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'groups':NAMES,'methods':METHODS,'shared_query':PROMPT,
              'empty_clip_groups':[name for name,seed in zip(NAMES,c) if not bool(seed.any())]}
        tmp=dest/f'{iid}.tmp.npz';np.savez_compressed(tmp,maps=maps.cpu().numpy(),a3_seed=a3.cpu().numpy(),
             sclip_seed=c.cpu().numpy(),metadata=json.dumps(meta));tmp.replace(dest/f'{iid}.npz');jsave(dest/f'{iid}.json',meta)
        print(f'[joint] {iid} {meta["seconds"]:.1f}s peak {meta["peak_gpu_allocated_gib"]:.2f}GiB',flush=True)
        rec.clear();rec.attribution_hidden_states=None;runtime['recorders'].clear()
        del rec,sa,seeds,maps,raw,a3,c,fusion,prob,z0,x
        gc.collect();torch.cuda.empty_cache()


def color_labels(labels):return Image.fromarray(COLORS[labels])

def overlay(img,labels):
    tinted=np.rint(.6*img+.4*COLORS[labels]).astype(np.uint8)
    tinted[labels==0]=img[labels==0]
    return Image.fromarray(tinted)


def render_one(iid):
    torch.set_num_threads(1);report=OUT/'image_reports'/f'{iid}.json'
    if report.exists():
        old=json.loads(report.read_text());assert old['config_sha256']==CONFIG_HASH;return iid
    rows={r['image_id']:r for r in json.loads((OUT/'manifest.json').read_text())};row=rows[iid]
    assert digest(DATA/row['image_file'])==row['png_sha256']
    img=np.asarray(Image.open(DATA/row['image_file']).convert('RGB')).copy();h,w=img.shape[:2]
    with np.load(OUT/'maps'/f'{iid}.npz') as z:
        assert json.loads(str(z['metadata']))['config_sha256']==CONFIG_HASH;maps=z['maps']
    crf=_import_densecrf(prior.st.CRF_SITE);stats=[];all_labels={mode:[] for mode in ['raw','crf']}
    for mi,method in enumerate(METHODS):
        full=np.stack([prior.st.up(m,(h,w)) for m in maps[mi]])
        winner=full.argmax(0).astype(np.uint8)+1;confidence=full.max(0)
        per_mode={'raw':[],'crf':[]}
        for t in THRESHOLDS:
            raw=assign(full,t)
            proposed=np.asarray(crf.densecrf(img,np.eye(11,dtype=np.float32)[raw],OVAM_CRF_PARAMETERS))
            assert proposed.shape==(h,w) and proposed.min()>=0 and proposed.max()<=10
            refined=np.where((proposed==winner)&(confidence>=t),winner,0).astype(np.uint8)
            for mode,labels in [('raw',raw),('crf',refined)]:
                assert np.all((labels==0)|(labels==winner));assert not np.any(labels[confidence<t])
                dest=OUT/'labels'/iid;dest.mkdir(parents=True,exist_ok=True)
                Image.fromarray(labels).save(dest/f'{method}_t{t:.2f}_{mode}.png')
                color_labels(labels).save(dest/f'{method}_t{t:.2f}_{mode}_color.png')
                per_mode[mode].append(labels)
                stats.append({'method':method,'threshold':t,'mode':mode,'fractions':np.bincount(labels.ravel(),minlength=11).astype(float).tolist()})
                stats[-1]['fractions']=[v/(h*w) for v in stats[-1]['fractions']]
            assert not np.any(refined[raw==0])
        for mode in per_mode:all_labels[mode].append(per_mode[mode])
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    for mode,methods in all_labels.items():
        cellw,cellh=256,220;sheet=Image.new('RGB',(1024,880),'white');draw=ImageDraw.Draw(sheet)
        def tile(im,col,rr,text):
            thumb=ImageOps.contain(im,(256,188));xx=col*256+(256-thumb.width)//2;yy=rr*220
            sheet.paste(thumb,(xx,yy));draw.text((col*256+3,yy+190),text,font=font,fill='black')
        tile(Image.fromarray(img),0,0,f'{iid} ORIGINAL | {mode}')
        for mi,method in enumerate(METHODS):tile(color_labels(methods[mi][1]),mi+1,0,method+' labels @ 0.55')
        for ti,t in enumerate(THRESHOLDS):
            tile(Image.fromarray(img),0,ti+1,f'tau {t:.2f} | background unchanged')
            for mi,method in enumerate(METHODS):tile(overlay(img,methods[mi][ti]),mi+1,ti+1,method+f' | bg {(methods[mi][ti]==0).mean():.1%}')
        dest=OUT/'comparisons';dest.mkdir(exist_ok=True);sheet.save(dest/f'{iid}_{mode}.jpg',quality=88)
    jsave(report,{'image_id':iid,'config_sha256':CONFIG_HASH,'stats':stats});return iid


def render(a):
    ids=[r['image_id'] for r in json.loads((OUT/'manifest.json').read_text())]
    if a.max_images:ids=ids[:a.max_images]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for n,iid in enumerate(pool.map(render_one,ids),1):print(f'[render] {n}/{len(ids)} {iid}',flush=True)


def report():
    rows=json.loads((OUT/'manifest.json').read_text());assert len(rows)==65
    for row in rows:
        iid=row['image_id'];assert (OUT/'image_reports'/f'{iid}.json').exists()
        for method in METHODS:
            for t in THRESHOLDS:
                for mode in ['raw','crf']:
                    with Image.open(OUT/'labels'/iid/f'{method}_t{t:.2f}_{mode}.png') as im:
                        im.load();assert im.size==(row['generation']['width'],row['generation']['height'])
                        assert np.asarray(im).max()<=10
    options=''.join(f'<option>{r["image_id"]}</option>' for r in rows)
    legend=''.join(f'<span style="display:inline-block;margin:5px;padding:4px;background:rgb{tuple(map(int,col))};color:white">{k}: {name}</span>' for k,(col,name) in enumerate(zip(COLORS,['background']+NAMES)))
    html='''<!doctype html><meta charset="utf-8"><title>Owner-selected Cityscapes — joint segmentation</title>
<style>body{font:16px system-ui;max-width:1150px;margin:20px auto;background:#eee}select,button{font:inherit;padding:8px}img{width:100%}header{position:sticky;top:0;background:#eee;padding:10px}</style>
<h1>65 owner-selected images: joint 10-class segmentation</h1><p>Columns: BRACE / SCLIP-seeded / early geometric. Rows: .40 / .55 / .70. Every pixel has one label or background. Class scores use a shared scale. No accuracy is measured without GT.</p>LEGEND
<header><select id="im">OPTIONS</select> <select id="mode"><option value="crf">Joint + R2 winner gate</option><option value="raw">Exact joint argmax</option></select> <button onclick="move(-1)">Previous</button> <button onclick="move(1)">Next</button></header><img id="sheet">
<script>const im=document.getElementById('im'),mode=document.getElementById('mode');function update(){document.getElementById('sheet').src=`comparisons/${im.value}_${mode.value}.jpg`}function move(d){im.selectedIndex=(im.selectedIndex+d+im.length)%im.length;update()}im.onchange=mode.onchange=update;update()</script>'''.replace('OPTIONS',options).replace('LEGEND',legend)
    (OUT/'index.html').write_text(html)
    shutil.copy2(RUN/'PREREG.md',OUT/'README.md')
    jsave(OUT/'summary.json',{'status':'complete','images':65,'groups':NAMES,'methods':METHODS,'thresholds':THRESHOLDS,
          'raw_label_maps':585,'refined_label_maps':585,'comparison_sheets':130,'config_sha256':CONFIG_HASH,
          'validation':'All label maps decode, have native dimensions and values 0..10; winner/threshold rules checked during rendering.'})
    tmp=Path('/tmp/cityscapes_joint_owner_results.zip')
    files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.zip','.sha256'] and '.tmp.' not in p.name and p.name!='status.json']
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(files):z.write(p,Path('cityscapes_joint_owner')/p.relative_to(OUT))
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    dest=OUT/tmp.name;shutil.copy2(tmp,dest);dest.with_suffix('.zip.sha256').write_text(digest(dest)+'  '+dest.name+'\n')
    jsave(OUT/'status.json',{'status':'complete','images':65,'raw_label_maps':585,'refined_label_maps':585,'archive':dest.name})
    print('COMPLETE: 65 images, 585 raw + 585 refined joint label maps; archive verified.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','selftest','capture','render','report'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--num-shards',type=int,default=1);p.add_argument('--max-images',type=int);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();assert 0<=a.shard<a.num_shards
    if a.command=='prepare':prepare()
    elif a.command=='selftest':selftest_cpu()
    elif a.command=='capture':capture(a)
    elif a.command=='render':render(a)
    else:report()

if __name__=='__main__':main()
