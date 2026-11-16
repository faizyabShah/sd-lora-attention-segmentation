"""Build an offline viewer from saved Cityscapes arrays; no model inference."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/segmentation/cityscapes_joint_owner'
DATA=ROOT/'datasets/cityscapes_lora_benchmark_20261116'
KINDS=['ca_seed','sclip_seed','brace_propagated','sclip_propagated','fused_propagated']

HTML=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cityscapes: segmentation, seeds and propagated attention</title>
<style>
*{box-sizing:border-box}body{font:16px system-ui;margin:0;background:#edf0f4;color:#172333}main{max-width:1500px;margin:auto;padding:18px}h1{font-size:25px}p{line-height:1.5}header{position:sticky;top:0;z-index:5;background:#fff;padding:12px;border-bottom:1px solid #ccd3dc;box-shadow:0 2px 6px #0001}select,button{font:inherit;padding:7px;border:1px solid #aab3c0;border-radius:5px;background:white}label{display:inline-flex;align-items:center;gap:5px;margin:4px}button{cursor:pointer}#controls{max-width:1500px;margin:auto;display:flex;flex-wrap:wrap;gap:7px;align-items:center}.hidden{display:none!important}#sheet{width:100%;height:auto;background:#fff}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.panel{background:white;padding:10px;border:1px solid #d3d8e0;border-radius:8px;min-width:0}.panel h3{font-size:16px;margin:0 0 8px}.panel canvas,.panel img{width:100%;height:auto;display:block}.metrics{font:12px ui-monospace,monospace;white-space:pre-wrap;min-height:44px;margin-top:7px}.notice{background:#fff;padding:12px;border-left:4px solid #285a91}.scale{height:14px;width:260px;background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725);display:inline-block;vertical-align:middle;margin:0 8px}.legend span{display:inline-block;margin:4px;padding:4px 8px;color:white;border-radius:3px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}.card{background:white;padding:8px;border-radius:7px}.card img{width:100%;height:auto}.card button{width:100%;margin-top:5px}#message{font-size:14px;color:#344c67}details{margin:18px 0}input[type=range]{width:130px}@media(max-width:850px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:540px){.panels{grid-template-columns:1fr}header{position:static}main{padding:10px}}
</style></head><body>
<header><div id="controls">
<label>Image <select id="image">IMAGE_OPTIONS</select></label><button id="prev">Previous</button><button id="next">Next</button>
<label>View <select id="view"><option value="labelled">Segmentation with class names</option><option value="attention">CA / SCLIP seeds and propagated attention</option><option value="crf">Segmentation overlays — CRF</option><option value="raw">Segmentation overlays — raw</option></select></label>
<label id="classControl" class="hidden">Class <select id="cls">CLASS_OPTIONS</select></label>
</div></header><main>
<h1>All 65 images: predictions, seeds and propagated attention</h1>
<div id="segmentation"><p>Columns: original, BRACE, SCLIP-seeded BRACE, geometric fusion. Rows: thresholds 0.40 / 0.55 / 0.70. These are predicted labels; black is background.</p><div class="legend">CLASS_LEGEND</div><img id="sheet" alt="Original image and three segmentation methods at three thresholds"></div>
<div id="attention" class="hidden">
<p class="notice">These are the <b>saved class maps</b>, before thresholding and CRF. No model or propagation is rerun. CA and SCLIP seeds are shown beside the already-computed SA-propagated outputs. Select a class above to compare it across all five panels.</p>
<div><label><input type="checkbox" id="contrast">Enhance each panel's contrast (display only)</label><label><input type="checkbox" id="overlay">Overlay on image</label><label>Heatmap opacity <input type="range" min="0" max="100" value="60" id="opacity"></label><label><input type="checkbox" checked id="smooth">Smooth grid display</label></div>
<p id="scaleNote"><span>Low</span><span class="scale"></span><span>High</span> — shared saved scale, 0–1. Hover over a map to see its stored 64×64 cell value.</p>
<p id="message" aria-live="polite"></p>
<div class="panels">
<section class="panel"><h3>Original image</h3><img id="original" alt="Original candidate image"><div class="metrics" id="originalInfo"></div></section>
<section class="panel"><h3>CA seed — merged class group</h3><canvas data-kind="ca_seed"></canvas><div class="metrics" id="ca_seedInfo"></div></section>
<section class="panel"><h3>SCLIP seed — after recorded filtering</h3><canvas data-kind="sclip_seed"></canvas><div class="metrics" id="sclip_seedInfo"></div></section>
<section class="panel"><h3>SA propagated — BRACE (CA seed)</h3><canvas data-kind="brace_propagated"></canvas><div class="metrics" id="brace_propagatedInfo"></div></section>
<section class="panel"><h3>SA propagated — SCLIP-seeded BRACE</h3><canvas data-kind="sclip_propagated"></canvas><div class="metrics" id="sclip_propagatedInfo"></div></section>
<section class="panel"><h3>SA propagated — geometric fusion</h3><canvas data-kind="fused_propagated"></canvas><div class="metrics" id="fused_propagatedInfo"></div></section>
</div>
<p>Seeds retain the run's merged groups and common peak scaling. SCLIP seeds include its recorded area filter. Each method's propagated class stack was scaled by its common peak. Scores are <b>relative activations, not calibrated probabilities</b>. Contrast enhancement rescales only the displayed panel, and can visually amplify weak classes; stored values and segmentation predictions stay unchanged. An exactly-zero map remains zero.</p>
<p>“SA propagated” means the class score after propagation through captured self-attention. Raw pairwise SA matrices/Q/K and individual propagation stages were not retained; they are not reconstructed here.</p>
</div>
<details><summary>Choose any of the 65 images</summary><div class="cards">IMAGE_CARDS</div></details>
</main><script>
'use strict';
const classNames=CLASS_NAMES, kinds=MAP_KINDS;
const imageSelect=document.getElementById('image'),view=document.getElementById('view'),classSelect=document.getElementById('cls');
const original=document.getElementById('original');
let current=null, requested=null, scriptNode=null;
const canvasByKind=Object.fromEntries([...document.querySelectorAll('canvas[data-kind]')].map(c=>[c.dataset.kind,c]));
function chosenClass(){return Number(classSelect.value)}
function format(x){return Number(x).toPrecision(6)}
function readValues(encoded){const raw=atob(encoded),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);const dv=new DataView(bytes.buffer),values=new Float32Array(raw.length/4);for(let i=0;i<values.length;i++)values[i]=dv.getFloat32(i*4,true);return values}
window.receiveCityscapesAttention=function(payload){if(payload.image_id!==imageSelect.value)return;current={...payload,arrays:{}};for(const kind of kinds)current.arrays[kind]=readValues(payload.data[kind]);document.getElementById('message').textContent='Loaded saved maps for '+payload.image_id+' — '+classNames[chosenClass()];drawMaps()};
function loadData(){const id=imageSelect.value;if(requested===id&&current&&current.image_id===id){drawMaps();return}requested=id;current=null;document.getElementById('message').textContent='Loading saved maps for '+id+'…';original.src='attention_images/'+id+'.png';document.getElementById('originalInfo').textContent=id+' — original source PNG';for(const c of Object.values(canvasByKind)){c.getContext('2d').clearRect(0,0,c.width,c.height)}if(scriptNode)scriptNode.remove();scriptNode=document.createElement('script');scriptNode.src='attention_data/'+id+'.js';scriptNode.onerror=()=>{if(requested===id)document.getElementById('message').textContent='Could not load attention_data/'+id+'.js. Extract the whole ZIP, keeping its folders beside this HTML.'};document.body.appendChild(scriptNode)}
const stops=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function color(t){t=Math.max(0,Math.min(1,t));const s=t*4,i=Math.min(3,Math.floor(s)),f=s-i;return stops[i].map((v,k)=>Math.round(v*(1-f)+stops[i+1][k]*f))}
function drawMaps(){if(!current||current.image_id!==imageSelect.value)return;const k=chosenClass(),enhance=document.getElementById('contrast').checked,over=document.getElementById('overlay').checked,smooth=document.getElementById('smooth').checked,alpha=Number(document.getElementById('opacity').value)/100;
const width=current.width,height=current.height;document.getElementById('message').textContent=current.image_id+' — '+classNames[k];
document.getElementById('scaleNote').innerHTML='<span>Low</span><span class="scale"></span><span>High</span> — '+(enhance?'each panel uses its displayed min/max; hover values remain on the saved scale.':'shared saved scale, 0–1. Hover for the stored 64×64 cell value.');
for(const kind of kinds){const values=current.arrays[kind].subarray(k*4096,(k+1)*4096);let lo=Infinity,hi=-Infinity,sum=0;for(const v of values){lo=Math.min(lo,v);hi=Math.max(hi,v);sum+=v}const grid=document.createElement('canvas');grid.width=grid.height=64;const gctx=grid.getContext('2d'),pixels=gctx.createImageData(64,64);for(let i=0;i<4096;i++){const v=enhance?(hi>lo?(values[i]-lo)/(hi-lo):Math.max(0,Math.min(1,values[i]))):values[i],rgb=color(v);pixels.data.set([...rgb,255],i*4)}gctx.putImageData(pixels,0,0);const c=canvasByKind[kind];c.width=width;c.height=height;const ctx=c.getContext('2d');ctx.imageSmoothingEnabled=smooth;if(over&&original.complete&&original.naturalWidth){ctx.drawImage(original,0,0,width,height);ctx.globalAlpha=alpha}ctx.drawImage(grid,0,0,width,height);ctx.globalAlpha=1;const text='min '+format(lo)+' | max '+format(hi)+' | mean '+format(sum/4096)+(hi===0?'\nEMPTY / all zero':'');document.getElementById(kind+'Info').textContent=text;c.dataset.stats=text;c.title=text}}
original.onload=()=>drawMaps();
for(const [kind,c] of Object.entries(canvasByKind)){c.onmousemove=e=>{if(!current)return;const r=c.getBoundingClientRect(),x=Math.max(0,Math.min(63,Math.floor((e.clientX-r.left)/r.width*64))),y=Math.max(0,Math.min(63,Math.floor((e.clientY-r.top)/r.height*64))),v=current.arrays[kind][chosenClass()*4096+y*64+x];document.getElementById(kind+'Info').textContent=c.dataset.stats+'\ncell ('+x+', '+y+') = '+format(v)};c.onmouseleave=()=>{document.getElementById(kind+'Info').textContent=c.dataset.stats||''}}
function sync(){const att=view.value==='attention';document.getElementById('attention').classList.toggle('hidden',!att);document.getElementById('segmentation').classList.toggle('hidden',att);document.getElementById('classControl').classList.toggle('hidden',!att);if(att)loadData();else document.getElementById('sheet').src=view.value==='labelled'?'labelled_comparisons/'+imageSelect.value+'.jpg':'comparisons/'+imageSelect.value+'_'+view.value+'.jpg';const hash=new URLSearchParams({image:imageSelect.value,view:view.value,cls:String(chosenClass())});history.replaceState(null,'','#'+hash.toString())}
function move(d){imageSelect.selectedIndex=(imageSelect.selectedIndex+d+imageSelect.length)%imageSelect.length;sync()}
window.pickImage=function(id){imageSelect.value=id;sync();window.scrollTo({top:0,behavior:'smooth'})};
imageSelect.onchange=view.onchange=classSelect.onchange=sync;document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);for(const id of ['contrast','overlay','opacity','smooth'])document.getElementById(id).oninput=drawMaps;
const params=new URLSearchParams(location.hash.slice(1));if([...imageSelect.options].some(o=>o.value===params.get('image')))imageSelect.value=params.get('image');if([...view.options].some(o=>o.value===params.get('view')))view.value=params.get('view');const ci=Number(params.get('cls'));if(Number.isInteger(ci)&&ci>=0&&ci<classNames.length)classSelect.value=String(ci);sync();
</script></body></html>
'''


def main():
    rows=json.loads((OUT/'manifest.json').read_text())
    names=json.loads((OUT/'summary.json').read_text())['groups']
    payloads=OUT/'attention_data';images=OUT/'attention_images'
    payloads.mkdir(exist_ok=True);images.mkdir(exist_ok=True)
    for row in rows:
        iid=row['image_id'];p=OUT/'maps'/f'{iid}.npz'
        with np.load(p) as z:
            arrays={'ca_seed':z['a3_seed'],'sclip_seed':z['sclip_seed'],**dict(zip(KINDS[2:],z['maps']))}
            assert list(arrays)==KINDS
            encoded={}
            for key,array in arrays.items():
                assert array.shape==(10,64,64) and np.isfinite(array).all()
                little=np.ascontiguousarray(array,dtype='<f4')
                encoded[key]=base64.b64encode(little.tobytes()).decode('ascii')
                decoded=np.frombuffer(base64.b64decode(encoded[key]),dtype='<f4').reshape(array.shape)
                assert np.array_equal(decoded,array), (iid,key)
        source=DATA/row['image_file']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row['png_sha256']
        target=images/f'{iid}.png';shutil.copy2(source,target)
        payload={'image_id':iid,'width':row['generation']['width'],'height':row['generation']['height'],
                 'encoding':'base64 little-endian float32, [10,64,64], no quantization','data':encoded}
        (payloads/f'{iid}.js').write_text('window.receiveCityscapesAttention('+json.dumps(payload,separators=(',',':'))+');\n')
    colors=[(0,0,0),(128,64,128),(244,35,232),(70,70,70),(107,142,35),(70,130,180),(220,20,60),(0,0,142),(0,60,100),(0,80,100),(119,11,32)]
    replacements={
        'IMAGE_OPTIONS':''.join(f'<option value="{r["image_id"]}">{r["candidate_number"]} — {r["image_id"]}</option>' for r in rows),
        'CLASS_OPTIONS':''.join(f'<option value="{i}">{name}</option>' for i,name in enumerate(names)),
        'CLASS_LEGEND':''.join(f'<span style="background:rgb{color}">{i}: {name}</span>' for i,(color,name) in enumerate(zip(colors,['background']+names))),
        'IMAGE_CARDS':''.join(f'<article class="card"><a href="labelled_comparisons/{r["image_id"]}.jpg" target="_blank"><img loading="lazy" alt="Candidate {r["candidate_number"]}" src="labelled_comparisons/{r["image_id"]}_thumb.jpg"></a><button onclick="pickImage(\'{r["image_id"]}\')">View image {r["candidate_number"]}</button></article>' for r in rows),
        'CLASS_NAMES':json.dumps(names),'MAP_KINDS':json.dumps(KINDS)}
    html=HTML
    for key,value in replacements.items():html=html.replace(key,value)
    for name in ['index.html','labelled_gallery.html']:(OUT/name).write_text(html)
    # Offline packaging: include all assets required by every viewer mode.
    dest=OUT/'cityscapes_viewer_with_attention.zip';tmp=Path('/tmp/cityscapes_viewer_with_attention.zip')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for name in ['index.html','labelled_gallery.html']:
            archive.write(OUT/name,name)
        for directory in ['attention_data','attention_images','comparisons','labelled_comparisons']:
            for p in sorted((OUT/directory).glob('*')):
                if p.is_file():archive.write(p,Path(directory)/p.name)
    with zipfile.ZipFile(tmp) as archive:assert archive.testzip() is None
    shutil.copy2(tmp,dest)
    verification={'images':len(rows),'groups':len(names),'saved_map_kinds':KINDS,'float32_roundtrips_exact':True,
                  'source_image_hashes_verified':True,'model_inference':False,'SA_propagation_rerun':False,
                  'raw_pairwise_SA_retained':False,'archive_bytes':dest.stat().st_size}
    (OUT/'attention_viewer_validation.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(f'Built saved-map viewer for {len(rows)} images; exact float32 arrays; no inference. ZIP {dest.stat().st_size/2**20:.1f} MiB.')

if __name__=='__main__':main()
