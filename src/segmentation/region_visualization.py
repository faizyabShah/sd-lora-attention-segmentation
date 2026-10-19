"""SG13 eight-panel sheets with fixed score scales and exact region masks."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.segmentation.sg_visual_audit import _overlay, _mask_rgb, _iou


def render_region_sheets(directory: Path,spec,summary):
    visual=directory/'visuals';visual.mkdir(exist_ok=True)
    caches=sorted((directory/'visual_cache').glob('*.npz'))
    if not caches and (visual/'index.json').exists():
        old=json.loads((visual/'index.json').read_text())
        if {r['class'] for r in old}==set(spec['visual_classes']) and all((visual/r['file']).exists() for r in old):return
    records=[]
    for path in caches:
        with np.load(path,allow_pickle=False) as data:
            d={k:data[k] for k in data.files}
        classname,image_id=str(d['classname']),str(d['image_id'])
        truth=d['target'].astype(bool);height,width=truth.shape
        original=np.asarray(Image.fromarray(d['image']).resize((width,height)))
        labels=d['labels'].reshape(64,64)
        full=np.asarray(Image.fromarray(labels.astype(np.uint8)).resize((width,height),Image.Resampling.NEAREST))
        rng=np.random.default_rng(88);colors=rng.integers(30,240,(32,3),dtype=np.uint8)
        partition=(.35*original+.65*colors[full]).astype(np.uint8)
        edge=np.zeros_like(full,dtype=bool)
        edge[1:]|=full[1:]!=full[:-1];edge[:,1:]|=full[:,1:]!=full[:,:-1]
        partition[edge]=0
        def heat(value):
            array=np.asarray(value,dtype=np.float32).reshape(64,64)
            return np.asarray(Image.fromarray(array).resize((width,height),Image.Resampling.BILINEAR))
        ca=_overlay(original,heat(d['seed']))
        propagated=_overlay(original,heat(d['propagated']))
        consensus=_overlay(original,heat(d['consensus']))
        oracle=d['oracle_region_positive'][full]
        oracle_image=_mask_rgb(oracle,truth)
        try:font=ImageFont.truetype('DejaVuSans.ttf',13)
        except OSError:font=ImageFont.load_default()
        tile=220;rowheight=280
        canvas=Image.new('RGB',(8*tile,40+3*rowheight),'white');draw=ImageDraw.Draw(canvas)
        draw.text((6,8),f'SG13 / SPEC32 / {classname} / {image_id} / all heatmaps use fixed 0-1 scale',font=font,fill='black')
        for r in range(3):
            name=f'W/SG13/R{r+1}'
            threshold=summary['cells'][name]['voc_threshold']
            score=d['scores'][r]
            predicted=score[full]>=threshold
            panels=[('Image',original),('Fixed SPEC32 partition',partition),('CA seed',ca),('Raw propagated CA',propagated),
                    ('SG8 consensus',consensus),('Raw region score',_overlay(original,score[full])),
                    (f'Pred t={threshold:.3f}, IoU={_iou(predicted,truth):.3f}',_mask_rgb(predicted,truth)),
                    ('Oracle regions; GT cyan',oracle_image)]
            y=40+r*rowheight
            draw.text((5,y),name,font=font,fill='black')
            for col,(label,panel) in enumerate(panels):
                x=col*tile
                draw.text((x+4,y+22),label,font=font,fill='black')
                canvas.paste(Image.fromarray(panel).resize((tile,tile)),(x,y+45))
        filename=f"{classname.replace(' ','_')}_sg13.jpg"
        canvas.save(visual/filename,quality=93)
        records.append({'class':classname,'image_id':image_id,'file':filename,'sampling':spec['visual_sampling']})
    if {r['class'] for r in records}!=set(spec['visual_classes']):raise ValueError('missing fixed visual examples')
    (visual/'index.json').write_text(json.dumps(records,indent=2)+'\n')
    for path in caches:path.unlink()
