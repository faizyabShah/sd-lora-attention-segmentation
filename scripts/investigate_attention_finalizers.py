#!/usr/bin/env python3
"""Saved-count, diagnostic and provenance investigation for completed attention jobs."""
import json
import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.segmentation import controlled_cv as cv
from scripts.select_propagation_cv import pooled_miou
base=ROOT/'outputs/segmentation/controlled_attention_cv'

def load(family,ds):
 return cv.load_results([p for p in sorted((base/family/ds/'shards').glob('*.npz')) if '.diagnostics.' not in p.name and '.regions.' not in p.name])

def compare(a,an,b,bn):
 order=[b.image_ids.index(n) for n in a.image_ids]
 x=a.counts[a.index(an)]; y=b.counts[b.index(bn),order]
 indices=[b.thresholds.index(t) for t in a.thresholds];y=y[:,indices]
 delta=np.abs(x-y); fractions=delta/np.maximum(x.sum(-1)[...,None],1)
 worst=np.unravel_index(fractions.argmax(),fractions.shape)
 per_image=fractions.max((1,2))
 t=a.thresholds.index(.45)
 return {'images':len(order),'max_pixels':int(delta.max()),'max_fraction':float(fractions.max()),'worst_image':a.image_ids[worst[0]],'worst_class':a.classes[worst[0]],'worst_threshold':a.thresholds[worst[1]],'exact_images':int((per_image==0).sum()),'above_0.002':int((per_image>.002).sum()),'fraction_quantiles':np.quantile(per_image,[.5,.9,.95,.99,1]).tolist(),'miou_0.45_new':pooled_miou(x[:,t],a.classes),'miou_0.45_old':pooled_miou(y[:,t],a.classes),'worst_new_counts':x[worst[0],worst[1]].tolist(),'worst_old_counts':y[worst[0],worst[1]].tolist()}


def main():
 parser=argparse.ArgumentParser(description="Audit complete saved SG11/SG12 and SG2+CRF shards without changing measurements")
 parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/segmentation/analysis/finalizer_reference_audit')
 args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
 comparisons={};contracts={};sidecars={};hashes={}
 for ds in ('voc','coco'):
  new=load('sg11_sg12_observation_reliability',ds);sg8=load('sg8_observation_consensus',ds);dense=load('sg2_densecrf_confirmation',ds);sg2=load('sg2_consensus_edges',ds);sg13=load('sg13_target_region_inference',ds)
  comparisons[ds]={}
  for label,a,an,b,bn in [('SG11_REF_vs_SG8',new,'W/REF',sg8,'W/REF'),('SG11_C_vs_SG8_C',new,'W/C/rho0p80',sg8,'W/C/rho0p80'),('Dense_REF_vs_SG2',dense,'W/REF',sg2,'W/REF'),('Dense_REF_vs_SG11',dense,'W/REF',new,'W/REF'),('Dense_SG2_vs_oldSG2',dense,'W/SG2',sg2,'W/G_cv_g20_n1'),('A100_SG8_vs_SG13_REF',sg8,'W/REF',sg13,'W/REF')]:
   comparisons[ds][label]=compare(a,an,b,bn)
  assert comparisons[ds]['Dense_REF_vs_SG11']['max_fraction']<=.002
  assert comparisons[ds]['A100_SG8_vs_SG13_REF']['max_fraction']<=.002
  for label,r in [('SG11',new),('Dense',dense)]:
   assert len(r.complete)==20 and all(r.complete)
   assert len(set(r.image_ids))==(490 if ds=='voc' else 707)
   assert set(r.image_ids)==set(sg8.image_ids)
   if ds=='voc':
    lookup={n:i for i,n in enumerate(sg8.image_ids)}
    assert np.array_equal(r.folds,sg8.folds[[lookup[n] for n in r.image_ids]])
   c=r.counts
   assert (c>=0).all()
   assert np.all(c.sum(-1)==c.sum(-1)[0,:,0][None,:,None])
   assert np.all((c[...,0]+c[...,2])==(c[0,:,0,0]+c[0,:,0,2])[None,:,None])
   raw=c[:14 if label=='SG11' else 3]
   assert np.all(np.diff(raw[...,0]+raw[...,1],axis=-1)<=0)
   assert np.isfinite(r.ap[:14 if label=='SG11' else 3]).all()
   contracts[f'{label}/{ds}']={'images':len(r.image_ids),'complete_shards':20,'count_invariants':'passed','folds':'matched','environments':r.environments}
  n=0
  for path in sorted((base/'sg11_sg12_observation_reliability'/ds/'shards').glob('*.diagnostics.npz')):
   with np.load(path,allow_pickle=False) as d:
    n+=len(d['image_ids']);scores=d['reliability'];selected=d['selected'];pair=d['pairwise_sum']
    assert np.isfinite(scores).all() and np.isfinite(pair).all()
    assert scores.min()>=-1e-6 and scores.max()<=1.00001
    assert np.allclose(pair,pair.T,atol=1e-5)
    assert np.all(d['histogram'].sum(-1)==d['histogram'].sum(-1)[...,:1])
    assert np.array_equal(d['histogram'][:,:,0].sum((1,2)),d['event_totals'][:,2])
    ranks=np.argsort(-scores,axis=1,kind='stable')
    for j,k in enumerate((124,174,199,224)):
     assert np.all(selected[:,j].sum(1)==k)
     expected=np.zeros_like(selected[:,j]);np.put_along_axis(expected,ranks[:,:k],True,axis=1)
     assert np.array_equal(expected,selected[:,j])
    avg=(pair.sum(1)-np.diag(pair))/(247*len(scores))
    assert np.allclose(avg,scores.mean(0),atol=3e-6)
   # Read both files together, independently of checkpoint ordering.
   with np.load(path,allow_pickle=False) as diag,np.load(path.with_name(path.name.replace('.diagnostics','')),allow_pickle=False) as metric:
    assert set(diag['image_ids'])==set(metric['image_ids'])
  assert n==len(new.image_ids)
  sidecars[ds]={'images':n,'rankings_keep_counts_pairwise_and_histogram_identities':'passed'}
 for family in ('sg11_sg12_observation_reliability','sg2_densecrf_confirmation'):
  hashes[family]={}
  for path in sorted((base/family).glob('*/shards/*.npz')):
   hashes[family][str(path.relative_to(base/family))]=hashlib.sha256(path.read_bytes()).hexdigest()
 output={'status':'CPU structural and compatible-reference checks passed; GPU reproduction audit must also pass before recovery',
         'comparisons':comparisons,'contracts':contracts,'sg11_sidecars':sidecars,'input_sha256':hashes}
 (args.output_dir/'count_audit.json').write_text(json.dumps(output,indent=2)+'\n')
 print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
