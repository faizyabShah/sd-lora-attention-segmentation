#!/usr/bin/env python3
"""Small synthetic invariants for the two prerequisite diagnostics."""
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.segmentation.pre_sg_diagnostics import (
    region_records,pixel_histograms,PIXEL_FEATURES,ENDPOINTS)
from scripts.finalize_pre_sg16_sg17_diagnostics import auc_exact,deciles

def main():
    labels=np.array([0,0,1,1]);full=labels.reshape(2,2)
    truth=np.array([[1,1],[0,0]],bool)
    prior=np.stack((truth,truth,np.zeros_like(truth)))
    seed=np.array([.8,.7,.1,.2]);first=np.tile(seed,(248,1))
    records,image=region_records(labels,full,truth,prior,seed,first,np.arange(124))
    assert len(records)==2 and [r['oracle'] for r in records]==[1,0]
    assert records[0]['p1_occupancy']==1 and records[1]['p1_occupancy']==0
    assert image['oracle_gain_vs_p1']==0
    assert auc_exact([0,1,0,1],[0,1,0,1])['roc_auc']==1
    assert auc_exact([0,0],[.1,.2]) is None
    native=np.zeros((64,64),np.float32);native[:,32:]=1
    one=torch.from_numpy(native.reshape(-1));two=1-one
    features={name:torch.zeros(4096) for name in PIXEL_FEATURES if not name.endswith('_direction')}
    for name in ('support_reliability_up','support_reliability_down',
                 'top50_support_up','top50_support_down',
                 'second_q_up','second_q_down','second_median','second_trim10',
                 'top50_median','top50_trim10'):
        features[name]=torch.zeros(4096)
    gt=np.zeros((64,64),bool);gt[:,:32]=True
    hist,image=pixel_histograms(features,one,two,gt,bins=32)
    assert hist.shape==(len(ENDPOINTS),2,len(PIXEL_FEATURES),32)
    assert image['useful']==4096 and image['harmful']==0
    assert image['two_pass_iou_gain']==1
    assert hist[0,0,0].sum()==4096
    rows,edges=deciles(hist,'second_q_direction')
    assert len(rows)==10 and len(edges)==11
    print('Pre-SG16/17 synthetic diagnostics passed')

if __name__=='__main__':main()
