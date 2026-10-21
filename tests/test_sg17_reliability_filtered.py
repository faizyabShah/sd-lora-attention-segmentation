#!/usr/bin/env python3
"""Algebra and accounting checks for the locked SG17 gate."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.segmentation import sg17_reliability as sg


def main():
    hist=np.ones(249,np.int64);hist[0]=0
    cut=sg.quantile_cutoffs(hist)
    assert cut['Q70']['count_cutoff']<=cut['Q80']['count_cutoff']<=cut['Q90']['count_cutoff']
    assert all(0<v['actual_retained_fraction']<=1 for v in cut.values())
    tied=np.zeros(249,np.int64);tied[248]=100
    assert all(v['count_cutoff']==248 and v['actual_retained_fraction']==1
               for v in sg.quantile_cutoffs(tied).values())
    one=torch.zeros(4096);two=torch.ones(4096);two[0]=-1
    count=torch.zeros(4096,dtype=torch.uint8);count[100:]=248
    maps=sg.candidate_maps(one,two,count,cut)
    assert maps.shape==(4,64,64)
    assert torch.equal(maps[0].reshape(-1),two)
    assert all(maps[i].reshape(-1)[0]==-1 for i in (1,2,3))
    assert all(maps[i].reshape(-1)[1]==0 for i in (1,2,3))
    assert all(maps[i].reshape(-1)[100]==1 for i in (1,2,3))
    h=sg.native_positive_hist(one.numpy(),two.numpy(),count.numpy())
    assert h.sum()==4095 and h[0]==99 and h[248]==3996
    truth=np.zeros((64,64),bool);truth[:8,:8]=True
    before=np.zeros_like(truth);p1=np.zeros_like(truth);p1[:8,:8]=True;p1[10:18,10:18]=True
    cand=p1.copy();cand[10:18,10:18]=False
    q=np.full((64,64),200,np.uint8)
    m=sg.mechanism(before,p1,cand,truth,q,210,np.ones((64,64)))
    assert m['retained']['tp']==64 and m['rejected']['fp']==64
    assert m['rejected']['tp']==0 and m['retained']['fp']==0
    bins=sg.reliability_bins(before,p1,truth,q,[0,50,75,100,125,150,175,200,225,240,249])
    assert bins.sum(0)[0]==128 and bins.sum(0)[1]==64 and bins.sum(0)[2]==64
    print('SG17 algebra and mechanism checks passed')

if __name__=='__main__':main()
