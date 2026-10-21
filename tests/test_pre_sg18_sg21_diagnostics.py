#!/usr/bin/env python3
"""Small algebra and accounting checks for temporal/head diagnostics."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.segmentation import temporal_head_diagnostics as d


def main():
    c=torch.full((4096,),.4)
    x=torch.full((31,8,4096),.4)
    x[:10]+=torch.arange(10).reshape(10,1,1)*.01
    x[0]=.39
    x[10:21]-=.1
    x[21:]+=.2
    f=d.temporal_features(x,c)
    assert len(f)==len(d.FEATURES)
    assert abs(f['time_up_fraction'][0].item()-19/31)<1e-6
    assert f['longest_up_run'][0]==10
    assert f['longest_down_run'][0]==11
    assert f['sign_change_count'][0]==3
    assert abs(f['late_minus_early'][0].item()-.156)<1e-5
    truth=np.zeros((8,8),bool);truth[:4,:4]=True
    ca=np.zeros((8,8),bool);after=ca.copy();after[1,1]=True;after[6,6]=True
    e=d.event_context(ca,after,truth)
    counts=d.event_counts(ca,after,truth)
    assert counts[0]==1 and counts[1]==1
    assert counts[4]==1 and e['tp'].sum()==1
    h=d.feature_histograms({name:torch.full((4096,),sum(bounds)/2)
                            for name,bounds in d.FEATURE_BOUNDS.items()},e,truth.shape,64)
    assert h.shape==(5,2,len(d.FEATURES),64)
    assert h[0,0,0].sum()==1 and h[0,1,0].sum()==1
    print('Pre-SG18/SG21 algebra and event-accounting checks passed')

if __name__=='__main__':main()
