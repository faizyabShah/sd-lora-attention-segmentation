#!/usr/bin/env python3
"""Registered windows and direct binary mechanism accounting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.segmentation import sg18_temporal as sg


def main():
    assert sg.WINDOWS=={'W/T1':(10,21),'W/T2':(10,31),'W/T3':(6,31)}
    assert [(b-a)*8 for a,b in sg.WINDOWS.values()]==[88,168,200]
    native=np.linspace(0,1,4096,dtype=np.float32)
    ca_mask,method_masks=sg.masks({'W/REF':native},native,np.zeros((64,64),bool),{'W/REF':.45})
    assert np.array_equal(ca_mask,method_masks['W/REF'])
    ca=np.zeros((32,32),bool);truth=np.zeros_like(ca);truth[2:10,2:10]=True
    ref=ca.copy();ref[2:6,2:6]=True;ref[25:30,25:30]=True
    cand=ca.copy();cand[2:8,2:8]=True;cand[25:27,25:27]=True
    rc,_=sg.events(ca,ref,truth);cc,_=sg.events(ca,cand,truth)
    z=sg.compare(ref,cand,truth,ca)
    assert rc[0]==16 and rc[1]==25 and cc[0]==36 and cc[1]==4
    assert z['tp_retained']==16 and z['tp_lost']==0 and z['tp_new']==20
    assert z['fp_retained']==4 and z['fp_removed']==21 and z['fp_new']==0
    assert z['reference_new_tp_supported']==z['tp_retained']
    assert z['reference_new_fp_supported']==z['fp_retained']
    r=sg.rates(rc)
    assert r['tp_fraction_additions']==16/41 and r['fp_separate']==25
    print('SG18 window and mechanism checks passed')

if __name__=='__main__':main()
