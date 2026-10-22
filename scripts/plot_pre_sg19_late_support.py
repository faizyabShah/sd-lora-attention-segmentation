#!/usr/bin/env python3
"""Plot VOC-defined late-support bin ratios in the plotting environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=json.loads(args.results.read_text())
    selected=result['selected_bin_features_voc_only']
    bins=result['reliability_bins']
    fig,axes=plt.subplots(1,len(selected),figsize=(5*len(selected),4),sharey=False)
    if len(selected)==1:axes=[axes]
    for ax,name in zip(axes,selected):
        for ds,color in [('voc','tab:blue'),('coco','tab:orange')]:
            values=bins[ds][name]
            ratio=[v['tp_fp_ratio'] if v['tp_fp_ratio'] is not None else np.nan for v in values]
            ax.plot(range(1,len(ratio)+1),ratio,'o-',label=ds.upper(),color=color)
        ax.set(title=name,xlabel='VOC-defined late-support bin',ylabel='TP / FP')
        ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=170);plt.close(fig)


if __name__=='__main__':main()
