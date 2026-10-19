#!/usr/bin/env python3
"""Plot compact SG12 aggregates using a CPU environment with Matplotlib.

Kept separate so GPU inference keeps its validated model environment.
"""
import argparse
import os
import tempfile
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / f'sd_attention_mpl_{os.getuid()}'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir/'diagnostics'
    for dataset in ('voc', 'coco'):
        with np.load(output/f'{dataset}_aggregates.npz', allow_pickle=False) as data:
            avg = data['reliability_mean']
            frequency = data['selected_frequency']
            pairwise = data['pairwise_agreement']
            hist, edges = data['reliability_hist_counts'], data['reliability_hist_edges']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].bar(edges[:-1], hist, width=np.diff(edges), align='edge')
        axes[0].set(xlabel='Mean cosine reliability', ylabel='Observations', title=f'{dataset}: all image/query scores')
        axes[1].plot(np.arange(8), avg.reshape(31, 8).mean(0), 'o-')
        axes[1].set(xlabel='SA head', ylabel='Mean reliability')
        axes[2].plot(np.arange(31), avg.reshape(31, 8).mean(1), 'o-')
        axes[2].set(xlabel='Denoising call (zero based)', ylabel='Mean reliability')
        fig.tight_layout(); fig.savefig(output/f'{dataset}_reliability.png', dpi=160); plt.close(fig)
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        for j, (axis, fraction) in enumerate(zip(axes, (.5, .7, .8, .9))):
            im = axis.imshow(frequency[j].reshape(31, 8), vmin=0, vmax=1, aspect='auto')
            axis.set(title=f'Top {round(100*fraction)}%', xlabel='Head', ylabel='Call')
        fig.colorbar(im, ax=list(axes), label='Selection frequency')
        fig.savefig(output/f'{dataset}_selection_frequency.png', dpi=160, bbox_inches='tight'); plt.close(fig)
        fig, axis = plt.subplots(figsize=(7, 6))
        im = axis.imshow(pairwise, vmin=0, vmax=1)
        axis.set(xlabel='Observation (8*call + head)', ylabel='Observation', title=f'{dataset}: mean pairwise cosine')
        fig.colorbar(im, ax=axis); fig.tight_layout()
        fig.savefig(output/f'{dataset}_pairwise_agreement.png', dpi=160); plt.close(fig)
    print('Observation reliability, selection frequency and pairwise plots written')


if __name__ == '__main__':
    main()
