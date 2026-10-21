"""Check oracle majority, frozen A050/B identities and sheet rendering."""
from pathlib import Path
import sys
import tempfile
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_sg15_label_oracle_20 import compare,selected_rows
from src.segmentation import locked_refinements as lr


def main():
    selected=selected_rows()
    assert len(selected)==20 and len({classname for classname,_ in selected})==20
    labels=np.repeat(np.arange(4)[None,:],4,axis=0)
    gt=np.array([[1,1,1,0],[1,1,1,0],[1,0,1,1],[0,0,1,1]],bool)
    prior=np.array([[1,1,1,0],[0,1,1,0],[0,0,1,1],[0,0,0,1]],bool)
    current,snaps,oracle,rows=compare(labels,gt,np.stack([prior]*3))
    gt_fraction,area=lr.region_fractions(labels,gt)
    np.testing.assert_array_equal(oracle,(gt_fraction>=.5)[labels])
    np.testing.assert_array_equal(current[0],lr.refine_regions(labels,prior,'A050')[0])
    np.testing.assert_array_equal(snaps[0],lr.refine_regions(labels,prior,'B')[0])
    assert np.array_equal(current[0],current[1]) and rows[0]['oracle_iou']==rows[1]['oracle_iou']
    assert rows[0]['A050_oracle_disagreement_pixels']==np.count_nonzero(current[0]!=oracle)
    assert rows[0]['A050_oracle_false_positive_regions']==0
    # One independent CPU rendering check protects the requested visual.
    image=np.repeat(np.stack([gt.astype(np.uint8)*180]*3,axis=-1),1,axis=0)
    with tempfile.TemporaryDirectory(prefix='sg15_oracle_plot_') as tmp:
        cache=Path(tmp)/'sample.npz';fig=Path(tmp)/'sheet.png'
        np.savez_compressed(cache,image=image,truth=gt,labels=labels,prior=np.stack([prior]*3),
                            majority=current,snapped=snaps,oracle=oracle,
                            names=np.asarray(lr.PRIORS),thresholds=np.array([.45,.425,.45]),
                            image_id=np.array('synthetic'),classname=np.array('example'))
        subprocess.run(['/home/shafa/miniconda3/envs/ovam_sdxl/bin/python',str(ROOT/'scripts/plot_sg15_label_oracle_20.py'),
                        '--cache',str(cache),'--output',str(fig)],check=True)
        assert fig.stat().st_size>1000
    print('SG15 A050/B, oracle and nine-column visual checks passed')

if __name__=='__main__':main()
