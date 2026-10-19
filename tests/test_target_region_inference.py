"""Independent SG13 formula, region-mask, oracle and default-CV checks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import json
import tempfile
import numpy as np
import torch
from src.segmentation.target_region_inference import region_quantities, score_regions, oracle_diagnostic, observation_statistics, expanded_labels
from src.segmentation.attention_regions import spectral_partition
from scripts.narmyn.attention.sg4_region_labels import partitions
from src.segmentation import controlled_cv as cv
from scripts.select_propagation_cv import best_threshold, run_screen
from src.segmentation.region_diagnostics import RegionDiagnosticWriter


def test_formulas_and_exact_region_boundaries():
    labels=torch.tensor([0,0,1,1])
    c=torch.tensor([.1,.9,.2,.4]); p=torch.tensor([.2,.4,.5,.7]); q=torch.tensor([.8,1.,.5,.9])
    quantities,scores,area=region_quantities(labels,c,p,q)
    torch.testing.assert_close(quantities,torch.tensor([[.5,.3,.9,.9],[.3,.6,.7,.4]]))
    torch.testing.assert_close(scores,torch.tensor([[.3,.6],[.27,.42],[.585,.41]]))
    truth=np.indices((5,7))[0]%2==0
    thresholds=(.3,.45,.6)
    counts=score_regions(labels,scores,truth,thresholds)
    identity=expanded_labels(labels,truth.shape)
    for j,t in enumerate(thresholds):
        masks=(scores[:,identity]>=t).numpy()
        np.testing.assert_array_equal(counts[:,j],cv.score_masks(masks,truth,thresholds=(t,))[:,0])
    fraction,oracle=oracle_diagnostic(labels,np.array([[1,0],[1,1]],bool))
    torch.testing.assert_close(fraction,torch.tensor([.5,1.]))
    assert oracle.tolist()==[3,1,0,0]


def test_partition_and_streaming_reference():
    generator=torch.Generator().manual_seed(88)
    mean=torch.rand(40,40,generator=generator).softmax(-1)
    torch.testing.assert_close(spectral_partition(mean,32),partitions(mean)['SPEC32'])
    calls=[(torch.randn(2,4,3,generator=generator),torch.randn(2,4,3,generator=generator)) for _ in range(3)]
    seed=torch.rand(4,generator=generator)
    graph,mean,twice,consensus=observation_statistics(calls,seed)
    matrices=torch.cat([(q@k.transpose(-1,-2)/3**.5).softmax(-1) for q,k in calls])
    z=matrices@seed
    torch.testing.assert_close(graph,matrices.mean(0))
    torch.testing.assert_close(mean,z.mean(0))
    torch.testing.assert_close(twice,(matrices@z[...,None])[...,0].mean(0))
    torch.testing.assert_close(consensus,torch.maximum((z>seed).float().mean(0),(z<seed).float().mean(0)))


def test_optional_grid_preserves_default_and_respects_restriction():
    # Single class/five images: candidate/reference both best at .1 if unrestricted,
    # but the registered control grid permits .45 only.
    counts=np.tile(np.array([[[[9,1,1,9],[5,5,5,5]]]],np.int64),(2,5,1,1))
    result=cv.Results(('ref','candidate'),(.1,.45),tuple(map(str,range(5))),('x',)*5,np.arange(5),counts,np.zeros((2,5)),{})
    mask=np.ones(5,bool)
    assert best_threshold(result,0,mask)[0]==0
    assert best_threshold(result,0,mask,[1])[0]==1
    a=run_screen(result,'ref',['candidate'],{})
    b=run_screen(result,'ref',['candidate'],{},threshold_indices=None)
    assert a==b
    c=run_screen(result,'ref',['candidate'],{},threshold_indices={'ref':[1],'candidate':[0,1]})
    assert c['frozen']=='candidate' and all(f['reference_threshold']==.45 for f in c['folds'])


def test_resume_sidecar():
    spec=json.loads((ROOT/'configs/sg13_target_region_inference.json').read_text())
    labels=torch.tensor([0,0,1,1]); seed=torch.tensor([.1,.9,.2,.4])
    q,s,a=region_quantities(labels,seed,seed,seed)
    f,o=oracle_diagnostic(labels,np.array([[1,0],[1,1]],bool))
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'regions.npz'
        writer=RegionDiagnosticWriter(path,spec)
        writer.add('a',q,s,a,f,o)
        writer=RegionDiagnosticWriter(path,spec)
        writer.add('a',q,s,a,f,o)
        assert len(writer.records)==1
        np.testing.assert_array_equal(writer.records['a'][2][:2],[2,2])


if __name__=='__main__':
    for name,check in sorted(list(globals().items())):
        if name.startswith('test_'):
            print(name,flush=True);check()
    print('SG13 formula, partition, masks, oracle, CV grid and resume checks passed',flush=True)
